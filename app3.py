import streamlit as st
import whisper
import os
import tempfile
import textwrap
import google.generativeai as genai
import json
import subprocess
import random
import nest_asyncio
import pandas as pd
import time
import firebase_admin
from firebase_admin import credentials, firestore
from fpdf import FPDF
from openai import OpenAI 

# --- AYARLAR ---
st.set_page_config(page_title="Gemini Eğitim Platformu (Cloud)", layout="wide")
nest_asyncio.apply()

# --- API KEYLER ---
gemini_api_key = st.secrets["gemini_key"]
openai_api_key = st.secrets["openai_key"]
ADMIN_PASSWORD = st.secrets["admin_password"]

# --- FIREBASE BAĞLANTISI (KESİN ÇÖZÜM) ---
if not firebase_admin._apps:
    try:
        # Secrets'tan veriyi al
        key_dict = dict(st.secrets["firebase"])
        
        # 🔥 "\n" yazılarını gerçek enter tuşuna çevirir (PEM Hatası Önleyici)
        key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")
        
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebase Hatası: {e}")
        st.stop()

db = firestore.client()

# --- API BAĞLANTILARI ---
try:
    genai.configure(api_key=gemini_api_key)
    client = OpenAI(api_key=openai_api_key)
except: pass

# --- STATE YÖNETİMİ ---
def init_state():
    defaults = {
        'step': 0, 
        'user_role': None, 
        'student_info': {},
        'scores': {'pre': 0, 'post': 0},
        'pre_answers': {},
        'user_answers_post': {},
        'exam_finished': False,
        'data': [],
        'mistakes': [],
        'audio_speed': 1.0 
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# --- FIREBASE FONKSİYONLARI ---
def save_results_to_firebase(student_data):
    try:
        doc_ref = db.collection('exam_results').document(student_data['no'])
        doc_ref.set(student_data)
        return True
    except Exception as e:
        st.error(f"Veritabanı Hatası: {e}")
        return False

def get_class_data_from_firebase():
    docs = db.collection('exam_results').stream()
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    return data

# --- WHISPER & AI FONKSİYONLARI ---

@st.cache_resource
def load_whisper():
    # 🔥 RAM TASARRUFU İÇİN 'tiny' MODEL VE CPU AYARI
    return whisper.load_model("tiny", device="cpu")

def sesi_sokup_al(video_path, audio_path):
    # FFMPEG komutu
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-q:a", "9", "-y", audio_path]
    try: 
        subprocess.run(command, capture_output=True, text=True)
        return True
    except: 
        return False

def analyze_full_text_with_gemini(full_text):
    # GÜNCELLEME: Daha stabil olan 1.5-flash modeline geçildi
    model = genai.GenerativeModel('gemini-1.5-flash') 
    
    # DEBUG: Whisper'ın ne duyduğunu ekrana yazdıralım
    st.info(f"🕵️ DEBUG: Whisper {len(full_text)} karakterlik metin çıkardı.")
    
    if len(full_text) < 50:
        st.warning(f"⚠️ UYARI: Çıkarılan metin çok kısa! Muhtemelen ses anlaşılmadı veya ffmpeg çalışmadı. Metin: '{full_text}'")
        return []

    prompt = f"""GÖREV: Aşağıdaki metni eğitim materyaline dönüştür. 
    Çıktı SADECE geçerli bir JSON formatında olmalı.
    
    İstenen JSON Yapısı (Liste içinde objeler):
    [
      {{
        "alt_baslik": "Konu Başlığı",
        "ozet": "Kısa ve net özet.",
        "soru_data": {{
            "soru": "Konuyla ilgili çoktan seçmeli soru?",
            "A": "Seçenek A",
            "B": "Seçenek B",
            "C": "Seçenek C",
            "D": "Seçenek D",
            "dogru_sik": "A"
        }}
      }}
    ]

    METİN: "{full_text}"
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        start = text.find('[')
        end = text.rfind(']') + 1
        return json.loads(text[start:end])
    except Exception as e:
        st.error(f"🚨 GEMINI HATASI: {e}")
        return []

def generate_audio_openai(text, speed):
    if not client or len(text) < 2: return None
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tfile.close()
    try:
        response = client.audio.speech.create(model="tts-1", voice="alloy", input=text, speed=speed)
        response.stream_to_file(tfile.name)
        return tfile.name
    except: return None
    
# --- PDF FONKSİYONU ---
def create_study_pdf(data, mistakes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "CALISMA PLANI", ln=1, align='C')
    
    pdf.set_font("Arial", '', 12)
    for i in mistakes:
        item = data[i]
        # Türkçe karakter sorunu olmaması için latin-1 replace kullanıyoruz
        baslik = item.get('alt_baslik', 'Konu').encode('latin-1', 'replace').decode('latin-1')
        ozet = item.get('ozet', '').encode('latin-1', 'replace').decode('latin-1')
        
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 10, f"KONU: {baslik}")
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, ozet)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= ARAYÜZ =================

st.title("☁️ Gemini Eğitim Platformu (Online)")

LESSON_FILE = "lesson_data.json"

# Eğer ders dosyası varsa ve state boşsa yükle
if os.path.exists(LESSON_FILE) and not st.session_state['data']:
    try:
        with open(LESSON_FILE, 'r', encoding='utf-8') as f:
            st.session_state['data'] = json.load(f)
    except:
        pass # Dosya bozuksa geç

# --- GİRİŞ EKRANI ---
if st.session_state['step'] == 0:
    tab1, tab2 = st.tabs(["👨‍🎓 Öğrenci Girişi", "👨‍🏫 Öğretmen Paneli"])
    
    with tab1:
        st.subheader("Öğrenci Girişi")
        s_name = st.text_input("Ad Soyad")
        s_no = st.text_input("Öğrenci No")
        
        if st.button("Sınava Başla"):
            if s_name and s_no:
                # Veri kontrolü
                if not st.session_state['data']:
                    st.error("Sistemde yüklü ders yok! Lütfen öğretmenin dersi yüklemesini bekleyin.")
                else:
                    st.session_state['student_info'] = {'name': s_name, 'no': s_no}
                    st.session_state['user_role'] = 'student'
                    st.session_state['step'] = 2 
                    st.rerun()
            else:
                st.warning("Bilgileri doldurunuz.")

    with tab2:
        st.subheader("Öğretmen Girişi")
        password = st.text_input("Yönetici Şifresi", type="password")
        if st.button("Yönetici Girişi"):
            if password == ADMIN_PASSWORD:
                st.session_state['user_role'] = 'admin'
                st.session_state['step'] = 1
                st.rerun()
            else:
                st.error("Hatalı Şifre")

# --- ADIM 1: ÖĞRETMEN PANELİ ---
elif st.session_state['step'] == 1 and st.session_state['user_role'] == 'admin':
    st.header("👨‍🏫 Öğretmen Kontrol Paneli")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("1. Yeni Ders Yükle")
        up = st.file_uploader("Ders Videosu Seç (.mp4)", type=["mp4"])
        
        if up and st.button("Videoyu İşle ve Yayına Al"):
            with st.spinner("Video işleniyor... (Tiny model kullanılıyor)"):
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(up.read())
                    
                    audio_path = tfile.name.replace(".mp4", ".mp3")
                    
                    # 1. Ses Ayırma
                    basari = sesi_sokup_al(tfile.name, audio_path)
                    if not basari:
                        st.error("FFMPEG Hatası: Ses ayrıştırılamadı. packages.txt dosyasını kontrol et.")
                        st.stop()
                    
                    # 2. Transkripsiyon (Tiny Model)
                    model_w = load_whisper()
                    result = model_w.transcribe(audio_path)
                    full_text = result['text']
                    
                    # 3. Gemini Analiz (DEBUG modunda çalışacak)
                    analysis = analyze_full_text_with_gemini(full_text)
                    
                    if analysis and len(analysis) > 0:
                        with open(LESSON_FILE, 'w', encoding='utf-8') as f:
                            json.dump(analysis, f, ensure_ascii=False)
                        st.session_state['data'] = analysis
                        st.success("✅ Ders başarıyla işlendi ve yayına alındı!")
                    else:
                        st.error("Gemini analizi başarısız oldu (Yukarıdaki hata detayına bakın).")
                except Exception as e:
                    st.error(f"Bir hata oluştu: {e}")

    with col_b:
        st.subheader("2. Sınıf Raporları")
        if st.button("Sonuçları Getir"):
            class_data = get_class_data_from_firebase()
            if class_data:
                df = pd.DataFrame(class_data)
                st.dataframe(df)
                csv = df.to_csv(index=False, sep=';').encode('utf-8-sig')
                st.download_button("Excel/CSV İndir", csv, "sinif_raporu.csv")
            else:
                st.info("Kayıt bulunamadı.")

# --- ADIM 2: ÖĞRENCİ - ÖN TEST ---
elif st.session_state['step'] == 2:
    if not st.session_state['data']:
        st.warning("⚠️ Ders içeriği yüklenemedi. Öğretmeninizle görüşün.")
        if st.button("Yenile"):
            st.rerun()
    else:
        st.info(f"Hoşgeldin, **{st.session_state['student_info']['name']}**. Test başlıyor.")
        st.subheader("📝 Ön Bilgi Testi")
        
        with st.form("pre_test_form"):
            ans = {}
            for i, item in enumerate(st.session_state['data']):
                q = item.get('soru_data', {})
                soru_metni = q.get('soru', 'Soru yüklenemedi')
                
                st.write(f"**{i+1}.** {soru_metni}")
                
                secenekler = [
                    q.get('A', 'A'), 
                    q.get('B', 'B'), 
                    q.get('C', 'C'), 
                    q.get('D', 'D')
                ]
                
                ans[i] = st.radio("Cevap", secenekler, key=f"p_{i}", index=None)
                st.markdown("---")
            
            if st.form_submit_button("Testi Bitir"):
                score = 0
                mistakes = []
                
                for i, item in enumerate(st.session_state['data']):
                    q = item.get('soru_data', {})
                    dogru_harf = q.get('dogru_sik', 'A').strip().upper()
                    dogru_metin = q.get(dogru_harf)
                    
                    verilen_cevap = ans.get(i)
                    
                    if verilen_cevap and verilen_cevap == dogru_metin:
                        score += 1
                    else:
                        mistakes.append(i)
                
                st.session_state['scores']['pre'] = score
                st.session_state['mistakes'] = mistakes
                st.session_state['step'] = 3
                st.rerun()

# --- ADIM 3: ÇALIŞMA EKRANI ---
elif st.session_state['step'] == 3:
    st.success(f"Ön Test Puanın: {st.session_state['scores']['pre']}")
    
    if st.session_state['mistakes']:
        st.warning("Eksik konular aşağıda listelenmiştir. Lütfen çalışın.")
        
        if st.button("📄 Eksik Konuları PDF Olarak İndir"):
            pdf_bytes = create_study_pdf(st.session_state['data'], st.session_state['mistakes'])
            st.download_button(label="Çalışma Planını İndir", 
                               data=pdf_bytes, 
                               file_name="calisma_plani.pdf", 
                               mime='application/pdf')

    if st.button("Son Sınava Geç ->"):
        st.session_state['step'] = 4
        st.rerun()
        
    st.markdown("---")
    
    for i, item in enumerate(st.session_state['data']):
        if i in st.session_state['mistakes']:
            st.error(f"Eksik Konu: {item.get('alt_baslik', 'Konu')}")
            ozet_metni = item.get('ozet', 'Özet yok')
            st.write(ozet_metni)
            
            if st.button("🔊 Dinle", key=f"listen_{i}"):
                with st.spinner("Ses oluşturuluyor..."):
                    path = generate_audio_openai(ozet_metni, 1.0)
                    if path:
                        st.audio(path)
                    else:
                        st.error("Ses oluşturulamadı.")
            st.markdown("---")

# --- ADIM 4: SON TEST ---
elif st.session_state['step'] == 4:
    st.subheader("🎓 Son Değerlendirme")
    
    with st.form("post_test_form"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item.get('soru_data', {})
            st.write(f"**{i+1}.** {q.get('soru', '')}")
            
            secenekler = [q.get('A'), q.get('B'), q.get('C'), q.get('D')]
            ans[i] = st.radio("Cevap", secenekler, key=f"last_{i}", index=None)
            st.markdown("---")
        
        if st.form_submit_button("Sınavı Tamamla ve Kaydet"):
            score = 0
            for i, item in enumerate(st.session_state['data']):
                q = item.get('soru_data', {})
                dogru_harf = q.get('dogru_sik', 'A').strip().upper()
                dogru_metin = q.get(dogru_harf)
                
                if ans.get(i) == dogru_metin:
                    score += 1
            
            final_data = {
                "ad_soyad": st.session_state['student_info']['name'],
                "no": st.session_state['student_info']['no'],
                "tarih": time.strftime("%Y-%m-%d %H:%M"),
                "on_test_puan": st.session_state['scores']['pre'],
                "son_test_puan": score,
                "gelisim": score - st.session_state['scores']['pre']
            }
            
            if save_results_to_firebase(final_data):
                st.balloons()
                st.success("Tebrikler! Sonuçlar kaydedildi.")
                st.metric("Son Puan", score, delta=score - st.session_state['scores']['pre'])
            else:
                st.error("Kayıt hatası oluştu.")
