import streamlit as st
import whisper
import os
import tempfile
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
# --- API KEYLER (SECRETS'TAN ÇEKİLİYOR - ARTIK GÜVENLİ) ---
gemini_api_key = st.secrets["gemini_key"]
openai_api_key = st.secrets["openai_key"]
ADMIN_PASSWORD = st.secrets["admin_password"]

# --- FIREBASE BAĞLANTISI (GÜVENLİ & PUBLIC YÖNTEM) ---
# --- FIREBASE BAĞLANTISI (KESİN ÇÖZÜM) ---
if not firebase_admin._apps:
    try:
        # Secrets'tan veriyi al
        key_dict = dict(st.secrets["firebase"])
        
        # 🔥 BU SATIR ÇOK ÖNEMLİ: \n yazılarını gerçek ENTER tuşuna çevirir
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
        'step': 0, # 0: Giriş, 1: Admin, 2: Öğrenci Sınav
        'user_role': None, # 'student' veya 'admin'
        'student_info': {},
        'scores': {'pre': 0, 'post': 0},
        'pre_answers': {},
        'user_answers_post': {},
        'exam_finished': False,
        'data': [],
        'mistakes': [],
        'shuffled_ops_post': {},
        'audio_speed': 1.0 
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# --- FIREBASE FONKSİYONLARI ---
def save_results_to_firebase(student_data):
    """Sonuçları Firebase Firestore'a kaydeder"""
    try:
        # Koleksiyon adı: 'exam_results'
        # Belge adı: Öğrenci No (Benzersiz olması için)
        doc_ref = db.collection('exam_results').document(student_data['no'])
        doc_ref.set(student_data)
        return True
    except Exception as e:
        st.error(f"Veritabanı Hatası: {e}")
        return False

def get_class_data_from_firebase():
    """Öğretmen için tüm sonuçları çeker"""
    docs = db.collection('exam_results').stream()
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    return data

# --- DİĞER FONKSİYONLAR (SES, PDF, ANALİZ) ---
# (Burada önceki kodundaki analyze_full_text, sesi_sokup_al, generate_audio fonksiyonları aynen duracak)
# Yer kaplamasın diye kısalttım, sen önceki koddan kopyalayabilirsin veya istersen tam halini atarım.
@st.cache_resource
def load_whisper(): return whisper.load_model("base")

def sesi_sokup_al(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-q:a", "9", "-y", audio_path]
    try: subprocess.run(command, capture_output=True, text=True); return True
    except: return False

def analyze_full_text_with_gemini(full_text):
    model = genai.GenerativeModel('gemini-2.0-flash') 
    prompt = f"""GÖREV: Aşağıdaki metni eğitim için analiz et. 
    1. VİDEO ÖZETİ (ozet)
    2. EK KAYNAK (ek_bilgi)
    3. SORU (soru_data) - A,B,C,D ve dogru_sik.
    METİN: "{full_text}"
    JSON formatında döndür."""
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        start = text.find('['); end = text.rfind(']') + 1
        return json.loads(text[start:end])
    except: return []

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
def create_study_pdf(data, mistakes, include_extra=False):
    pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, "CALISMA PLANI", ln=1, align='C')
    # ... (PDF detayları önceki kodla aynı)
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= ARAYÜZ =================

st.title("☁️ Gemini Eğitim Platformu (Online)")

# SİSTEMDEKİ HAZIR VERİYİ KONTROL ET
# Öğretmen bir kez işleyince veriyi 'lesson_data.json' dosyasına kaydederiz.
# Öğrenciler veritabanına değil, bu statik dosyaya erişir (daha hızlı).
LESSON_FILE = "lesson_data.json"

if os.path.exists(LESSON_FILE) and not st.session_state['data']:
    with open(LESSON_FILE, 'r', encoding='utf-8') as f:
        st.session_state['data'] = json.load(f)

# --- GİRİŞ EKRANI ---
if st.session_state['step'] == 0:
    tab1, tab2 = st.tabs(["👨‍🎓 Öğrenci Girişi", "👨‍🏫 Öğretmen Paneli"])
    
    with tab1:
        st.subheader("Öğrenci Girişi")
        s_name = st.text_input("Ad Soyad")
        s_no = st.text_input("Öğrenci No")
        
        if st.button("Sınava Başla"):
            if s_name and s_no:
                if not st.session_state['data']:
                    st.error("Sistemde yüklü ders yok! Lütfen öğretmenin dersi yüklemesini bekleyin.")
                else:
                    st.session_state['student_info'] = {'name': s_name, 'no': s_no}
                    st.session_state['user_role'] = 'student'
                    st.session_state['step'] = 2 # Direkt Ön Teste Git
                    st.rerun()
            else:
                st.warning("Bilgileri doldurunuz.")

    with tab2:
        st.subheader("Öğretmen Girişi")
        password = st.text_input("Yönetici Şifresi", type="password")
        if st.button("Yönetici Girişi"):
            if password == ADMIN_PASSWORD:
                st.session_state['user_role'] = 'admin'
                st.session_state['step'] = 1 # Video Yükleme Paneli
                st.rerun()
            else:
                st.error("Hatalı Şifre")

# --- ADIM 1: ÖĞRETMEN PANELİ (VİDEO İŞLEME & RAPORLAMA) ---
elif st.session_state['step'] == 1 and st.session_state['user_role'] == 'admin':
    st.header("👨‍🏫 Öğretmen Kontrol Paneli")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("1. Yeni Ders Yükle")
        up = st.file_uploader("Ders Videosu Seç (.mp4)", type=["mp4"])
        if up and st.button("Videoyu İşle ve Yayına Al"):
            with st.spinner("Video işleniyor... Bu işlem biraz sürebilir."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(up.read())
                
                # Ses Ayıkla
                audio_path = tfile.name.replace(".mp4", ".mp3")
                sesi_sokup_al(tfile.name, audio_path)
                
                # Whisper
                model_w = load_whisper()
                full_text = model_w.transcribe(audio_path)['text']
                
                # Gemini Analiz
                analysis = analyze_full_text_with_gemini(full_text)
                
                if analysis:
                    # JSON Olarak Kaydet (Tüm öğrenciler bunu görecek)
                    with open(LESSON_FILE, 'w', encoding='utf-8') as f:
                        json.dump(analysis, f, ensure_ascii=False)
                    st.session_state['data'] = analysis
                    st.success("✅ Ders başarıyla işlendi ve yayına alındı!")
                else:
                    st.error("Analiz başarısız oldu.")

    with col_b:
        st.subheader("2. Sınıf Raporları (Firebase)")
        if st.button("Sonuçları Getir"):
            class_data = get_class_data_from_firebase()
            if class_data:
                df = pd.DataFrame(class_data)
                st.dataframe(df)
                
                # CSV İndir (UTF-8 Sig ile Türkçe karakter uyumlu)
                csv = df.to_csv(index=False, sep=';').encode('utf-8-sig')
                st.download_button("Excel/CSV İndir", csv, "sinif_raporu.csv")
            else:
                st.info("Henüz sınavı tamamlayan öğrenci yok.")

# --- ADIM 2: ÖĞRENCİ - ÖN TEST ---
elif st.session_state['step'] == 2:
    st.info(f"Hoşgeldin, **{st.session_state['student_info']['name']}**. Başarılar!")
    st.subheader("📝 Ön Bilgi Testi")
    
    with st.form("pre_test_form"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            st.write(f"**{i+1}.** {q['soru']}")
            ans[i] = st.radio("Cevap", [q['A'], q['B'], q['C'], q['D']], key=f"p_{i}", index=None)
            st.markdown("---")
            
        if st.form_submit_button("Testi Bitir"):
            score = 0
            mistakes = []
            pre_test_data = {}
            for i, item in enumerate(st.session_state['data']):
                correct = item['soru_data'][item['soru_data']['dogru_sik']]
                user_res = ans.get(i)
                is_correct = (user_res == correct)
                pre_test_data[i] = {"given": user_res, "correct": is_correct}
                if is_correct: score += 1
                else: mistakes.append(i)
            
            st.session_state['scores']['pre'] = score
            st.session_state['mistakes'] = mistakes
            st.session_state['step'] = 3
            st.rerun()

# --- ADIM 3: ÖĞRENCİ - ÇALIŞMA EKRANI ---
elif st.session_state['step'] == 3:
    st.success(f"Ön Test Puanın: {st.session_state['scores']['pre']}")
    if st.session_state['mistakes']:
        st.warning("Aşağıdaki eksik konulara çalışmalısın.")
    
    # ... (PDF ve Ses Çalma Kodları Buraya - Önceki koddan al) ...
    # Hız Ayarı vs. hepsi burada olacak.
    
    if st.button("Son Sınava Geç"):
        st.session_state['step'] = 4
        st.rerun()
        
    # İçerik Gösterimi (Özet)
    for i, item in enumerate(st.session_state['data']):
        if i in st.session_state['mistakes']:
            st.error(f"Eksik Konu: {item['alt_baslik']}")
            st.write(item['ozet'])
            if st.button("🔊 Dinle", key=f"list_{i}"):
                path = generate_audio_openai(item['ozet'], 1.0)
                st.audio(path)

# --- ADIM 4: SON TEST & FIREBASE KAYIT ---
elif st.session_state['step'] == 4:
    st.subheader("🎓 Son Değerlendirme")
    
    with st.form("post_test_form"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            st.write(f"**{i+1}.** {q['soru']}")
            ans[i] = st.radio("Cevap", [q['A'], q['B'], q['C'], q['D']], key=f"last_{i}")
            st.markdown("---")
            
        if st.form_submit_button("Sınavı Tamamla"):
            score = 0
            for i, item in enumerate(st.session_state['data']):
                if ans.get(i) == item['soru_data'][item['soru_data']['dogru_sik']]:
                    score += 1
            
            # --- FIREBASE'E KAYDETME ANI ---
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
                st.success("Tebrikler! Sonuçların sisteme başarıyla kaydedildi.")
                st.info(f"Son Puanın: {score}")
                st.stop() # Uygulamayı bitir
            else:

                st.error("Kayıt sırasında bir hata oluştu.")




