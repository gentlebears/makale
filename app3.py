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
st.set_page_config(page_title="Gemini Eğitim Platformu (v2.5 Powered)", layout="wide")
nest_asyncio.apply()

# --- API KEYLER ---
gemini_api_key = st.secrets["gemini_key"]
openai_api_key = st.secrets["openai_key"]
ADMIN_PASSWORD = st.secrets["admin_password"]

# --- FIREBASE BAĞLANTISI ---
if not firebase_admin._apps:
    try:
        key_dict = dict(st.secrets["firebase"])
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

# --- FIREBASE KAYIT FONKSİYONLARI ---
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

# --- YARDIMCI: PDF İÇİN KARAKTER DÜZELTİCİ ---
def safe_text(text):
    if text is None: return ""
    tr_map = {
        ord('ı'):'i', ord('İ'):'I', ord('ğ'):'g', ord('Ğ'):'G', 
        ord('ü'):'u', ord('Ü'):'U', ord('ş'):'s', ord('Ş'):'S', 
        ord('ö'):'o', ord('Ö'):'O', ord('ç'):'c', ord('Ç'):'C',
        ord('•'):'-', ord('’'):"'" 
    }
    try:
        return text.translate(tr_map).encode('latin-1', 'replace').decode('latin-1')
    except:
        return text

# --- WHISPER & AI FONKSİYONLARI ---
@st.cache_resource
def load_whisper():
    return whisper.load_model("base", device="cpu")

def sesi_sokup_al(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-y", audio_path]
    try: 
        subprocess.run(command, capture_output=True, text=True)
        return True
    except: 
        return False

def analyze_full_text_with_gemini(full_text):
    # 🔥 DOĞRU KONFİGÜRASYON: Önce Gemini 2.5 Flash dene
    # Eğer API hatası olursa 1.5'e düş (Akıllı Yedekleme)
    
    primary_model = "gemini-2.5-flash"
    fallback_model = "gemini-1.5-flash"
    
    model = None
    try:
        # Önce en yeni 2.5 modelini deniyoruz
        model = genai.GenerativeModel(primary_model)
        # Test çağrısı (Modelin yüklendiğinden emin olmak için)
        model.generate_content("test") 
    except:
        # Hata alırsak 1.5'e geçiyoruz
        st.warning(f"⚠️ {primary_model} yoğunluk nedeniyle yanıt vermedi, {fallback_model} devreye alındı.")
        model = genai.GenerativeModel(fallback_model)

    st.info(f"🕵️ DEBUG: Whisper {len(full_text)} karakterlik metin çıkardı.")
    
    if len(full_text) < 50:
        st.warning(f"⚠️ Metin çok kısa.")
        return []

    # PROMPT: Hem Özet Hem Ek Kaynak İstiyoruz
    prompt = f"""
    Sen uzman bir öğretmensin. Aşağıdaki video transkriptini analiz et.
    
    GÖREVLERİN:
    1. Konuyu mantıklı alt başlıklara böl.
    2. Her başlık için videodan bir ÖZET çıkar.
    3. [ÖNEMLİ] Her başlık için, videoda olmasa bile kendi veritabanından derinlemesine AKADEMİK EK BİLGİ ekle.
    4. Her başlık için çoktan seçmeli bir soru hazırla.

    Çıktı SADECE geçerli bir JSON formatında olmalı:
    [
      {{
        "alt_baslik": "Konu Başlığı",
        "ozet": "Konunun özeti buraya.",
        "ek_bilgi": "Konuyla ilgili ekstra akademik detay veya ilginç bilgi.",
        "soru_data": {{
            "soru": "Soru metni?",
            "A": "...", "B": "...", "C": "...", "D": "...",
            "dogru_sik": "A"
        }}
      }}
    ]

    METİN: "{full_text}"
    """
    try:
        response = model.generate_content(prompt)
        text = response.text
        text = text.replace("```json", "").replace("```", "").strip()
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
    
# --- GELİŞMİŞ PDF FONKSİYONU ---
def create_study_pdf(data, mistakes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Başlık
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "KISISELLESTIRILMIS CALISMA PLANI", ln=1, align='C')
    pdf.ln(5)
    
    for i, item in enumerate(data):
        baslik = safe_text(item.get('alt_baslik', 'Konu'))
        ozet = safe_text(item.get('ozet', ''))
        ek_bilgi = safe_text(item.get('ek_bilgi', ''))
        
        # Hata kontrolü ve Renklendirme
        if i in mistakes:
            # HATA VARSA KIRMIZI
            pdf.set_text_color(200, 0, 0)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"(!) {baslik} - [TEKRAR ET]", ln=1)
        else:
            # DOĞRUYSA YEŞİL
            pdf.set_text_color(0, 100, 0)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"{baslik} (Tamamlandi)", ln=1)
        
        # İçerik
        pdf.set_text_color(0)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 6, ozet)
        pdf.ln(2)
        
        # Ek Bilgi (Gri ve İtalik)
        if ek_bilgi:
            pdf.set_text_color(80, 80, 80)
            pdf.set_font("Arial", 'I', 10)
            pdf.multi_cell(0, 6, f"[EK KAYNAK]: {ek_bilgi}")
            pdf.ln(2)
            
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= ARAYÜZ =================

st.title("☁️ Gemini 2.5 Eğitim Platformu (Cloud)")

LESSON_FILE = "lesson_data.json"

if os.path.exists(LESSON_FILE) and not st.session_state['data']:
    try:
        with open(LESSON_FILE, 'r', encoding='utf-8') as f:
            st.session_state['data'] = json.load(f)
    except: pass

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
                    st.error("Sistemde yüklü ders yok!")
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
            with st.spinner("Video işleniyor... (Bu işlem biraz sürebilir)"):
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(up.read())
                    
                    audio_path = tfile.name.replace(".mp4", ".mp3")
                    
                    if not sesi_sokup_al(tfile.name, audio_path):
                        st.error("FFMPEG Hatası.")
                        st.stop()
                    
                    model_w = load_whisper()
                    result = model_w.transcribe(audio_path)
                    full_text = result['text']
                    
                    # Gemini 2.5 ile Analiz
                    analysis = analyze_full_text_with_gemini(full_text)
                    
                    if analysis and len(analysis) > 0:
                        with open(LESSON_FILE, 'w', encoding='utf-8') as f:
                            json.dump(analysis, f, ensure_ascii=False)
                        st.session_state['data'] = analysis
                        st.success("✅ Ders (Gemini 2.5 ile) başarıyla işlendi!")
                    else:
                        st.error("Gemini analizi başarısız oldu.")
                except Exception as e:
                    st.error(f"Hata: {e}")

    with col_b:
        st.subheader("2. Sınıf Raporları")
        if st.button("Sonuçları Getir"):
            class_data = get_class_data_from_firebase()
            if class_data:
                df = pd.DataFrame(class_data)
                st.dataframe(df)
            else:
                st.info("Kayıt yok.")

# --- ADIM 2: ÖĞRENCİ - ÖN TEST ---
elif st.session_state['step'] == 2:
    if not st.session_state['data']:
        st.warning("Ders yüklenemedi.")
        if st.button("Yenile"): st.rerun()
    else:
        st.info(f"Hoşgeldin, **{st.session_state['student_info']['name']}**.")
        with st.form("pre_test_form"):
            ans = {}
            for i, item in enumerate(st.session_state['data']):
                q = item.get('soru_data', {})
                st.write(f"**{i+1}.** {q.get('soru', '')}")
                secenekler = [q.get('A'), q.get('B'), q.get('C'), q.get('D')]
                ans[i] = st.radio("Cevap", secenekler, key=f"p_{i}", index=None)
                st.markdown("---")
            
            if st.form_submit_button("Testi Bitir"):
                score = 0
                mistakes = []
                for i, item in enumerate(st.session_state['data']):
                    q = item.get('soru_data', {})
                    dogru = q.get(q.get('dogru_sik', 'A').strip().upper())
                    if ans.get(i) == dogru: score += 1
                    else: mistakes.append(i)
                
                st.session_state['scores']['pre'] = score
                st.session_state['mistakes'] = mistakes
                st.session_state['step'] = 3
                st.rerun()

# --- ADIM 3: ÇALIŞMA ---
elif st.session_state['step'] == 3:
    st.success(f"Puanın: {st.session_state['scores']['pre']}")
    
    # PDF Butonu
    if st.session_state['mistakes']:
        if st.button("📄 Gelişmiş PDF İndir"):
            pdf_bytes = create_study_pdf(st.session_state['data'], st.session_state['mistakes'])
            st.download_button("İndir", pdf_bytes, "Ozel_Calisma_Plani.pdf", "application/pdf")

    if st.button("Son Sınava Geç ->"):
        st.session_state['step'] = 4
        st.rerun()
    
    # --- SES HIZI KONTROLÜ ---
    st.divider()
    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        st.info("🎚️ **Ses Hızı**")
    with col_s2:
        audio_speed = st.select_slider(
            "Yapay Zeka Okuma Hızı:", 
            options=[0.75, 1.0, 1.25, 1.5, 2.0], 
            value=1.0,
            key="speed_slider"
        )
    st.divider()

    for i, item in enumerate(st.session_state['data']):
        if i in st.session_state['mistakes']:
            st.error(f"Eksik: {item.get('alt_baslik')}")
            st.write(item.get('ozet'))
            
            # EK BİLGİ ALANI
            extra = item.get('ek_bilgi')
            if extra:
                with st.expander("📚 Akademik/Ek Kaynak Bilgisi"):
                    st.info(extra)
                    if st.button("🎧 Ek Bilgiyi Dinle", key=f"ex_aud_{i}"):
                        with st.spinner(f"Ek bilgi okunuyor ({audio_speed}x)..."):
                            path_ex = generate_audio_openai(extra, audio_speed)
                            if path_ex: st.audio(path_ex)
            
            # ANA ÖZETİ DİNLEME BUTONU
            if st.button("🔊 Özeti Dinle", key=f"ls_{i}"):
                with st.spinner(f"Seslendiriliyor ({audio_speed}x Hız)..."):
                    path = generate_audio_openai(item.get('ozet'), audio_speed)
                    if path: st.audio(path)
            st.markdown("---")

# --- ADIM 4: SON TEST ---
elif st.session_state['step'] == 4:
    with st.form("post_test"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item.get('soru_data', {})
            st.write(f"**{i+1}.** {q.get('soru')}")
            ans[i] = st.radio("Cevap", [q.get('A'), q.get('B'), q.get('C'), q.get('D')], key=f"l_{i}")
            st.markdown("---")
        
        if st.form_submit_button("Bitir"):
            score = 0
            for i, item in enumerate(st.session_state['data']):
                q = item.get('soru_data', {})
                if ans.get(i) == q.get(q.get('dogru_sik', 'A').strip().upper()): score += 1
            
            final_data = {
                "ad_soyad": st.session_state['student_info']['name'],
                "no": st.session_state['student_info']['no'],
                "tarih": time.strftime("%Y-%m-%d %H:%M"),
                "on_test_puan": st.session_state['scores']['pre'],
                "son_test_puan": score,
                "gelisim": score - st.session_state['scores']['pre']
            }
            save_results_to_firebase(final_data)
            st.balloons()
            st.success(f"Bitti! Puan: {score}")
