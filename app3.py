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
st.set_page_config(page_title="Gemini Eğitim Platformu (v4 Stable)", layout="wide")
nest_asyncio.apply()

# --- API KEYLER ---
gemini_api_key = st.secrets["gemini_key"]
openai_api_key = st.secrets["openai_key"]
ADMIN_PASSWORD = st.secrets["admin_password"]

# --- FIREBASE BAĞLANTISI (DÜZELTİLMİŞ) ---
# Önce db değişkenini boş tanımlayalım ki NameError vermesin
db = None 

if not firebase_admin._apps:
    try:
        key_dict = dict(st.secrets["firebase"])
        key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebase Bağlantı Hatası: {e}")
        st.stop()

# BU SATIR ARTIK 'IF' BLOĞUNUN DIŞINDA VE GÜVENDE 
try:
    db = firestore.client()
except Exception as e:
    st.error(f"Veritabanı İstemcisi Hatası: {e}")

# --- API BAĞLANTILARI (GÜVENLİ MOD) ---
client = None # NameError önleyici
try:
    genai.configure(api_key=gemini_api_key)
    client = OpenAI(api_key=openai_api_key)
except: 
    pass # Hata olsa bile client=None olduğu için kod patlamaz

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
    if db is None:
        st.error("Veritabanı bağlantısı yok!")
        return False
    try:
        doc_ref = db.collection('exam_results').document(student_data['no'])
        doc_ref.set(student_data)
        return True
    except Exception as e:
        st.error(f"Veritabanı Hatası: {e}")
        return False

def get_class_data_from_firebase():
    if db is None:
        st.error("Veritabanı bağlantısı yok!")
        return []
    try:
        docs = db.collection('exam_results').stream()
        data = []
        for doc in docs:
            data.append(doc.to_dict())
        return data
    except Exception as e:
        st.error(f"Veri Çekme Hatası: {e}")
        return []

# --- YARDIMCI: PDF İÇİN KARAKTER DÜZELTİCİ ---
def safe_text(text):
    if text is None: return ""
    tr_map = {
        ord('ı'):'i', ord('İ'):'I', ord('ğ'):'g', ord('Ğ'):'G', 
        ord('ü'):'u', ord('Ü'):'U', ord('ş'):'s', ord('Ş'):'S', 
        ord('ö'):'o', ord('Ö'):'O', ord('ç'):'c', ord('Ç'):'C',
        ord('’'):"'", '‘':"'", '“':'"', '”':'"', '–':'-'
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
    primary_model = "gemini-2.5-flash"
    fallback_model = "gemini-2.0-flash"
    
    model = None
    try:
        model = genai.GenerativeModel(primary_model)
        model.generate_content("test") 
    except:
        st.warning(f"⚠️ {primary_model} yanıt vermedi, {fallback_model} kullanılıyor.")
        model = genai.GenerativeModel(fallback_model)

    if len(full_text) < 50: return []

    prompt = f"""
    Sen uzman bir eğitim asistanısın. Video transkriptini analiz et.
    
    GÖREVLER:
    1. Konuyu alt başlıklara böl.
    2. Her başlık için video içeriğinden bir ÖZET çıkar.
    3. [KRİTİK] Her başlık için, videoda geçmese bile, o konuyu akademik olarak destekleyen EK BİLGİ (Extra Resource) ekle.
    4. Her başlık için bir test sorusu yaz.

    Çıktı JSON Formatı:
    [
      {{
        "alt_baslik": "Konu Başlığı",
        "ozet": "Video özeti...",
        "ek_bilgi": "Akademik ve teknik detay bilgi...",
        "soru_data": {{
            "soru": "Soru?",
            "A": "...", "B": "...", "C": "...", "D": "...",
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
        st.error(f"AI Hatası: {e}")
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
    
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "KISISELLESTIRILMIS CALISMA PLANI", ln=1, align='C')
    pdf.ln(5)
    
    for i, item in enumerate(data):
        baslik = safe_text(item.get('alt_baslik', 'Konu'))
        ozet = safe_text(item.get('ozet', ''))
        ek_bilgi = safe_text(item.get('ek_bilgi', ''))
        
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
        
        # Ek Bilgi
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

st.title("☁️ Gemini Eğitim Platformu (Cloud v4 Stable)")

LESSON_FILE = "lesson_data.json"

if os.path.exists(LESSON_FILE) and not st.session_state['data']:
    try:
        with open(LESSON_FILE, 'r', encoding='utf-8') as f:
            st.session_state['data'] = json.load(f)
    except: pass

# --- GİRİŞ ---
if st.session_state['step'] == 0:
    tab1, tab2 = st.tabs(["👨‍🎓 Öğrenci Girişi", "👨‍🏫 Öğretmen Paneli"])
    
    with tab1:
        st.subheader("Öğrenci Girişi")
        s_name = st.text_input("Ad Soyad")
        s_no = st.text_input("Öğrenci No")
        if st.button("Sınava Başla"):
            if s_name and s_no:
                if not st.session_state['data']:
                    st.error("Ders bulunamadı.")
                else:
                    st.session_state['student_info'] = {'name': s_name, 'no': s_no}
                    st.session_state['user_role'] = 'student'
                    st.session_state['step'] = 2 
                    st.rerun()
            else: st.warning("Bilgileri giriniz.")

    with tab2:
        st.subheader("Öğretmen Girişi")
        pwd = st.text_input("Şifre", type="password")
        if st.button("Giriş"):
            if pwd == ADMIN_PASSWORD:
                st.session_state['user_role'] = 'admin'
                st.session_state['step'] = 1
                st.rerun()
            else: st.error("Hatalı Şifre")

# --- ADIM 1: ÖĞRETMEN ---
elif st.session_state['step'] == 1 and st.session_state['user_role'] == 'admin':
    st.header("Yönetici Paneli")
    col1, col2 = st.columns(2)
    with col1:
        up = st.file_uploader("Video (.mp4)", type=["mp4"])
        if up and st.button("Dersi İşle"):
            with st.spinner("Yapay zeka çalışıyor..."):
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(up.read())
                    audio_path = tfile.name.replace(".mp4", ".mp3")
                    
                    if sesi_sokup_al(tfile.name, audio_path):
                        model_w = load_whisper()
                        res = model_w.transcribe(audio_path)
                        analysis = analyze_full_text_with_gemini(res['text'])
                        
                        if analysis:
                            with open(LESSON_FILE, 'w', encoding='utf-8') as f:
                                json.dump(analysis, f, ensure_ascii=False)
                            st.session_state['data'] = analysis
                            st.success("Ders hazırlandı!")
                        else: st.error("AI Yanıt Vermedi.")
                    else: st.error("Ses ayrıştırılamadı.")
                except Exception as e: st.error(str(e))
    
    with col2:
        if st.button("Sonuçları Gör"):
            data = get_class_data_from_firebase()
            if data: st.dataframe(pd.DataFrame(data))
            else: st.info("Henüz sonuç yok.")

# --- ADIM 2: ÖN TEST ---
elif st.session_state['step'] == 2:
    st.info(f"Merhaba {st.session_state['student_info']['name']}, sınava hoşgeldin.")
    with st.form("pre_test"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            st.write(f"**{i+1})** {q['soru']}")
            ans[i] = st.radio("Cevap", [q['A'], q['B'], q['C'], q['D']], key=f"p_{i}", index=None)
            st.write("---")
        
        if st.form_submit_button("Testi Bitir"):
            score = 0
            mistakes = []
            for i, item in enumerate(st.session_state['data']):
                q = item['soru_data']
                correct = q[q['dogru_sik'].strip()]
                if ans.get(i) == correct: score += 1
                else: mistakes.append(i)
            
            st.session_state['scores']['pre'] = score
            st.session_state['mistakes'] = mistakes
            st.session_state['step'] = 3
            st.rerun()

# --- ADIM 3: ÇALIŞMA ---
elif st.session_state['step'] == 3:
    st.success(f"Puan: {st.session_state['scores']['pre']}")
    
    if st.session_state['mistakes']:
        st.warning(f"Toplam {len(st.session_state['mistakes'])} konuda eksiklerin var.")
        if st.button("📥 Kişiselleştirilmiş Çalışma Planı (PDF)"):
            pdf_data = create_study_pdf(st.session_state['data'], st.session_state['mistakes'])
            st.download_button("Planı İndir", pdf_data, "Calisma_Plani.pdf", "application/pdf")
    else:
        st.balloons()
        st.success("Tebrikler! Hiç eksiğin yok. Yine de konuları tekrar edebilirsin.")

    if st.button("Son Sınava Geç ➡️"):
        st.session_state['step'] = 4
        st.rerun()

    st.divider()
    col_s1, col_s2 = st.columns([1, 4])
    with col_s1: st.markdown("### 🎚️ Hız:")
    with col_s2: 
        audio_speed = st.select_slider("", options=[0.75, 1.0, 1.25, 1.5, 2.0], value=1.0)
    st.divider()

    for i, item in enumerate(st.session_state['data']):
        is_wrong = i in st.session_state['mistakes']
        
        if is_wrong:
            st.error(f"🔻 {item['alt_baslik']} (Eksik Konu)")
            st.write(f"**Özet:** {item['ozet']}")
            
            ek_bilgi = item.get('ek_bilgi')
            if ek_bilgi:
                with st.expander("📚 Akademik Ek Kaynak (Okuman Önerilir)"):
                    st.info(ek_bilgi)
                    if st.button("🎧 Ek Bilgiyi Dinle", key=f"ek_dinle_{i}"):
                        with st.spinner("Okunuyor..."):
                            path = generate_audio_openai(ek_bilgi, audio_speed)
                            if path: st.audio(path)
        else:
            st.success(f"✅ {item['alt_baslik']} (Tamamlandı)")
            with st.expander("Konu Özetini Gör"):
                st.write(item['ozet'])
        
        if st.button(f"🔊 Özeti Dinle", key=f"dinle_{i}"):
            with st.spinner("Seslendiriliyor..."):
                path = generate_audio_openai(item['ozet'], audio_speed)
                if path: st.audio(path)
        
        st.write("---")

# --- ADIM 4: SON TEST ---
elif st.session_state['step'] == 4:
    with st.form("post_test"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            st.write(f"**{i+1})** {q['soru']}")
            ans[i] = st.radio("Cevap", [q['A'], q['B'], q['C'], q['D']], key=f"son_{i}")
            st.write("---")
        
        if st.form_submit_button("Sınavı Bitir"):
            score = 0
            for i, item in enumerate(st.session_state['data']):
                q = item['soru_data']
                correct = q[q['dogru_sik'].strip()]
                if ans.get(i) == correct: score += 1
            
            res = {
                "ad_soyad": st.session_state['student_info']['name'],
                "no": st.session_state['student_info']['no'],
                "tarih": time.strftime("%Y-%m-%d %H:%M"),
                "on_test": st.session_state['scores']['pre'],
                "son_test": score
            }
            if save_results_to_firebase(res):
                st.balloons()
                st.success(f"Sınav Bitti! Puan: {score}")
