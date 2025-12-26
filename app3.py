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
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, firestore
from fpdf import FPDF
from openai import OpenAI 

# --- AYARLAR ---
st.set_page_config(page_title="Gemini Eğitim Platformu", layout="wide", page_icon="🎓")
nest_asyncio.apply()

# =============================================================================
# --- CSS VE TASARIM ENTEGRASYONU ---
# (Senin gönderdiğin font ailesini CDN üzerinden çekip uyguluyoruz)
# =============================================================================
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Code+Pro:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">

<style>
    /* 1. GLOBAL FONT AYARLARI (Senin CSS'ine uygun) */
    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
        color: #2c3e50;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Source Serif 4', serif;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    code {
        font-family: 'Source Code Pro', monospace;
    }

    /* 2. KAĞIT / KART TASARIMI */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    /* Kartların Stili */
    .stCard {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        transition: box-shadow 0.3s ease;
    }
    .stCard:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    }

    /* 3. ÖZEL BİLEŞENLER */
    
    /* Soru Kartları */
    .question-box {
        background-color: #fff;
        border-left: 4px solid #3498db;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-family: 'Source Serif 4', serif;
        font-size: 1.1rem;
    }
    
    /* Çalışma Planı Kutuları */
    .topic-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 24px;
        background: white;
    }
    
    .topic-header {
        padding: 12px 20px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .topic-header.success {
        background-color: #f0fdf4;
        color: #166534;
        border-bottom: 1px solid #dcfce7;
    }
    
    .topic-header.error {
        background-color: #fef2f2;
        color: #991b1b;
        border-bottom: 1px solid #fee2e2;
    }
    
    .topic-content {
        padding: 20px;
        font-family: 'Source Sans 3', sans-serif;
        line-height: 1.6;
        color: #374151;
    }
    
    /* Ek Kaynak Kutusu - Akademik Görünüm */
    .extra-source {
        margin-top: 15px;
        padding: 15px;
        background-color: #f8f9fa;
        border-top: 1px solid #eee;
        font-family: 'Source Serif 4', serif;
        font-style: italic;
        color: #555;
        font-size: 0.95rem;
    }
    
    /* Buton Özelleştirmesi */
    .stButton > button {
        border-radius: 6px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    
    /* Input Alanları */
    .stTextInput > div > div > input {
        font-family: 'Source Sans 3', sans-serif;
    }
    
    /* Sekme (Tab) Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        padding: 8px 16px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f3f4f6;
        color: #111827;
    }

</style>
""", unsafe_allow_html=True)

# --- API KEYLER ---
gemini_api_key = st.secrets["gemini_key"]
openai_api_key = st.secrets["openai_key"]
ADMIN_PASSWORD = st.secrets["admin_password"]

# --- FIREBASE BAĞLANTISI ---
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

try:
    db = firestore.client()
except Exception as e:
    st.error(f"Veritabanı İstemcisi Hatası: {e}")

# --- API BAĞLANTILARI ---
client = None 
try:
    genai.configure(api_key=gemini_api_key)
    client = OpenAI(api_key=openai_api_key)
except: 
    pass 

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

# --- FIREBASE KAYIT ---
def save_results_to_firebase(student_data):
    if db is None:
        st.error("Veritabanı bağlantısı yok!")
        return False
    try:
        doc_ref = db.collection('exam_results').document(str(student_data['no']))
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

# --- VERİ DÜZELTME MOTORU ---
def format_data_for_csv(df, soru_sayisi_input=None):
    if 'on_test_puan' in df.columns and 'on_test' in df.columns:
        df['1. Test Doğru Sayısı'] = df['on_test_puan'].combine_first(df['on_test'])
    elif 'on_test' in df.columns:
        df['1. Test Doğru Sayısı'] = df['on_test']
    elif 'on_test_puan' in df.columns:
        df['1. Test Doğru Sayısı'] = df['on_test_puan']
    else:
        df['1. Test Doğru Sayısı'] = 0 

    if 'son_test_puan' in df.columns and 'son_test' in df.columns:
        df['2. Test Doğru Sayısı'] = df['son_test_puan'].combine_first(df['son_test'])
    elif 'son_test' in df.columns:
        df['2. Test Doğru Sayısı'] = df['son_test']
    elif 'son_test_puan' in df.columns:
        df['2. Test Doğru Sayısı'] = df['son_test_puan']
    else:
        df['2. Test Doğru Sayısı'] = 0

    df['1. Test Doğru Sayısı'] = pd.to_numeric(df['1. Test Doğru Sayısı'], errors='coerce').fillna(0).astype(int)
    df['2. Test Doğru Sayısı'] = pd.to_numeric(df['2. Test Doğru Sayısı'], errors='coerce').fillna(0).astype(int)

    df['NET'] = df['2. Test Doğru Sayısı'] - df['1. Test Doğru Sayısı']

    if 'ad_soyad' in df.columns: df['Ad Soyad'] = df['ad_soyad']
    else: df['Ad Soyad'] = "Bilinmiyor"
        
    if 'no' in df.columns: df['Öğrenci No'] = df['no']
    else: df['Öğrenci No'] = 0

    final_count = soru_sayisi_input if soru_sayisi_input and soru_sayisi_input > 0 else 15
    df['Soru Sayısı'] = final_count

    target_columns = ['Ad Soyad', 'Öğrenci No', 'Soru Sayısı', '1. Test Doğru Sayısı', '2. Test Doğru Sayısı', 'NET']
    
    for col in target_columns:
        if col not in df.columns:
            df[col] = 0 if 'Sayısı' in col or 'NET' in col or 'No' in col else ""

    return df[target_columns]

# --- YARDIMCI FONKSİYONLAR ---
def safe_text(text):
    if text is None: return ""
    tr_map = {
        ord('ı'):'i', ord('İ'):'I', ord('ğ'):'g', ord('Ğ'):'G', 
        ord('ü'):'u', ord('Ü'):'U', ord('ş'):'s', ord('Ş'):'S', 
        ord('ö'):'o', ord('Ö'):'O', ord('ç'):'c', ord('Ç'):'C',
        ord('’'):"'", '‘':"'", '“':'"', '”':'"', '–':'-', '…':'...'
    }
    try:
        return text.translate(tr_map).encode('latin-1', 'replace').decode('latin-1')
    except:
        return text

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
    
# --- GELİŞMİŞ PDF TASARIMI (İSTEĞİNE UYGUN) ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, safe_text('Gemini Egitim Platformu | Kisisel Calisma Plani'), 0, 1, 'R')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(180, 180, 180)
        self.cell(0, 10, safe_text('Sayfa ') + str(self.page_no()), 0, 0, 'C')

    def topic_section(self, title, summary, extra_info, is_mistake, include_extra):
        # Renk Paleti (CSS'e uyumlu)
        if is_mistake:
            header_fill = (254, 242, 242) # Açık Kırmızı
            header_text = (153, 27, 27)   # Koyu Kırmızı
            border_col = (252, 165, 165)  # Kenarlık Kırmızı
            status_text = "(!) TEKRAR ET"
        else:
            header_fill = (240, 253, 244) # Açık Yeşil
            header_text = (22, 101, 52)   # Koyu Yeşil
            border_col = (134, 239, 172)  # Kenarlık Yeşil
            status_text = "TAMAMLANDI"

        # Kutu Çizimi
        self.set_draw_color(*border_col)
        self.set_line_width(0.3)
        
        # Başlık Arka Planı
        self.set_fill_color(*header_fill)
        self.set_text_color(*header_text)
        self.set_font('Arial', 'B', 11)
        
        # X ve Y koordinatlarını sakla
        x = self.get_x()
        y = self.get_y()
        
        # Başlık Hücresi
        title_full = f"{status_text}: {safe_text(title)}"
        self.cell(0, 10, title_full, 1, 1, 'L', True)
        
        # İçerik Alanı (Kenarlıklar için)
        content_start_y = self.get_y()
        
        # Özet Metni
        self.set_text_color(50, 50, 50)
        self.set_font('Arial', '', 10)
        self.set_xy(x + 2, content_start_y + 3) # Biraz içeriden başla
        self.multi_cell(0, 5, safe_text(summary))
        
        # Ek Kaynak (Eğer isteniyorsa ve varsa)
        if include_extra and extra_info:
            self.ln(3)
            # Ayırıcı çizgi
            line_y = self.get_y()
            self.set_draw_color(220, 220, 220)
            self.line(x + 2, line_y, 200, line_y)
            self.ln(3)
            
            # Ek Bilgi Başlığı
            self.set_font('Arial', 'BI', 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, safe_text("Akademik Not / Ek Kaynak:"), 0, 1)
            
            # Ek Bilgi Metni
            self.set_font('Arial', 'I', 9)
            self.multi_cell(0, 5, safe_text(extra_info))
        
        # Alt boşluk ve kutu kapama
        self.ln(3)
        content_end_y = self.get_y()
        
        # Kutunun dış çerçevesini çiz (Başlıktan aşağıya kadar)
        self.set_draw_color(*border_col)
        self.set_xy(x, content_start_y)
        self.rect(x, content_start_y, 190, content_end_y - content_start_y)
        
        self.set_y(content_end_y + 6) # Bir sonraki kutu için boşluk

def create_study_pdf(data, mistakes, include_extra=True):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # Ana Başlık
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 15, safe_text("CALISMA PLANI RAPORU"), ln=1, 'C')
    
    # Alt Bilgi
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(100, 100, 100)
    type_str = "Detayli Akademik Rapor" if include_extra else "Ozet Konu Anlatimi"
    pdf.cell(0, 8, safe_text(f"Rapor Turu: {type_str}"), ln=1, 'C')
    pdf.ln(10)
    
    for i, item in enumerate(data):
        baslik = item.get('alt_baslik', 'Konu')
        ozet = item.get('ozet', '')
        ek_bilgi = item.get('ek_bilgi', '')
        is_mistake = i in mistakes
        
        pdf.topic_section(baslik, ozet, ek_bilgi, is_mistake, include_extra)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= ARAYÜZ (YENİLENMİŞ TASARIM) =================

st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-family: "Source Serif 4", serif; color: #1e293b; font-size: 3rem;'>Gemini Eğitim Platformu</h1>
        <p style='font-family: "Source Sans 3", sans-serif; color: #64748b; font-size: 1.2rem;'>Yapay Zeka Destekli, Akademik Standartlarda Öğrenme Deneyimi</p>
    </div>
""", unsafe_allow_html=True)

LESSON_FILE = "lesson_data.json"

if os.path.exists(LESSON_FILE) and not st.session_state['data']:
    try:
        with open(LESSON_FILE, 'r', encoding='utf-8') as f:
            st.session_state['data'] = json.load(f)
    except: pass

# --- GİRİŞ ---
if st.session_state['step'] == 0:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Öğrenci Girişi", "Öğretmen Girişi"])
        
        with tab1:
            st.markdown("### 🎓 Öğrenci Portalı")
            s_name = st.text_input("Ad Soyad", placeholder="Tam adınızı giriniz")
            s_no = st.text_input("Öğrenci Numarası", placeholder="Numaranızı giriniz")
            if st.button("Sınava Başla", type="primary"):
                if s_name and s_no:
                    if not st.session_state['data']:
                        st.error("Sistemde yüklü ders bulunamadı.")
                    else:
                        st.session_state['student_info'] = {'name': s_name, 'no': s_no}
                        st.session_state['user_role'] = 'student'
                        st.session_state['step'] = 2 
                        st.rerun()
                else: st.warning("Lütfen bilgileri eksiksiz giriniz.")

        with tab2:
            st.markdown("### 🏛️ Yönetici Paneli")
            pwd = st.text_input("Yönetici Şifresi", type="password")
            if st.button("Panele Giriş", type="secondary"):
                if pwd == ADMIN_PASSWORD:
                    st.session_state['user_role'] = 'admin'
                    st.session_state['step'] = 1
                    st.rerun()
                else: st.error("Erişim reddedildi.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- ADIM 1: ÖĞRETMEN ---
elif st.session_state['step'] == 1 and st.session_state['user_role'] == 'admin':
    st.markdown("<h2 style='text-align:center;'>Yönetici Kontrol Paneli</h2>", unsafe_allow_html=True)
    
    tab_ders, tab_sonuc = st.tabs(["📚 Ders İçeriği Yönetimi", "📊 Sınav Analitikleri"])
    
    with tab_ders:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.subheader("Yeni Ders Yükle")
        col1, col2 = st.columns([2, 1])
        with col1:
            up = st.file_uploader("Video Dosyası Seç (.mp4)", type=["mp4"])
        with col2:
            if up: st.video(up)
            
        if up and st.button("Analizi Başlat", type="primary"):
            with st.spinner("Yapay zeka akademik analiz yapıyor..."):
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
                            st.success("Ders içeriği başarıyla oluşturuldu.")
                        else: st.error("AI yanıt vermedi.")
                    else: st.error("Ses işleme hatası.")
                except Exception as e: st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_sonuc:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Verileri Yenile"):
                 st.session_state['data_raw'] = get_class_data_from_firebase()
        
        data_raw = st.session_state.get('data_raw', get_class_data_from_firebase())
        
        if data_raw:
            df_raw = pd.DataFrame(data_raw)
            mevcut_soru = len(st.session_state['data']) if st.session_state['data'] else 15
            df_clean = format_data_for_csv(df_raw, soru_sayisi_input=mevcut_soru)
            
            st.dataframe(df_clean, use_container_width=True)
            
            csv = df_clean.to_csv(sep=';', index=False, encoding='utf-8-sig')
            st.download_button("📥 Excel Olarak İndir", csv, "sonuclar.csv", "text/csv")
        else:
            st.info("Kayıtlı veri bulunamadı.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- ADIM 2: ÖN TEST ---
elif st.session_state['step'] == 2:
    st.info(f"Hoş geldin, {st.session_state['student_info']['name']}. Lütfen seviye tespit sınavını tamamla.")
    
    with st.form("pre_test_form"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            
            st.markdown(f"""
            <div class="question-box">
                <strong>SORU {i+1}:</strong> {q['soru']}
            </div>
            """, unsafe_allow_html=True)
            
            ans[i] = st.radio("Seçiniz:", [q['A'], q['B'], q['C'], q['D']], key=f"p_{i}", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Sınavı Tamamla", type="primary")
        
        if submitted:
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

# --- ADIM 3: ÇALIŞMA PLANI ---
elif st.session_state['step'] == 3:
    st.markdown(f"""
    <div style='text-align:center; padding: 20px;'>
        <h2 style='color:#2c3e50;'>Kişiselleştirilmiş Çalışma Planı</h2>
        <p style='font-size:1.2rem;'>Ön Test Puanı: <strong>{st.session_state['scores']['pre']} / {len(st.session_state['data'])}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state['mistakes']:
            # PDF 1: Sadece Özet
            pdf_ozet = create_study_pdf(st.session_state['data'], st.session_state['mistakes'], include_extra=False)
            st.download_button("📄 Özet Raporu İndir", pdf_ozet, "Ozet_Calisma_Plani.pdf", "application/pdf", use_container_width=True)
            
    with col2:
        if st.session_state['mistakes']:
            # PDF 2: Geniş (Ek Kaynaklı)
            pdf_genis = create_study_pdf(st.session_state['data'], st.session_state['mistakes'], include_extra=True)
            st.download_button("📑 Detaylı Akademik Rapor İndir", pdf_genis, "Detayli_Calisma_Plani.pdf", "application/pdf", type="primary", use_container_width=True)

    with col3:
        if st.button("Son Sınava Geç ➡️", type="primary", use_container_width=True):
            st.session_state['step'] = 4
            st.rerun()

    st.markdown("---")
    
    for i, item in enumerate(st.session_state['data']):
        is_wrong = i in st.session_state['mistakes']
        status_class = "error" if is_wrong else "success"
        icon = "⚠️" if is_wrong else "✅"
        status_text = "Eksik Konu - Tekrar Gerekli" if is_wrong else "Konu Anlaşıldı"
        
        st.markdown(f"""
        <div class="topic-box">
            <div class="topic-header {status_class}">
                <span>{icon}</span>
                <span style="flex-grow:1;">{item['alt_baslik']}</span>
                <span style="font-size:0.8rem; opacity:0.8;">{status_text}</span>
            </div>
            <div class="topic-content">
                {item['ozet']}
        """, unsafe_allow_html=True)
        
        # Ek Bilgi Kısmı (Sadece Hatalıysa veya gösterilmek isteniyorsa)
        if is_wrong and item.get('ek_bilgi'):
            st.markdown(f"""
                <div class="extra-source">
                    <strong>📚 Akademik Ek Kaynak:</strong><br>
                    {item['ek_bilgi']}
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div></div>", unsafe_allow_html=True)

# --- ADIM 4: SON SINAV ---
elif st.session_state['step'] == 4:
    st.markdown("<h2 style='text-align:center;'>Dönem Sonu Değerlendirme Sınavı</h2>", unsafe_allow_html=True)
    
    with st.form("post_test_form"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            st.markdown(f"""
            <div class="question-box">
                <strong>SORU {i+1}:</strong> {q['soru']}
            </div>
            """, unsafe_allow_html=True)
            ans[i] = st.radio("Cevap:", [q['A'], q['B'], q['C'], q['D']], key=f"son_{i}", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
        
        if st.form_submit_button("Sınavı Bitir ve Kaydet", type="primary"):
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
                st.markdown(f"""
                <div class='stCard' style='text-align:center; background-color:#f0fdf4;'>
                    <h1 style='color:#166534;'>🎉 Tebrikler!</h1>
                    <h3>Son Sınav Puanı: {score} / {len(st.session_state['data'])}</h3>
                    <p>Sonuçlarınız sisteme başarıyla kaydedildi.</p>
                </div>
                """, unsafe_allow_html=True)
