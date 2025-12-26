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
st.set_page_config(page_title="Gemini Eğitim Platformu (v4 Stable)", layout="wide", page_icon="🎓")
nest_asyncio.apply()

# --- STİL (CSS) ---
st.markdown("""
<style>
    /* Genel sayfa stili */
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #343a40;
    }
    
    /* Tab sekmeleri stili */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e9ecef;
        border-radius: 5px 5px 0px 0px;
        gap: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #495057;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #007bff !important;
        border-top: 3px solid #007bff;
    }

    /* Form ve kutu stilleri */
    .stForm, .css-1r6slb0 {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    
    /* Buton stilleri */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Soru kartları stili */
    .question-card {
        background-color: #ffffff;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 8px;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .question-title {
        font-weight: bold;
        color: #343a40;
        margin-bottom: 10px;
    }

    /* Çalışma planı kutuları stili */
    .study-card {
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    .study-card-header {
        padding: 15px;
        font-weight: bold;
        color: white;
    }
    .study-card-body {
        background-color: #ffffff;
        padding: 20px;
        line-height: 1.6;
    }
    .study-card-error {
        border: 2px solid #dc3545;
    }
    .study-card-error .study-card-header {
        background-color: #dc3545;
    }
    .study-card-success {
        border: 2px solid #28a745;
    }
    .study-card-success .study-card-header {
        background-color: #28a745;
    }
    
    /* Ek kaynak kutusu stili */
    .extra-resource-box {
        background-color: #f1f3f5;
        border-left: 4px solid #6c757d;
        padding: 15px;
        margin-top: 15px;
        border-radius: 4px;
    }
    .extra-resource-title {
        font-weight: bold;
        color: #495057;
        margin-bottom: 5px;
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

# --- FIREBASE KAYIT FONKSİYONLARI ---
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

# --- VERİ DÜZELTME VE FORMATLAMA MOTORU ---
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
    
# --- GELİŞMİŞ PDF SINIFI VE FONKSİYONU ---
class PDF(FPDF):
    def header(self):
        # Üst bilgi (Header)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, safe_text('Gemini Egitim Platformu - Kisisellestirilmis Calisma Plani'), 0, 1, 'R')
        self.ln(5)

    def footer(self):
        # Alt bilgi (Footer)
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, safe_text('Sayfa ') + str(self.page_no()), 0, 0, 'C')

    def chapter_body(self, body, is_extra=False):
        # Konu içeriği metni
        if is_extra:
            self.set_font('Arial', 'I', 10)
            self.set_text_color(80, 80, 80) # Ek bilgi için gri renk
            self.multi_cell(0, 5, safe_text("[EK KAYNAK]: " + body))
        else:
            self.set_font('Arial', '', 11)
            self.set_text_color(0, 0, 0) # Normal metin siyah
            self.multi_cell(0, 6, safe_text(body))
        self.ln()

    def draw_topic_box(self, title, summary, extra_info, is_mistake, include_extra):
        # Konu kutusunu çiz
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.5)
        
        # Başlık rengi: Hata ise kırmızı, değilse mavi/yeşil
        if is_mistake:
            self.set_fill_color(220, 53, 69) # Kırmızı
            title_prefix = "(!) [TEKRAR ET] "
        else:
            self.set_fill_color(40, 167, 69) # Yeşil
            title_prefix = "[TAMAMLANDI] "
            
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 12)
        
        # Başlık hücresi
        self.cell(0, 10, safe_text(title_prefix + title), 1, 1, 'L', True)
        
        # İçerik kutusu
        self.set_fill_color(250, 250, 250)
        self.set_text_color(0, 0, 0)
        self.set_font('Arial', '', 11)
        
        # İçerik için başlangıç Y koordinatı
        start_y = self.get_y()
        
        # Kenarlık çizmek için dikdörtgen, yüksekliği sonra ayarlanacak
        self.rect(self.get_x(), start_y, self.w - 2 * self.l_margin, 1, 'D')
        
        self.set_xy(self.get_x() + 2, start_y + 2) # İçeriği biraz içeriden başlat
        
        # Özet metni
        self.chapter_body(summary)
        
        # Ek bilgi varsa ve isteniyorsa ekle
        if include_extra and extra_info:
            self.ln(2)
            self.set_draw_color(150, 150, 150)
            self.line(self.get_x(), self.get_y(), self.w - self.r_margin - 2, self.get_y())
            self.ln(3)
            self.chapter_body(extra_info, is_extra=True)
            
        # İçerik bittikten sonra kutunun alt kenarlığını çizmek için yüksekliği hesapla
        end_y = self.get_y()
        box_height = end_y - start_y + 2
        
        # Daha önce çizilen dikdörtgenin yüksekliğini güncelle
        self.set_xy(self.l_margin, start_y)
        self.rect(self.get_x(), self.get_y(), self.w - 2 * self.l_margin, box_height, 'D')
        
        self.set_y(end_y + 5) # Bir sonraki kutu için boşluk

def create_study_pdf(data, mistakes, include_extra=True):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    
    # Kapak Sayfası gibi başlık
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 20, safe_text("KISISELLESTIRILMIS CALISMA PLANI"), ln=1, align='C')
    
    # Alt başlık
    pdf.set_font("Arial", '', 14)
    pdf.set_text_color(100, 100, 100)
    type_text = "Genis Ozet (Ek Kaynaklar Dahil)" if include_extra else "Ozet (Sadece Konu Anlatimi)"
    pdf.cell(0, 10, safe_text(f"Rapor Tipi: {type_text}"), ln=1, align='C')
    pdf.ln(10)
    
    for i, item in enumerate(data):
        baslik = item.get('alt_baslik', 'Konu')
        ozet = item.get('ozet', '')
        ek_bilgi = item.get('ek_bilgi', '')
        is_mistake = i in mistakes
        
        pdf.draw_topic_box(baslik, ozet, ek_bilgi, is_mistake, include_extra)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= ARAYÜZ =================

# Sayfa başlığı ve ikonu
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=80)
with col2:
    st.title("Gemini Eğitim Platformu (Cloud v4)")
    st.markdown("*Yapay Zeka Destekli Kişiselleştirilmiş Öğrenme Deneyimi*")

st.markdown("---")

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
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("student_login_form"):
                st.subheader("Öğrenci Girişi")
                st.markdown("Sınava başlamak için bilgilerinizi giriniz.")
                s_name = st.text_input("Ad Soyad", placeholder="Örn: Ali Yılmaz")
                s_no = st.text_input("Öğrenci No", placeholder="Örn: 12345")
                
                submitted = st.form_submit_button("🚀 Sınava Başla")
                if submitted:
                    if s_name and s_no:
                        if not st.session_state['data']:
                            st.error("Henüz bir ders yüklenmemiş. Lütfen öğretmeninize danışın.")
                        else:
                            st.session_state['student_info'] = {'name': s_name, 'no': s_no}
                            st.session_state['user_role'] = 'student'
                            st.session_state['step'] = 2 
                            st.rerun()
                    else: st.warning("Lütfen tüm bilgileri eksiksiz giriniz.")

    with tab2:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("admin_login_form"):
                st.subheader("Öğretmen Girişi")
                st.markdown("Yönetici paneline erişmek için şifre giriniz.")
                pwd = st.text_input("Şifre", type="password", placeholder="••••••")
                
                submitted = st.form_submit_button("🔑 Giriş Yap")
                if submitted:
                    if pwd == ADMIN_PASSWORD:
                        st.session_state['user_role'] = 'admin'
                        st.session_state['step'] = 1
                        st.rerun()
                    else: st.error("Hatalı Şifre! Lütfen tekrar deneyin.")

# --- ADIM 1: ÖĞRETMEN ---
elif st.session_state['step'] == 1 and st.session_state['user_role'] == 'admin':
    st.header("👨‍🏫 Yönetici Paneli")
    
    tab_ders, tab_sonuc = st.tabs(["📚 Ders İşle / Yükle", "📊 Sınav Sonuçları"])
    
    with tab_ders:
        st.subheader("Yeni Ders İçeriği Oluştur")
        st.markdown("Bir video dosyası yükleyin, yapay zeka sizin için ders notları ve sorular hazırlasın.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            up = st.file_uploader("Video Yükle (.mp4)", type=["mp4"], help="Maksimum 200MB boyutunda bir video dosyası seçin.")
        with col2:
            if up:
                st.video(up)
            
        if up and st.button("✨ Dersi İşle ve Hazırla", type="primary"):
            with st.spinner("Yapay zeka videoyu analiz ediyor, lütfen bekleyin..."):
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
                            st.success("🎉 Ders başarıyla hazırlandı ve kaydedildi!")
                            st.balloons()
                        else: st.error("AI analiz sırasında bir hata oluştu veya yanıt vermedi.")
                    else: st.error("Videonun sesi ayrıştırılamadı. Dosya formatını kontrol edin.")
                except Exception as e: st.error(f"Bir hata oluştu: {str(e)}")
    
    with tab_sonuc:
        st.subheader("Öğrenci Performans Raporları")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            refresh = st.button("🔄 Sonuçları Yenile")
            
        if refresh or 'data_raw' not in st.session_state:
             st.session_state['data_raw'] = get_class_data_from_firebase()

        data_raw = st.session_state.get('data_raw', [])
        
        if data_raw:
            df_raw = pd.DataFrame(data_raw)
            mevcut_soru_sayisi = len(st.session_state['data']) if st.session_state['data'] else 15
            df_clean = format_data_for_csv(df_raw, soru_sayisi_input=mevcut_soru_sayisi)
            
            st.dataframe(df_clean, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Toplam Öğrenci", len(df_clean))
            with col2:
                st.metric("Ortalama NET", f"{df_clean['NET'].mean():.2f}")
                
            csv_data = df_clean.to_csv(sep=';', index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tabloyu Excel (CSV) Olarak İndir",
                data=csv_data,
                file_name="ogrenci_sinav_sonuclari.csv",
                mime="text/csv",
                type="secondary"
            )
        else: 
            st.info("Henüz veritabanında kayıtlı sınav sonucu bulunmamaktadır.")

# --- ADIM 2: ÖN TEST ---
elif st.session_state['step'] == 2:
    st.info(f"👋 Merhaba **{st.session_state['student_info']['name']}**, ön teste hoş geldin. Lütfen tüm soruları dikkatlice cevapla.")
    
    with st.form("pre_test_form"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            
            st.markdown(f"""
            <div class="question-card">
                <div class="question-title">SORU {i+1}</div>
                <div>{q['soru']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            ans[i] = st.radio(
                "Cevabınızı Seçin:", 
                [q['A'], q['B'], q['C'], q['D']], 
                key=f"p_{i}", 
                index=None,
                format_func=lambda x: f"{x}" # Seçeneklerin metnini doğrudan göster
            )
            st.write("") # Boşluk bırak

        st.markdown("---")
        submitted = st.form_submit_button("✅ Testi Bitir ve Sonuçları Gör", type="primary")
        
        if submitted:
            # Tüm soruların cevaplanıp cevaplanmadığını kontrol et (İsteğe bağlı)
            if any(a is None for a in ans.values()):
                st.warning("Lütfen tüm soruları cevaplayınız.")
            else:
                score = 0
                mistakes = []
                for i, item in enumerate(st.session_state['data']):
                    q = item['soru_data']
                    correct_option = q['dogru_sik'].strip()
                    correct_answer_text = q[correct_option]
                    
                    # Seçilen cevap metni ile doğru cevap metnini karşılaştır
                    if ans.get(i) == correct_answer_text:
                        score += 1
                    else:
                        mistakes.append(i)
                
                st.session_state['scores']['pre'] = score
                st.session_state['mistakes'] = mistakes
                st.session_state['step'] = 3
                st.rerun()

# --- ADIM 3: ÇALIŞMA ---
elif st.session_state['step'] == 3:
    st.header("📝 Kişiselleştirilmiş Çalışma Planı")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Ön Test Puanın", f"{st.session_state['scores']['pre']} / {len(st.session_state['data'])}")
    
    with col2:
        if st.session_state['mistakes']:
            st.warning(f"Toplam **{len(st.session_state['mistakes'])}** konuda eksiğin tespit edildi. Aşağıdaki çalışma planını dikkatlice incele.")
        else:
            st.balloons()
            st.success("Tebrikler! Hiç eksiğin yok. Konuları tekrar ederek bilgilerini pekiştirebilirsin.")

    st.markdown("---")
    
    col_pdf1, col_pdf2, col_next = st.columns([1.5, 1.5, 1])
    
    with col_pdf1:
        if st.session_state['mistakes']:
            # Sadece Özet PDF
            pdf_data_summary = create_study_pdf(st.session_state['data'], st.session_state['mistakes'], include_extra=False)
            st.download_button("📄 Planı İndir (Sadece Özet)", pdf_data_summary, "Calisma_Plani_Ozet.pdf", "application/pdf", type="secondary")
            
    with col_pdf2:
        if st.session_state['mistakes']:
            # Geniş Özet (Ek Kaynaklı) PDF
            pdf_data_full = create_study_pdf(st.session_state['data'], st.session_state['mistakes'], include_extra=True)
            st.download_button("📑 Planı İndir (Geniş Özet)", pdf_data_full, "Calisma_Plani_Genis.pdf", "application/pdf", type="primary")

    with col_next:
        if st.button("➡️ Son Sınava Geç", type="primary"):
            st.session_state['step'] = 4
            st.rerun()

    st.markdown("---")
    
    col_s1, col_s2 = st.columns([1, 4])
    with col_s1: st.markdown("### 🎚️ Okuma Hızı:")
    with col_s2: 
        audio_speed = st.select_slider("", options=[0.75, 1.0, 1.25, 1.5, 2.0], value=1.0, format_func=lambda x: f"{x}x")
    st.divider()

    for i, item in enumerate(st.session_state['data']):
        is_wrong = i in st.session_state['mistakes']
        card_class = "study-card-error" if is_wrong else "study-card-success"
        card_status = "TEKRAR ET" if is_wrong else "TAMAMLANDI"
        card_icon = "🔻" if is_wrong else "✅"
        
        st.markdown(f"""
        <div class="study-card {card_class}">
            <div class="study-card-header">
                {card_icon} {item['alt_baslik']} - [{card_status}]
            </div>
            <div class="study-card-body">
                {item['ozet']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_btns = st.columns([1, 4])
        with col_btns[0]:
             if st.button(f"🔊 Özeti Dinle", key=f"dinle_{i}"):
                with st.spinner("Seslendiriliyor..."):
                    path = generate_audio_openai(item['ozet'], audio_speed)
                    if path: st.audio(path)
        
        ek_bilgi = item.get('ek_bilgi')
        if ek_bilgi and is_wrong: # Sadece hatalı konularda ek bilgiyi göster
            st.markdown(f"""
            <div class="extra-resource-box">
                <div class="extra-resource-title">📚 Akademik Ek Kaynak</div>
                <div>{ek_bilgi}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎧 Ek Bilgiyi Dinle", key=f"ek_dinle_{i}"):
                 with st.spinner("Okunuyor..."):
                    path = generate_audio_openai(ek_bilgi, audio_speed)
                    if path: st.audio(path)

        st.markdown("---")

# --- ADIM 4: SON TEST ---
elif st.session_state['step'] == 4:
    st.header("🎯 Son Sınav")
    st.info("Artık öğrendiklerini test etme zamanı. Başarılar!")

    with st.form("post_test_form"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            
            st.markdown(f"""
            <div class="question-card">
                <div class="question-title">SORU {i+1}</div>
                <div>{q['soru']}</div>
            </div>
            """, unsafe_allow_html=True)

            ans[i] = st.radio(
                "Cevabınızı Seçin:", 
                [q['A'], q['B'], q['C'], q['D']], 
                key=f"son_{i}",
                index=None,
                format_func=lambda x: f"{x}"
            )
            st.write("")

        st.markdown("---")
        submitted = st.form_submit_button("🏁 Sınavı Bitir", type="primary")
        
        if submitted:
            if any(a is None for a in ans.values()):
                 st.warning("Lütfen tüm soruları cevaplayınız.")
            else:
                score = 0
                for i, item in enumerate(st.session_state['data']):
                    q = item['soru_data']
                    correct_option = q['dogru_sik'].strip()
                    correct_answer_text = q[correct_option]
                    
                    if ans.get(i) == correct_answer_text:
                        score += 1
                
                res = {
                    "ad_soyad": st.session_state['student_info']['name'],
                    "no": st.session_state['student_info']['no'],
                    "tarih": time.strftime("%Y-%m-%d %H:%M"),
                    "on_test": st.session_state['scores']['pre'],
                    "son_test": score
                }
                if save_results_to_firebase(res):
                    st.balloons()
                    
                    col1, col2, col3 = st.columns([1,2,1])
                    with col2:
                        st.success(f"Sınav Başarıyla Tamamlandı!")
                        st.metric("Son Sınav Puanın", f"{score} / {len(st.session_state['data'])}", delta=score - st.session_state['scores']['pre'])
                        st.markdown("Sonuçlarınız kaydedildi. Öğrenme yolculuğunuzda başarılar dileriz!")
