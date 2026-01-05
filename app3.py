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
st.set_page_config(page_title="Gemini Eğitim Platformu (v4 Final)", layout="wide")
nest_asyncio.apply()

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

def format_data_for_csv(df, soru_sayisi_input=None):
    # --- PUANLARI BİRLEŞTİR ---
    if 'on_test_puan' in df.columns and 'on_test' in df.columns:
        df['1. Test Doğru Sayısı'] = df['on_test_puan'].combine_first(df['on_test'])
    elif 'on_test' in df.columns: df['1. Test Doğru Sayısı'] = df['on_test']
    elif 'on_test_puan' in df.columns: df['1. Test Doğru Sayısı'] = df['on_test_puan']
    else: df['1. Test Doğru Sayısı'] = 0 

    if 'son_test_puan' in df.columns and 'son_test' in df.columns:
        df['2. Test Doğru Sayısı'] = df['son_test_puan'].combine_first(df['son_test'])
    elif 'son_test' in df.columns: df['2. Test Doğru Sayısı'] = df['son_test']
    elif 'son_test_puan' in df.columns: df['2. Test Doğru Sayısı'] = df['son_test_puan']
    else: df['2. Test Doğru Sayısı'] = 0

    # --- SAYISAL DÖNÜŞÜM ---
    df['1. Test Doğru Sayısı'] = pd.to_numeric(df['1. Test Doğru Sayısı'], errors='coerce').fillna(0).astype(int)
    df['2. Test Doğru Sayısı'] = pd.to_numeric(df['2. Test Doğru Sayısı'], errors='coerce').fillna(0).astype(int)
    df['NET'] = df['2. Test Doğru Sayısı'] - df['1. Test Doğru Sayısı']

    # --- İSİMLERİ AYARLA ---
    if 'ad_soyad' in df.columns: df['Ad Soyad'] = df['ad_soyad']
    else: df['Ad Soyad'] = "Bilinmiyor"
    if 'no' in df.columns: df['Öğrenci No'] = df['no']
    else: df['Öğrenci No'] = 0

    # --- SORU SAYISI (KRİTİK KISIM BURASI) ---
    # Eğer veritabanından gelen veride 'toplam_soru' varsa onu kullan.
    # Yoksa varsayılan (o anki dersin sorusu) değerini kullan.
    varsayilan = soru_sayisi_input if (soru_sayisi_input and soru_sayisi_input > 0) else 15
    
    if 'toplam_soru' in df.columns:
        df['Soru Sayısı'] = df['toplam_soru'].fillna(varsayilan).astype(int)
    else:
        df['Soru Sayısı'] = varsayilan

    # --- TABLO SÜTUNLARINI SEÇ ---
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
        ord('’'):"'", '‘':"'", '“':'"', '”':'"', '–':'-'
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
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Video ses dönüştürme hatası (FFmpeg): {result.stderr}")
            return False
            
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            st.error("Ses dosyası oluşturulamadı veya boş.")
            return False
            
        return True
    except Exception as e:
        st.error(f"Sistem Hatası: {e}")
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
    
# --- PDF OLUŞTURUCU ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Kisisellestirilmis Calisma Plani', 0, 1, 'C')
        self.ln(5)

    def topic_section(self, title, summary, extra_info, is_mistake, include_extra):
        if is_mistake:
            self.set_text_color(200, 0, 0)
            title = f"(!) {title} - [TEKRAR ET]"
        else:
            self.set_text_color(0, 100, 0)
            title = f"{title} (Tamamlandi)"
            
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, safe_text(title), ln=1)
        
        self.set_text_color(0)
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, safe_text(summary))
        self.ln(2)
        
        if include_extra and extra_info:
            self.set_text_color(80, 80, 80)
            self.set_font('Arial', 'I', 10)
            self.multi_cell(0, 6, safe_text(f"[EK KAYNAK]: {extra_info}"))
            self.ln(2)
            
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

def create_study_pdf(data, mistakes, include_extra=True):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    type_str = "Detayli Rapor (Ek Kaynakli)" if include_extra else "Ozet Rapor"
    pdf.cell(0, 10, safe_text(f"Rapor Turu: {type_str}"), ln=1, align='C')
    pdf.ln(5)
    
    for i, item in enumerate(data):
        baslik = item.get('alt_baslik', 'Konu')
        ozet = item.get('ozet', '')
        ek_bilgi = item.get('ek_bilgi', '')
        is_mistake = i in mistakes
        
        pdf.topic_section(baslik, ozet, ek_bilgi, is_mistake, include_extra)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= ARAYÜZ (SADE VE 2 SEKMELİ ADMIN) =================

st.title("☁️ Gemini Eğitim Platformu (Cloud v4 Stable)")

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

# --- ADIM 1: YÖNETİCİ PANELİ (2 SEKMELİ) ---
elif st.session_state['step'] == 1 and st.session_state['user_role'] == 'admin':
    st.header("Yönetici Paneli")
    
    # İki sekme oluşturuyoruz: Video Yükleme ve Sonuçlar
    tab_upload, tab_results = st.tabs(["📚 Ders İşle / Video Yükle", "📊 Sınav Sonuçları"])
    
    # 1. SEKME: VİDEO YÜKLEME
    with tab_upload:
        st.subheader("Yeni Ders İçeriği Yükle")
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
    
    # 2. SEKME: SINAV SONUÇLARI
    with tab_results:
        st.subheader("Öğrenci Sınav Sonuçları")
        if st.button("Sonuçları Gör / Yenile"):
            data_raw = get_class_data_from_firebase()
            if data_raw:
                df_raw = pd.DataFrame(data_raw)
                
                # O anki yüklü dersin soru sayısını yedek (varsayılan) olarak alıyoruz
                # 999 görürsen firebase kontrolü yap, hata ayıkla
                varsayilan_soru = len(st.session_state['data']) if st.session_state['data'] else 999
                
                # Fonksiyonu çağırırken veritabanı öncelikli çalışacak
                df_clean = format_data_for_csv(df_raw, soru_sayisi_input=varsayilan_soru)
                
                st.dataframe(df_clean, use_container_width=True)
                
                csv = df_clean.to_csv(sep=';', index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Tabloyu Excel (CSV) Olarak İndir",
                    data=csv,
                    file_name="ogrenci_sinav_sonuclari.csv",
                    mime="text/csv"
                )
            else: 
                st.info("Henüz veritabanında sonuç yok.")

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
    st.success(f"Ön Test Puanın: {st.session_state['scores']['pre']}")
    
    if st.session_state['mistakes']:
        st.warning(f"⚠️ Toplam {len(st.session_state['mistakes'])} konuda eksiğin var.")
    else:
        st.balloons()
        st.success("🎉 Tebrikler! Hiç eksiğin yok.")

    # --- PDFLERİ HAZIRLA ---
    pdf_ozet = create_study_pdf(st.session_state['data'], st.session_state['mistakes'], include_extra=False)
    pdf_full = create_study_pdf(st.session_state['data'], st.session_state['mistakes'], include_extra=True)

    # --- KONTROL PANELİ ---
    with st.container(border=True):
        col_pdf, col_speed, col_next = st.columns([2, 1, 1], gap="medium")
        
        with col_pdf:
            st.markdown("### 📄 Planı İndir")
            c1, c2 = st.columns(2)
            c1.download_button("📥 Özet İndir", pdf_ozet, "Ozet.pdf", "application/pdf", use_container_width=True)
            c2.download_button("📑 Detaylı İndir", pdf_full, "Detayli.pdf", "application/pdf", use_container_width=True)
        
        with col_speed:
            st.markdown("### 🎚️ Hız")
            audio_speed = st.select_slider("Ses Hızı", options=[0.75, 1.0, 1.25, 1.5, 2.0], value=1.0, label_visibility="collapsed")

        with col_next:
            st.markdown("### 🚀 Bitir")
            if st.button("Son Sınava Geç ➡️", use_container_width=True, type="primary"):
                st.session_state['step'] = 4
                st.rerun()

    st.divider()
    st.markdown("### 📝 Konu Listesi")

    # --- YENİ KART (CARD) TASARIMI ---
    for i, item in enumerate(st.session_state['data']):
        is_wrong = i in st.session_state['mistakes']
        
        # Her konu bir "Kutu" (Container) içinde olacak
        with st.container(border=True):
            
            # 1. BAŞLIK ALANI (Kutunun en üstü)
            if is_wrong:
                st.error(f"❌ {item['alt_baslik']} - [TEKRAR ET]", icon="⚠️")
            else:
                st.success(f"✅ {item['alt_baslik']} - [TAMAMLANDI]", icon="🎉")

            # 2. ÖZET VE DİNLEME BUTONU (Yan Yana)
            col_ozet, col_btn = st.columns([5, 1])
            
            with col_ozet:
                st.markdown(f"**📖 Özet:** {item['ozet']}")
            
            with col_btn:
                # Butonu dikeyde ortalamak için boşluk bırakabiliriz veya direkt koyarız
                st.write("") 
                if st.button("🔊 Dinle", key=f"d_{i}", help="Özeti Sesli Oku"):
                    with st.spinner("Ses hazırlanıyor..."):
                        p = generate_audio_openai(item['ozet'], st.session_state['audio_speed'])
                        if p: st.audio(p, autoplay=True)

            # 3. EK KAYNAK ALANI (Özetin hemen altında, kutunun içinde)
            ek_bilgi = item.get('ek_bilgi')
            if ek_bilgi:
                # Expander da bu container'ın sınırları içinde kalır
                with st.expander("📚 Akademik Ek Kaynak (Detaylı Bilgi)"):
                    st.info(ek_bilgi)
                    
                    # Ek kaynak dinleme butonu (Expander açılınca görünür)
                    if st.button("🎧 Ek Kaynağı Dinle", key=f"ed_{i}"):
                        with st.spinner("Ek kaynak seslendiriliyor..."):
                            p = generate_audio_openai(ek_bilgi, st.session_state['audio_speed'])
                            if p: st.audio(p, autoplay=True)
        
        st.divider() # Konular arasına çizgi

        st.write("---")

# --- ADIM 4: SON TEST (TOPLAM SORU EKLENDİ) ---
elif st.session_state['step'] == 4:
    with st.form("post_test"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            st.write(f"**{i+1})** {q['soru']}")
            
            secenekler = [q.get('A'), q.get('B'), q.get('C'), q.get('D')]
            secenekler = [s for s in secenekler if s]
            
            ans[i] = st.radio("Cevap", secenekler, key=f"son_{i}", index=None)
            st.write("---")
        
        if st.form_submit_button("Sınavı Bitir"):
            score = 0
            for i, item in enumerate(st.session_state['data']):
                q = item['soru_data']
                correct = q.get(q['dogru_sik'].strip())
                if ans.get(i) == correct: score += 1
            
            res = {
                "ad_soyad": st.session_state['student_info'].get('name', 'Bilinmiyor'),
                "no": st.session_state['student_info'].get('no', '0'),
                "tarih": time.strftime("%Y-%m-%d %H:%M"),
                "on_test": st.session_state['scores'].get('pre', 0),
                "son_test": score,
                "toplam_soru": len(st.session_state['data']) 
            }
            if save_results_to_firebase(res):
                st.balloons()
                st.success(f"Sınav Bitti! Puan: {score} / {len(st.session_state['data'])}")













