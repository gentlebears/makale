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
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from fpdf import FPDF
from openai import OpenAI 

# --- AYARLAR ---
st.set_page_config(page_title="Gemini Eğitim Platformu", layout="wide")
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
        'step': 0, 'user_role': None, 'student_info': {},
        'scores': {'pre': 0, 'post': 0}, 'mistakes': [],
        'data': [], 'audio_speed': 1.0,
        'audio_cache': {} # SES DOSYALARINI HATIRLAMAK İÇİN
    }
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val
init_state()

# --- FIREBASE KAYIT ---
def save_results_to_firebase(student_data):
    if db is None: return False
    try:
        doc_ref = db.collection('exam_results').document(str(student_data['no']))
        doc_ref.set(student_data)
        return True
    except: return False

def get_class_data_from_firebase():
    if db is None: return []
    try:
        docs = db.collection('exam_results').stream()
        return [doc.to_dict() for doc in docs]
    except: return []

# --- VERİ DÜZELTME MOTORU ---
def format_data_for_csv(df, soru_sayisi_input=None):
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

    df['1. Test Doğru Sayısı'] = pd.to_numeric(df['1. Test Doğru Sayısı'], errors='coerce').fillna(0).astype(int)
    df['2. Test Doğru Sayısı'] = pd.to_numeric(df['2. Test Doğru Sayısı'], errors='coerce').fillna(0).astype(int)
    df['NET'] = df['2. Test Doğru Sayısı'] - df['1. Test Doğru Sayısı']
    
    if 'ad_soyad' in df.columns: df['Ad Soyad'] = df['ad_soyad']
    else: df['Ad Soyad'] = "Bilinmiyor"
    if 'no' in df.columns: df['Öğrenci No'] = df['no']
    else: df['Öğrenci No'] = 0

    varsayilan = soru_sayisi_input if (soru_sayisi_input and soru_sayisi_input > 0) else 15
    if 'toplam_soru' in df.columns:
        df['Soru Sayısı'] = df['toplam_soru'].fillna(varsayilan).astype(int)
    else:
        df['Soru Sayısı'] = varsayilan

    target_columns = ['Ad Soyad', 'Öğrenci No', 'Soru Sayısı', '1. Test Doğru Sayısı', '2. Test Doğru Sayısı', 'NET']
    for col in target_columns:
        if col not in df.columns: df[col] = 0 if 'Sayısı' in col or 'NET' in col or 'No' in col else ""
    return df[target_columns]

def safe_text(text):
    if text is None: return ""
    return str(text)

@st.cache_resource
def load_whisper(): return whisper.load_model("base", device="cpu")

def sesi_sokup_al(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-y", audio_path]
    try: subprocess.run(command, capture_output=True); return True
    except: return False

def analyze_full_text_with_gemini(full_text):
    primary, fallback = "gemini-2.5-flash", "gemini-2.0-flash"
    try: model = genai.GenerativeModel(primary); model.generate_content("test")
    except: model = genai.GenerativeModel(fallback)
    
    if len(full_text) < 50: return []
    prompt = f"""
    Sen uzman bir eğitim asistanısın. Video transkriptini analiz et.
    1. Konuyu alt başlıklara böl.
    2. Her başlık için ÖZET çıkar.
    3. Her başlık için akademik EK BİLGİ (Extra Resource) ekle.
    4. Her başlık için bir test sorusu yaz.
    Çıktı JSON: [{{ "alt_baslik": "...", "ozet": "...", "ek_bilgi": "...", "soru_data": {{ "soru": "...", "A": "...", "B": "...", "C": "...", "D": "...", "dogru_sik": "A" }} }}]
    METİN: "{full_text}"
    """
    try:
        res = model.generate_content(prompt)
        text = res.text.replace("```json", "").replace("```", "").strip()
        s, e = text.find('['), text.rfind(']') + 1
        return json.loads(text[s:e])
    except: return []

def generate_audio_openai(text, speed):
    if not client or len(text) < 2: return None
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        res = client.audio.speech.create(model="tts-1", voice="alloy", input=text, speed=speed)
        res.stream_to_file(tfile.name)
        return tfile.name
    except: return None

# --- PDF SINIFI ---
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font('Roboto', '', 'Roboto-Regular.ttf', uni=True)
        self.add_font('Roboto', 'B', 'Roboto-Bold.ttf', uni=True)

    def header(self):
        self.set_font('Roboto', 'B', 14)
        self.cell(0, 10, 'Kişiselleştirilmiş Çalışma Planı', 0, 1, 'C'); self.ln(5)

    def topic_section(self, title, summary, extra, mistake, include_extra):
        if mistake:
            self.set_text_color(200, 0, 0); title = f"(!) {title} - [TEKRAR ET]"
        else:
            self.set_text_color(0, 100, 0); title = f"{title} (Tamamlandı)"
        
        self.set_font('Roboto', 'B', 12)
        self.cell(0, 10, title, ln=1)
        
        self.set_text_color(0)
        self.set_font('Roboto', '', 10)
        self.multi_cell(0, 6, summary); self.ln(2)
        
        if include_extra and extra:
            self.set_text_color(80)
            self.set_font('Roboto', '', 9)
            self.multi_cell(0, 6, f"[EK KAYNAK]: {extra}"); self.ln(2)
        
        self.set_draw_color(200); self.line(10, self.get_y(), 200, self.get_y()); self.ln(5)

def create_pdf(data, mistakes, extra=True):
    pdf = PDF(); pdf.add_page(); pdf.set_auto_page_break(True, 15)
    pdf.set_font("Roboto", '', 10); pdf.set_text_color(100)
    pdf.cell(0, 10, f"Rapor Türü: {'Detaylı' if extra else 'Özet'}", ln=1, align='C'); pdf.ln(5)
    for i, item in enumerate(data):
        pdf.topic_section(item['alt_baslik'], item['ozet'], item['ek_bilgi'], i in mistakes, extra)
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= ARAYÜZ =================
st.title("☁️ Gemini Eğitim Platformu")
LESSON_FILE = "lesson_data.json"
if os.path.exists(LESSON_FILE) and not st.session_state['data']:
    try: 
        with open(LESSON_FILE,'r',encoding='utf-8') as f: st.session_state['data'] = json.load(f)
    except: pass

# --- SAYFALAR ---
if st.session_state['step'] == 0:
    t1, t2 = st.tabs(["Öğrenci", "Öğretmen"])
    with t1:
        s_name = st.text_input("Ad Soyad")
        s_no = st.text_input("Öğrenci No")
        if st.button("Başla") and s_name and s_no:
            if st.session_state['data']:
                st.session_state.update({'student_info': {'name':s_name, 'no':s_no}, 'user_role':'student', 'step':2})
                st.rerun()
            else: st.error("Ders yok.")
            
    with t2:
        pwd = st.text_input("Şifre", type="password")
        if st.button("Giriş"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.update({'user_role':'admin', 'step':1}); st.rerun()
            else:
                st.error("Hatalı Şifre")

elif st.session_state['step'] == 1:
    st.header("Yönetici")
    t1, t2 = st.tabs(["Video Yükle", "Sonuçlar"])
    with t1:
        up = st.file_uploader("Video", ["mp4"])
        if up and st.button("İşle"):
            with st.spinner("Analiz ediliyor..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tfile.write(up.read())
                audio = tfile.name.replace(".mp4", ".mp3")
                if sesi_sokup_al(tfile.name, audio):
                    try:
                        w = load_whisper(); res = w.transcribe(audio)
                        an = analyze_full_text_with_gemini(res['text'])
                        if an:
                            with open(LESSON_FILE,'w',encoding='utf-8') as f: json.dump(an, f, ensure_ascii=False)
                            st.session_state['data'] = an; st.success("Hazır!")
                    except Exception as e: st.error(f"Hata: {e}")
                else: st.error("Ses ayrıştırılamadı.")
    with t2:
        if st.button("Yenile"):
            raw = get_class_data_from_firebase()
            if raw:
                varsayilan_soru = len(st.session_state['data']) if st.session_state['data'] else 15
                df = format_data_for_csv(pd.DataFrame(raw), varsayilan_soru)
                st.dataframe(df)
                st.download_button("Excel İndir", df.to_csv(sep=';', index=False, encoding='utf-8-sig'), "sonuc.csv")

elif st.session_state['step'] == 2:
    st.info(f"Hoş geldin {st.session_state['student_info']['name']}")
    with st.form("pre"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            st.write(f"**{i+1})** {q['soru']}")
            ans[i] = st.radio("", [q['A'], q['B'], q['C'], q['D']], key=f"p_{i}")
            st.write("---")
        if st.form_submit_button("Bitir"):
            sc, mis = 0, []
            for i, item in enumerate(st.session_state['data']):
                if ans.get(i) == item['soru_data'][item['soru_data']['dogru_sik'].strip()]: sc += 1
                else: mis.append(i)
            st.session_state.update({'scores':{'pre':sc}, 'mistakes':mis, 'step':3}); st.rerun()

elif st.session_state['step'] == 3:
    st.success(f"Puanın: {st.session_state['scores']['pre']}")
    
    if st.session_state['mistakes']:
        st.warning(f"Toplam {len(st.session_state['mistakes'])} konuda eksiğin var.")
    else:
        st.balloons(); st.success("Harika! Eksiğin yok.")

    # Üst Panel Butonları
    c1, c2, c3, c4 = st.columns([1, 1, 1.5, 1])
    with c1:
        pdf1 = create_pdf(st.session_state['data'], st.session_state['mistakes'], False)
        st.download_button("📥 Özet", pdf1, "Ozet.pdf", "application/pdf", use_container_width=True)
    with c2:
        pdf2 = create_pdf(st.session_state['data'], st.session_state['mistakes'], True)
        st.download_button("📑 Detaylı", pdf2, "Detayli.pdf", "application/pdf", use_container_width=True)
    with c3:
        speed_val = st.select_slider("Ses Hızı", options=[0.75, 1.0, 1.25, 1.5, 2.0], value=1.0, label_visibility="collapsed")
        st.session_state['audio_speed'] = speed_val
    with c4:
        if st.button("➡️ Devam", use_container_width=True):
            st.session_state['step'] = 4; st.rerun()
    
    st.divider()

    # --- YENİLENMİŞ LİSTE TASARIMI ---
    for i, item in enumerate(st.session_state['data']):
        wrong = i in st.session_state['mistakes']
        box = st.error if wrong else st.success
        
        # Kutunun rengi duruma göre değişir ama içeriği aynıdır
        with box(f"{'🔻' if wrong else '✅'} {item['alt_baslik']}"):
            
            # --- 1. KISIM: ÖZET ---
            col_txt, col_btn = st.columns([8, 1])
            with col_txt: 
                st.write(item['ozet'])
            
            # Özet Sesi Butonu
            with col_btn:
                summ_key = f"sum_{i}"
                if st.button("🔊", key=f"btn_sum_{i}", help="Özeti Dinle"):
                    with st.spinner("."):
                        p = generate_audio_openai(item['ozet'], st.session_state['audio_speed'])
                        if p: st.session_state['audio_cache'][summ_key] = p
            
            # Özet Player (Varsa Göster)
            if summ_key in st.session_state['audio_cache']:
                st.audio(st.session_state['audio_cache'][summ_key])

            # --- 2. KISIM: EK BİLGİ (AÇILIR KUTU) ---
            with st.expander("📚 Ek Bilgi ve Kaynaklar"):
                col_ek_txt, col_ek_btn = st.columns([8, 1])
                
                with col_ek_txt: 
                    st.info(item['ek_bilgi'])
                
                # Ek Bilgi Sesi Butonu
                with col_ek_btn:
                    extra_key = f"ext_{i}"
                    if st.button("🎧", key=f"btn_ext_{i}", help="Ek Bilgiyi Dinle"):
                        with st.spinner("."):
                            p = generate_audio_openai(item['ek_bilgi'], st.session_state['audio_speed'])
                            if p: st.session_state['audio_cache'][extra_key] = p
                
                # Ek Bilgi Player (Varsa Göster)
                if extra_key in st.session_state['audio_cache']:
                    st.audio(st.session_state['audio_cache'][extra_key])

elif st.session_state['step'] == 4:
    with st.form("post"):
        ans = {}
        for i, item in enumerate(st.session_state['data']):
            q = item['soru_data']
            st.write(f"**{i+1})** {q['soru']}")
            ans[i] = st.radio("", [q['A'], q['B'], q['C'], q['D']], key=f"s_{i}")
            st.write("---")
        if st.form_submit_button("Bitir"):
            sc = 0
            for i, item in enumerate(st.session_state['data']):
                if ans.get(i) == item['soru_data'][item['soru_data']['dogru_sik'].strip()]: sc += 1
            
            res = {
                "ad_soyad": st.session_state['student_info']['name'], 
                "no": st.session_state['student_info']['no'],
                "tarih": time.strftime("%Y-%m-%d %H:%M"), 
                "toplam_soru": len(st.session_state['data']),
                "on_test": st.session_state['scores']['pre'], 
                "son_test": sc
            }
            if save_results_to_firebase(res): st.balloons(); st.success(f"Puan: {sc}")
