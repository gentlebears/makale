import streamlit as st
import whisper
import os
import tempfile
import google.generativeai as genai
import json
import subprocess
import nest_asyncio
import pandas as pd
import time
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

# --- FIREBASE ---
db = None 
if not firebase_admin._apps:
    try:
        key_dict = dict(st.secrets["firebase"])
        key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
    except: st.stop()
try: db = firestore.client()
except: pass

# --- API BAĞLANTILARI ---
client = None 
try:
    genai.configure(api_key=gemini_api_key)
    client = OpenAI(api_key=openai_api_key)
except: pass 

# --- BASİT STATE (Karmaşık yapı kaldırıldı) ---
if 'step' not in st.session_state: st.session_state['step'] = 0
if 'data' not in st.session_state: st.session_state['data'] = []
if 'mistakes' not in st.session_state: st.session_state['mistakes'] = []
if 'scores' not in st.session_state: st.session_state['scores'] = {'pre':0, 'post':0}
if 'student_info' not in st.session_state: st.session_state['student_info'] = {}

# --- YARDIMCI FONKSİYONLAR ---
def save_results(data):
    if db:
        try: db.collection('exam_results').document(str(data['no'])).set(data); return True
        except: return False
    return False

def get_results():
    if db:
        try: return [d.to_dict() for d in db.collection('exam_results').stream()]
        except: return []
    return []

@st.cache_resource
def load_whisper(): return whisper.load_model("base", device="cpu")

def audio_extract(video, audio):
    cmd = ["ffmpeg", "-i", video, "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-y", audio]
    try: subprocess.run(cmd, capture_output=True); return True
    except: return False

def analyze_text(text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Video metnini analiz et.
        Çıktı JSON formatında olmalı: [{{ "alt_baslik": "...", "ozet": "...", "ek_bilgi": "...", "soru_data": {{ "soru": "...", "A": "...", "B": "...", "C": "...", "D": "...", "dogru_sik": "A" }} }}]
        METİN: {text[:15000]}
        """
        res = model.generate_content(prompt)
        clean = res.text.replace("```json", "").replace("```", "").strip()
        s, e = clean.find('['), clean.rfind(']') + 1
        return json.loads(clean[s:e])
    except: return []

def tts(text):
    if not client: return None
    try:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        res = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
        res.stream_to_file(tf.name)
        return tf.name
    except: return None

# --- GÜVENLİ PDF (Çökme Önleyici) ---
class PDF(FPDF):
    def header(self):
        try: 
            # Font varsa kullan, yoksa hata verme Arial kullan
            self.set_font('Roboto', 'B', 14)
        except:
            self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Calisma Plani', 0, 1, 'C'); self.ln(5)

    def topic(self, title, body, extra, wrong):
        # Başlık Rengi
        if wrong: self.set_text_color(200,0,0); title = f"(!) {title}"
        else: self.set_text_color(0,100,0)
        
        # Başlık Fontu
        try: self.set_font('Roboto', 'B', 12)
        except: self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=1)
        
        # İçerik Rengi ve Fontu
        self.set_text_color(0)
        try: self.set_font('Roboto', '', 10)
        except: self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, body); self.ln(2)
        
        # Ek Bilgi
        if extra:
            self.set_text_color(80)
            self.multi_cell(0, 6, f"EK: {extra}"); self.ln(2)
        self.line(10, self.get_y(), 200, self.get_y()); self.ln(5)

@st.cache_data(show_spinner=False)
def make_pdf(data, mistakes, detailed=False):
    pdf = PDF()
    
    # FONT YÜKLEME (HATA OLURSA GEÇ)
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        pdf.add_font('Roboto', '', os.path.join(base, 'Roboto-Regular.ttf'), uni=True)
        pdf.add_font('Roboto', 'B', os.path.join(base, 'Roboto-Bold.ttf'), uni=True)
    except: pass # Font yüklenemezse varsayılan fontlar çalışır
    
    pdf.add_page()
    for i, d in enumerate(data):
        pdf.topic(d['alt_baslik'], d['ozet'], d['ek_bilgi'] if detailed else "", i in mistakes)
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ================= ARAYÜZ =================
st.title("📚 Eğitim Platformu")

# STEP 0: GİRİŞ
if st.session_state['step'] == 0:
    t1, t2 = st.tabs(["Öğrenci", "Yönetici"])
    with t1:
        n = st.text_input("Ad Soyad")
        no = st.text_input("No")
        if st.button("Başla") and n and no:
            # Json dosyasını kontrol et
            if os.path.exists("lesson_data.json"):
                with open("lesson_data.json", "r", encoding="utf-8") as f:
                    st.session_state['data'] = json.load(f)
                st.session_state['student_info'] = {'name':n, 'no':no}
                st.session_state['step'] = 2
                st.rerun()
            else: st.error("Ders verisi bulunamadı.")
    with t2:
        if st.text_input("Şifre", type="password") == ADMIN_PASSWORD and st.button("Gir"):
            st.session_state['step'] = 1; st.rerun()

# STEP 1: YÖNETİCİ
elif st.session_state['step'] == 1:
    up = st.file_uploader("Video Yükle (.mp4)", ["mp4"])
    if up and st.button("Analiz Et"):
        with st.spinner("İşleniyor..."):
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tf.write(up.read())
            aud = tf.name.replace(".mp4", ".mp3")
            if audio_extract(tf.name, aud):
                w = load_whisper(); txt = w.transcribe(aud)['text']
                data = analyze_text(txt)
                if data:
                    with open("lesson_data.json", "w", encoding="utf-8") as f: json.dump(data, f)
                    st.session_state['data'] = data; st.success("Ders Hazır!")
    
    if st.button("Sonuçları İndir"):
        res = get_results()
        if res:
            df = pd.DataFrame(res)
            st.dataframe(df)
            st.download_button("CSV", df.to_csv(), "sonuc.csv")

# STEP 2: ÖN TEST
elif st.session_state['step'] == 2:
    st.info("Lütfen Soruları Cevaplayınız")
    with st.form("test1"):
        ans = {}
        for i, d in enumerate(st.session_state['data']):
            q = d['soru_data']
            st.write(f"**{i+1})** {q['soru']}")
            ans[i] = st.radio("", [q['A'],q['B'],q['C'],q['D']], key=f"q{i}")
            st.divider()
        if st.form_submit_button("Testi Bitir"):
            sc, mis = 0, []
            for i, d in enumerate(st.session_state['data']):
                if ans[i] == d['soru_data'][d['soru_data']['dogru_sik']]: sc += 1
                else: mis.append(i)
            st.session_state['scores']['pre'] = sc
            st.session_state['mistakes'] = mis
            st.session_state['step'] = 3
            st.rerun()

# STEP 3: ÇALIŞMA EKRANI (Sorunlu kısım burasıydı - Sadeleştirildi)
elif st.session_state['step'] == 3:
    st.metric("İlk Test Puanı", st.session_state['scores']['pre'])
    
    # PDF İNDİRME ALANI (Try-Except ile korumalı)
    c1, c2, c3 = st.columns(3)
    try:
        pdf_ozet = make_pdf(st.session_state['data'], st.session_state['mistakes'], False)
        c1.download_button("📥 Özet İndir", pdf_ozet, "Ozet.pdf", "application/pdf")
        
        pdf_detay = make_pdf(st.session_state['data'], st.session_state['mistakes'], True)
        c2.download_button("📑 Detaylı İndir", pdf_detay, "Detayli.pdf", "application/pdf")
    except:
        st.error("PDF oluşturulamadı.")
    
    c3.button("Son Teste Geç ➡️", on_click=lambda: st.session_state.update({'step': 4}))

    st.divider()
    
    # İÇERİK LİSTESİ
    for i, d in enumerate(st.session_state['data']):
        err = i in st.session_state['mistakes']
        box = st.error if err else st.success
        
        with box(f"{'Eksik Konu: ' if err else 'Tamam: '} {d['alt_baslik']}"):
            # 1. ÖZET
            c_txt, c_btn = st.columns([8, 1])
            c_txt.write(f"**Özet:** {d['ozet']}")
            if c_btn.button("🔊", key=f"s{i}"): # Basit buton
                 p = tts(d['ozet'])
                 if p: st.audio(p, autoplay=True)

            # 2. EK BİLGİ (Expander içinde)
            with st.expander("Ek Bilgi ve Kaynak"):
                ce_txt, ce_btn = st.columns([8, 1])
                ce_txt.info(d['ek_bilgi'])
                if ce_btn.button("🎧", key=f"e{i}"): # Basit buton
                    p = tts(d['ek_bilgi'])
                    if p: st.audio(p, autoplay=True)

# STEP 4: SON TEST (HATA DÜZELTİLMİŞ HALİ)
elif st.session_state['step'] == 4:
    with st.form("test2"):
        ans = {}
        for i, d in enumerate(st.session_state['data']):
            q = d['soru_data']
            st.write(f"**{i+1})** {q['soru']}")
            # Radyo butonu seçenekleri
            secenekler = [q.get('A',''), q.get('B',''), q.get('C',''), q.get('D','')]
            # Boş seçenekleri filtrele (Hata önleyici)
            secenekler = [s for s in secenekler if s] 
            
            ans[i] = st.radio("", secenekler, key=f"q2{i}")
            st.divider()
            
        if st.form_submit_button("Tamamla"):
            sc = 0
            for i, d in enumerate(st.session_state['data']):
                try:
                    # 1. Doğru şıkkın harfini al ve temizle (örn: "A " -> "A")
                    dogru_sik_harfi = d['soru_data']['dogru_sik'].strip()
                    
                    # 2. O harfin metnini al
                    dogru_cevap_metni = d['soru_data'][dogru_sik_harfi]
                    
                    # 3. Kullanıcının cevabıyla karşılaştır
                    if ans.get(i) == dogru_cevap_metni:
                        sc += 1
                except:
                    # Eğer veri bozuksa veya key bulunamazsa puan verme ama çökme
                    pass
            
            # Kaydet
            res = {
                "ad_soyad": st.session_state['student_info'].get('name', 'Bilinmiyor'),
                "no": st.session_state['student_info'].get('no', '0'),
                "on_test": st.session_state['scores'].get('pre', 0),
                "son_test": sc,
                "toplam_soru": len(st.session_state['data']),
                "tarih": time.strftime("%Y-%m-%d %H:%M")
            }
            save_results(res)
            st.balloons()
            st.success(f"Tebrikler! Son Puan: {sc} / {len(st.session_state['data'])}")
