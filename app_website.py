import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# KONFIGURASI HALAMAN WEBSITE
# ==========================================
st.set_page_config(
    page_title="Deteksi Code-Switching",
    page_icon="🇮🇩🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOGIKA BACKEND (SAMA SEPERTI PROTOTYPE)
# ==========================================

# 1. Definisi Lexicon (Kamus) - Disimpan di cache agar cepat
@st.cache_data
def get_lexicons():
    # CATATAN: Kata-kata ambiguous yang sering muncul di konteks Indonesia dihapus
    indo_markers = {
        # Kata Ganti & Tunjuk
        'aku', 'kamu', 'dia', 'kita', 'mereka', 'ini', 'itu', 'sini', 'situ', 'sana',
        'gue', 'lu', 'lo', 'gw', 'anda', 'saya', 'kalian',
        # Kata Sambung & Depan
        'dan', 'atau', 'tapi', 'tetapi', 'karena', 'krn', 'jika', 'kalau', 'kalo', 
        'yang', 'yg', 'dari', 'pada', 'dalam', 'untuk', 'utk', 'buat', 
        'dengan', 'dgn', 'sama', 'bisa', 'dapat', 'akan', 'ingin', 'mau', 'sudah', 
        'telah', 'sedang', 'lagi', 'lg', 'masih', 'belum', 'blm',
        # Kata Tanya & Seru
        'apa', 'kenapa', 'knp', 'mengapa', 'gimana', 'bagaimana', 'siapa', 'kapan', 
        'dimana', 'kok', 'sih', 'dong', 'deh', 'kan', 'yuk', 'wkwk', 'hehe', 'haha',
        'wah', 'nah', 'loh', 'lah', 'kah', 'pun',
        # Kata Kerja & Sifat Umum
        'makan', 'minum', 'tidur', 'jalan', 'lihat', 'dengar', 'baca', 'tulis', 
        'beli', 'jual', 'bayar', 'kerja', 'suka', 'cinta', 'benci', 'marah',
        'senang', 'sedih', 'takut', 'berani', 'malu', 'bangga', 'bagus', 'jelek',
        'baik', 'jahat', 'benar', 'salah', 'cepat', 'lambat', 'mahal', 'murah',
        'terima', 'kasih', 'tolong', 'maaf', 'selamat', 'pagi', 'siang', 'malam',
        'rumah', 'orang', 'anak', 'hari', 'tahun', 'waktu', 'uang', 'harga',
        'tidak', 'tak', 'gak', 'ga', 'nggak', 'bukan', 'jangan', 'usah', 'udah'
    }

    eng_markers = {
        # Pronouns & Prepositions (hapus yang ambiguous: in, on, as, be, so)
        'i', 'you', 'he', 'she', 'we', 'they', 'this', 'that', 'these', 'those',
        'my', 'your', 'his', 'her', 'our', 'their', 'mine', 'yours',
        'for', 'with', 'from', 'about', 'into', 'through', 'after', 'over', 'between', 'against',
        # Conjunctions & Verbs (Auxiliary) - hapus yang ambiguous
        'and', 'because', 'when', 'where', 'why', 'how',
        'is', 'am', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'does', 'did', 'done',
        'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
        # Common Verbs (Action)
        'want', 'need', 'know', 'think', 'take', 'see', 'get', 'give', 'come',
        'make', 'look', 'use', 'find', 'tell', 'ask', 'seem', 'feel', 'try',
        'leave', 'call', 'drink', 'eat', 'sleep', 'run', 'walk', 'talk', 'speak',
        'say', 'help', 'start', 'stop', 'move', 'write', 'read', 'pay', 'buy', 'sell',
        # Common Adjectives & Adverbs (hapus yang ambiguous: not, no, well, good, bad, etc)
        'great', 'high', 'low', 'big', 'small', 'long', 'short',
        'new', 'old', 'right', 'wrong', 'happy', 'sad', 'angry', 'afraid', 'brave',
        'beautiful', 'ugly', 'expensive', 'cheap', 'fast', 'slow', 'hard', 'soft',
        'actually', 'literally', 'basically', 'totally', 'honestly', 'probably',
        'maybe', 'please', 'thanks', 'sorry', 'excuse', 'hello', 'bye',
        'yeah', 'yep', 'nope', 'never', 'always', 'ever',
        'people', 'life', 'man', 'woman', 'love', 'really', 'very', 'just'
    }
    return indo_markers, eng_markers

# 2. Fungsi Pelabelan Otomatis (Ratio-Based Threshold)
def automated_labeling(text, indo_markers, eng_markers):
    """
    Pelabelan otomatis dengan threshold berbasis RASIO.
    MIX hanya jika kedua bahasa cukup seimbang (25-75%) DAN minimal 2 kata masing-masing.
    """
    if not isinstance(text, str): return 'ID'
    text_clean = text.lower()
    text_clean = re.sub(r'[^a-z\s]', ' ', text_clean)
    words = text_clean.split()
    if len(words) == 0: return 'ID'
    
    word_set = set(words)
    id_score = len(word_set.intersection(indo_markers))
    en_score = len(word_set.intersection(eng_markers))
    
    total_markers = id_score + en_score
    
    # Jika tidak ada marker sama sekali, default ke ID
    if total_markers == 0: return 'ID'
    
    # Hitung rasio
    id_ratio = id_score / total_markers
    en_ratio = en_score / total_markers
    
    # MIX: kedua bahasa harus cukup seimbang (25-75%) DAN minimal 2 kata masing-masing
    if id_score >= 2 and en_score >= 2 and 0.25 <= id_ratio <= 0.75:
        return 'MIX'
    # EN: mayoritas marker adalah English
    elif en_ratio > 0.6 or (en_score >= 2 and id_score == 0):
        return 'EN'
    # ID: default atau mayoritas Indonesia
    else:
        return 'ID'

# 3. Fungsi Preprocessing
def clean_text_final(text):
    text = str(text).lower()
    text = re.sub(r'\[username\]|\[url\]', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# 4. Fungsi Utama Training Model (Cached)
@st.cache_resource
def train_model():
    # Load Data
    try:
        df = pd.read_csv('codeswitch_emotion.csv', on_bad_lines='skip')
    except FileNotFoundError:
        # Dummy data jika file tidak ada (agar web tidak crash saat pertama buka)
        df = pd.DataFrame({'tweet': ["Aku stuck banget", "I love you", "Makan nasi"]})
    
    # Weak Supervision
    indo, eng = get_lexicons()
    df['label_bahasa'] = df['tweet'].apply(lambda x: automated_labeling(x, indo, eng))
    df['text_clean'] = df['tweet'].apply(clean_text_final)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], 
        df['label_bahasa'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label_bahasa']
    )
    
    # OVERSAMPLING untuk menyeimbangkan data training
    train_df = pd.DataFrame({'text': X_train.values, 'label': y_train.values})
    class_counts = train_df['label'].value_counts()
    max_count = class_counts.max()
    
    balanced_dfs = []
    for label in class_counts.index:
        class_df = train_df[train_df['label'] == label]
        oversampled = class_df.sample(n=max_count, replace=True, random_state=42)
        balanced_dfs.append(oversampled)
    
    train_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    X_train = train_balanced['text']
    y_train = train_balanced['label']
    
    # Training Pipeline
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=5000), 
        MultinomialNB(alpha=0.1)
    )
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['ID', 'EN', 'MIX'])
    
    return model, acc, report, df, cm


# ==========================================
# TAMPILAN FRONTEND (WEBSITE)
# ==========================================

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("Panel Kontrol")
    st.info("Aplikasi ini menggunakan metode **Weak Supervision** dengan algoritma **Naive Bayes**.")
    
    st.markdown("---")
    st.write("**Navigasi:**")
    page = st.radio("Pilih Halaman:", ["Demo Prediksi", "Statistik & Evaluasi", "Dataset"])
    
    st.markdown("---")
    st.caption("Tugas Akhir NLP - 2025")

# Memuat Model (Tampilkan spinner loading)
with st.spinner('Sedang melatih model dan memproses data...'):
    model, accuracy, report, df_labeled, conf_matrix = train_model()

# HALAMAN 1: DEMO PREDIKSI
if page == "Demo Prediksi":
    st.title("🇮🇩🇬🇧 Deteksi Code-Switching Otomatis")
    st.markdown("Masukkan kalimat tweet atau teks media sosial di bawah ini untuk mendeteksi bahasanya.")
    
    # Input User
    text_input = st.text_area("Masukkan Teks:", height=100, placeholder="Contoh: Jujurly aku capek banget but I have to survive")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analisis Teks", type="primary")
    
    if analyze_btn and text_input:
        # Proses Prediksi
        clean_input = clean_text_final(text_input)
        prediction = model.predict([clean_input])[0]
        proba = model.predict_proba([clean_input])[0]
        classes = model.classes_
        
        # Tampilkan Hasil dengan Style
        st.markdown("---")
        st.subheader("Hasil Analisis")
        
        # Tentukan warna berdasarkan prediksi
        if prediction == 'MIX':
            color = "orange"
            full_label = "CODE-SWITCHING (CAMPURAN)"
        elif prediction == 'EN':
            color = "blue"
            full_label = "BAHASA INGGRIS (ENGLISH)"
        else:
            color = "green"
            full_label = "BAHASA INDONESIA"
            
        st.success(f"🏷️ Prediksi Kelas: **{full_label}**")
        
        # Visualisasi Probabilitas (Bar Chart)
        st.write("📊 **Tingkat Keyakinan (Confidence Score):**")
        prob_df = pd.DataFrame({
            'Bahasa': classes,
            'Probabilitas': proba
        })
        
        # Custom Chart sederhana
        for idx, row in prob_df.iterrows():
            lang = row['Bahasa']
            score = row['Probabilitas']
            st.write(f"{lang} : {score:.2%}")
            st.progress(score)

# HALAMAN 2: STATISTIK & EVALUASI
elif page == "Statistik & Evaluasi":
    st.title("📊 Evaluasi Model")
    
    # Metrik Utama
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi Model", f"{accuracy:.2%}")
    col2.metric("Precision (MIX)", f"{report['MIX']['precision']:.2f}")
    col3.metric("Recall (MIX)", f"{report['MIX']['recall']:.2f}")
    col4.metric("F1-Score (MIX)", f"{report['MIX']['f1-score']:.2f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write("Visualisasi seberapa akurat model memprediksi setiap kelas.")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ID', 'EN', 'MIX'], 
                yticklabels=['ID', 'EN', 'MIX'], ax=ax)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Detail Laporan Klasifikasi")
    st.dataframe(pd.DataFrame(report).transpose())

# HALAMAN 3: DATASET
elif page == "Dataset":
    st.title("📂 Dataset Weak Supervision")
    st.markdown("Ini adalah dataset hasil pelabelan otomatis yang digunakan untuk melatih model.")
    
    # Distribusi Label
    st.subheader("Distribusi Label Bahasa")
    distribusi = df_labeled['label_bahasa'].value_counts()
    st.bar_chart(distribusi)
    
    # Tabel Data
    st.subheader("Sampel Data (100 Baris Pertama)")
    st.dataframe(df_labeled[['tweet', 'label_bahasa']].head(100), use_container_width=True)
    
    # Tombol Download
    csv = df_labeled.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Dataset Hasil Pelabelan (CSV)",
        csv,
        "dataset_weak_supervision.csv",
        "text/csv",
        key='download-csv'
    )