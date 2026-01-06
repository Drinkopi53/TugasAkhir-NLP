import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Fungsi pembantu untuk membuat Header Output agar rapi
def print_header(title):
    print("\n" + "="*70)
    print(f" {title} ".center(70, "="))
    print("="*70)

def print_separator():
    print("-" * 70)

# ==============================================================================
# TAHAP 1: WEAK SUPERVISION (PELABELAN OTOMATIS)
# ==============================================================================

print_header("TAHAP 1: WEAK SUPERVISION (PELABELAN DATA OTOMATIS)")

# 1. Memuat Dataset Mentah
try:
    df = pd.read_csv('codeswitch_emotion.csv', on_bad_lines='skip')
    print(f"[INFO] Dataset '{'codeswitch_emotion.csv'}' berhasil dimuat.")
    print(f"[INFO] Total data mentah: {len(df)} baris.")
except FileNotFoundError:
    print("[ERROR] File dataset tidak ditemukan. Menggunakan data dummy.")
    df = pd.DataFrame({'tweet': [
        "Aku stuck banget sama deadline tugas", "I love you so much", 
        "Makan nasi goreng enak di kantin", "Which is sebenernya dia fine aja"
    ]})

# 2. Definisi Lexicon
indo_markers = {
    # Kata Ganti & Tunjuk
    'aku', 'kamu', 'dia', 'kita', 'mereka', 'ini', 'itu', 'sini', 'situ', 'sana',
    'gue', 'lu', 'lo', 'gw', 'el', 'si', 'sang', 'para', 'anda', 'saya', 'kalian',
    # Kata Sambung & Depan
    'dan', 'atau', 'tapi', 'tetapi', 'karena', 'krn', 'jika', 'kalau', 'kalo', 
    'yang', 'yg', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'utk', 'buat', 
    'dengan', 'dgn', 'sama', 'bisa', 'dapat', 'akan', 'ingin', 'mau', 'sudah', 
    'telah', 'sedang', 'lagi', 'lg', 'masih', 'belum', 'blm',
    # Kata Tanya & Seru
    'apa', 'kenapa', 'knp', 'mengapa', 'gimana', 'bagaimana', 'siapa', 'kapan', 
    'dimana', 'kok', 'sih', 'dong', 'deh', 'kan', 'yuk', 'wkwk', 'hehe', 'haha',
    'wah', 'nah', 'loh', 'lah', 'kah', 'pun',
    # Kata Kerja & Sifat Umum
    'makan', 'minum', 'tidur', 'jalan', 'lihat', 'dengar', 'baca', 'tulis', 
    'beli', 'jual', 'bayar', 'kerja', 'main', 'suka', 'cinta', 'benci', 'marah',
    'senang', 'sedih', 'takut', 'berani', 'malu', 'bangga', 'bagus', 'jelek',
    'baik', 'jahat', 'benar', 'salah', 'cepat', 'lambat', 'mahal', 'murah',
    'terima', 'kasih', 'tolong', 'maaf', 'selamat', 'pagi', 'siang', 'malam',
    'rumah', 'orang', 'anak', 'hari', 'tahun', 'waktu', 'uang', 'harga',
    'tidak', 'tak', 'gak', 'ga', 'nggak', 'bukan', 'jangan', 'usah'
}

eng_markers = {
    # Pronouns & Prepositions
    'i', 'you', 'he', 'she', 'we', 'they', 'it', 'this', 'that', 'these', 'those',
    'my', 'your', 'his', 'her', 'our', 'their', 'its', 'mine', 'yours',
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about', 'of', 'as',
    'into', 'like', 'through', 'after', 'over', 'between', 'out', 'against',
    # Conjunctions & Verbs (Auxiliary)
    'and', 'or', 'but', 'so', 'because', 'if', 'when', 'where', 'why', 'how',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'done',
    'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    # Common Verbs (Action) - INI YANG PENTING DITAMBAH
    'want', 'need', 'know', 'think', 'take', 'see', 'get', 'give', 'go', 'come',
    'make', 'look', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try',
    'leave', 'call', 'drink', 'eat', 'sleep', 'run', 'walk', 'talk', 'speak',
    'say', 'help', 'start', 'stop', 'move', 'write', 'read', 'pay', 'buy', 'sell',
    # Common Adjectives & Adverbs
    'good', 'bad', 'great', 'high', 'low', 'big', 'small', 'long', 'short',
    'new', 'old', 'right', 'wrong', 'happy', 'sad', 'angry', 'afraid', 'brave',
    'beautiful', 'ugly', 'expensive', 'cheap', 'fast', 'slow', 'hard', 'soft',
    'actually', 'literally', 'basically', 'totally', 'honestly', 'probably',
    'maybe', 'please', 'thanks', 'sorry', 'excuse', 'hello', 'hi', 'bye',
    'not', 'no', 'yes', 'yeah', 'yep', 'nope', 'never', 'always', 'ever',
    'day', 'night', 'time', 'year', 'people', 'way', 'life', 'man', 'woman'
}

def automated_labeling(text):
    if not isinstance(text, str): return 'ID'
    text_clean = text.lower()
    text_clean = re.sub(r'[^a-z\s]', ' ', text_clean)
    words = text_clean.split()
    if len(words) == 0: return 'ID'
    
    word_set = set(words)
    id_score = len(word_set.intersection(indo_markers))
    en_score = len(word_set.intersection(eng_markers))
    
    if id_score >= 1 and en_score >= 1: return 'MIX' 
    elif en_score > id_score: return 'EN'
    else: return 'ID'

# 3. Terapkan Pelabelan
print("[...] Sedang menjalankan algoritma pelabelan otomatis...")
df['label_bahasa'] = df['tweet'].apply(automated_labeling)

# 4. Simpan Dataset
output_csv = 'dataset_hasil_pelabelan.csv'
df[['tweet', 'label_bahasa']].to_csv(output_csv, index=False)
print(f"[✓] File '{output_csv}' berhasil disimpan! (Lampirkan di skripsi)")

# Statistik Tabel Rapi
print("\n[INFO] Statistik Distribusi Label:")
print_separator()
print(f"{'KELAS BAHASA':<15} | {'JUMLAH SAMPLE':<15} | {'PERSENTASE':<15}")
print_separator()
stats = df['label_bahasa'].value_counts()
total = len(df)
for label, count in stats.items():
    print(f"{label:<15} | {count:<15} | {count/total:.2%}")
print_separator()


# ==============================================================================
# TAHAP 2: PEMODELAN (NAIVE BAYES & TF-IDF)
# ==============================================================================

print_header("TAHAP 2: PELATIHAN MODEL NAIVE BAYES")

def clean_text_final(text):
    text = str(text).lower()
    text = re.sub(r'\[username\]|\[url\]', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df['text_clean'] = df['tweet'].apply(clean_text_final)

X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], 
    df['label_bahasa'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label_bahasa']
)

print(f"[INFO] Data Latih : {len(X_train)} sampel")
print(f"[INFO] Data Uji   : {len(X_test)} sampel")

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=5000), 
    MultinomialNB(alpha=0.1)
)

print("[...] Sedang melatih model...")
model.fit(X_train, y_train)
print("[✓] Model berhasil dilatih.")


# ==============================================================================
# TAHAP 3: EVALUASI & VISUALISASI
# ==============================================================================

print_header("TAHAP 3: EVALUASI & HASIL")

y_pred = model.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
print(f"\n{' AKURASI MODEL ':*^40}")
print(f"{acc_score:.2%}".center(40))
print("*" * 40 + "\n")

print("[INFO] Detail Laporan Klasifikasi:")
print(classification_report(y_test, y_pred))

try:
    cm = confusion_matrix(y_test, y_pred, labels=['ID', 'EN', 'MIX'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ID', 'EN', 'MIX'], 
                yticklabels=['ID', 'EN', 'MIX'])
    plt.title('Confusion Matrix: Deteksi Code-Switching')
    plt.ylabel('Label Asli (Weak Supervision)')
    plt.xlabel('Prediksi Model')
    plt.savefig('confusion_matrix.png')
    print("[✓] Gambar 'confusion_matrix.png' berhasil disimpan.")
except Exception as e:
    print(f"[WARNING] Gagal membuat gambar confusion matrix: {e}")