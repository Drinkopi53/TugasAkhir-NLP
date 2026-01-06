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

def automated_labeling(text):
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

# 5. OVERSAMPLING untuk menyeimbangkan data training
print("[...] Melakukan oversampling untuk menyeimbangkan kelas...")
train_df = pd.DataFrame({'text': X_train.values, 'label': y_train.values})


# Hitung jumlah sampel per kelas
class_counts = train_df['label'].value_counts()
max_count = class_counts.max()

# Oversample setiap kelas agar seimbang
balanced_dfs = []
for label in class_counts.index:
    class_df = train_df[train_df['label'] == label]
    # Oversample dengan replacement jika perlu
    oversampled = class_df.sample(n=max_count, replace=True, random_state=42)
    balanced_dfs.append(oversampled)

train_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
X_train = train_balanced['text']
y_train = train_balanced['label']

print(f"[✓] Data training diseimbangkan: {len(train_balanced)} sampel ({max_count} per kelas)")

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