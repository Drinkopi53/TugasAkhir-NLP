# Dokumentasi Perbaikan Sistem Deteksi Code-Switching

**Proyek:** Implementasi Naive Bayes dan TF-IDF untuk Deteksi Kode-Switching Bahasa Indonesia-Inggris pada Teks Media Sosial

**Tanggal:** 6 Januari 2026

---

## Ringkasan Masalah

Sistem deteksi code-switching awalnya memiliki beberapa masalah akurasi:

| Masalah | Deskripsi |
|---------|-----------|
| **Distribusi label tidak seimbang** | 77.7% data dilabeli sebagai MIX, menyebabkan model bias |
| **Threshold terlalu sensitif** | Logika `if id_score >= 1 and en_score >= 1` menyebabkan over-classification MIX |
| **Kata ambiguous** | Kata seperti `well`, `down`, `on`, `in` memicu false positive |
| **Kelas EN lemah** | Model tidak bisa memprediksi bahasa Inggris dengan baik |
| **Hasil simulasi tidak konsisten** | `generator_gold_standart.py` menghasilkan nilai berbeda setiap dijalankan |

---

## Perbaikan yang Dilakukan

### 1. Perbaikan Logika Pelabelan (Weak Supervision)

**File:** `prototype_model.py`, `app_website.py`

**Sebelum:**
```python
if id_score >= 1 and en_score >= 1: return 'MIX'
elif en_score > id_score: return 'EN'
else: return 'ID'
```

**Sesudah:**
```python
total_markers = id_score + en_score
if total_markers == 0: return 'ID'

id_ratio = id_score / total_markers
en_ratio = en_score / total_markers

# MIX: kedua bahasa harus seimbang (25-75%) DAN minimal 2 kata masing-masing
if id_score >= 2 and en_score >= 2 and 0.25 <= id_ratio <= 0.75:
    return 'MIX'
elif en_ratio > 0.6 or (en_score >= 2 and id_score == 0):
    return 'EN'
else:
    return 'ID'
```

**Penjelasan:**
- MIX hanya jika ada **minimal 2 kata** dari masing-masing bahasa
- Rasio harus **seimbang (25-75%)** untuk dianggap code-switching
- Threshold berbasis persentase, bukan angka absolut

---

### 2. Pembersihan Lexicon (Hapus Kata Ambiguous)

**File:** `prototype_model.py`, `app_website.py`

**Kata yang dihapus dari `eng_markers`:**
- Preposisi umum: `in`, `on`, `at`, `to`, `by`, `of`, `as`, `out`
- Konjungsi pendek: `or`, `but`, `so`, `if`
- Kata umum: `it`, `its`, `be`, `do`, `go`, `work`, `hi`, `yes`, `no`, `not`
- Kata yang sering muncul di konteks Indonesia: `good`, `bad`, `day`, `night`, `time`, `year`, `way`

**Kata yang dihapus dari `indo_markers`:**
- `el`, `si`, `sang`, `para` (jarang dipakai sendiri)
- `di`, `ke` (terlalu pendek, sering jadi bagian kata lain)
- `main` (ambigu dengan bahasa Inggris)

---

### 3. Implementasi Oversampling untuk Data Training

**File:** `prototype_model.py`, `app_website.py`

**Kode yang ditambahkan:**
```python
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
```

**Hasil:**
| Kelas | Sebelum Oversampling | Sesudah Oversampling |
|-------|---------------------|---------------------|
| ID | 259 | 259 |
| MIX | 163 | 259 |
| EN | 43 | 259 |
| **Total** | **465** | **777** |

---

### 4. Perbaikan Reproducibility pada Gold Standard Generator

**File:** `generator_gold_standart.py`

**Kode yang ditambahkan:**
```python
import random
random.seed(42)  # Hasil simulasi konsisten setiap dijalankan
```

---

## Hasil Setelah Perbaikan

### Distribusi Label Dataset

| Label | Sebelum | Sesudah |
|-------|---------|---------|
| **MIX** | 452 (77.7%) | 204 (35.1%) |
| **ID** | 111 (19.1%) | 324 (55.7%) |
| **EN** | 19 (3.3%) | 54 (9.3%) |

### Performa Model

- **Akurasi:** ~70%
- **Data Training:** 777 sampel (seimbang)
- **Data Testing:** 117 sampel

### Validasi Gold Standard

- **Jumlah Sampel:** 100
- **Akurasi Validasi:** 90%
- **Hasil Konsisten:** Ya (dengan `random.seed(42)`)

---

## Cara Menjalankan

### 1. Generate Dataset & Train Model
```powershell
python prototype_model.py
```

### 2. Generate Validasi Gold Standard
```powershell
python generator_gold_standart.py
```

### 3. Jalankan Web Application
```powershell
streamlit cache clear
streamlit run app_website.py
```

---

## File yang Dimodifikasi

| File | Perubahan |
|------|-----------|
| `prototype_model.py` | Logika pelabelan, lexicon, oversampling |
| `app_website.py` | Logika pelabelan, lexicon, oversampling |
| `generator_gold_standart.py` | Tambah `random.seed(42)` |

---

## Catatan Penting

1. **Model tetap menggunakan Naive Bayes + TF-IDF** sesuai judul skripsi
2. **Weak supervision** digunakan untuk pelabelan awal dataset
3. **Oversampling** membantu model belajar dari kelas minoritas (EN)
4. Setelah mengubah kode, selalu jalankan `streamlit cache clear` sebelum restart aplikasi
