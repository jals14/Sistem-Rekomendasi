# Laporan Proyek Sistem-Rekomendasi - Rijal Zaidi Alfatih

## Project Overview

Dalam era digital saat ini, jumlah konten yang tersedia secara daring, seperti film, musik, atau produk, semakin melimpah. Kondisi ini menciptakan tantangan baru bagi pengguna untuk menemukan konten yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi menjadi salah satu solusi penting untuk menyaring informasi dan memberikan saran yang personal.

Proyek ini bertujuan untuk membangun sistem rekomendasi film berbasis data menggunakan dua pendekatan utama, yaitu Content-Based Filtering dan Collaborative Filtering. Pada pendekatan Content-Based Filtering, sistem merekomendasikan film berdasarkan kemiripan genre antar film, dengan menggunakan teknik seperti one-hot encoding dan cosine similarity. Sedangkan pada pendekatan Collaborative Filtering, sistem mempelajari pola perilaku pengguna dalam memberi rating film dan membangun model pembelajaran mesin berbasis neural network menggunakan TensorFlow.

Dengan menggabungkan teknik eksplorasi data, pemrosesan fitur, dan pemodelan, proyek ini diharapkan dapat menghasilkan sistem rekomendasi yang mampu memberikan saran film yang relevan dan meningkatkan pengalaman pengguna dalam menjelajahi katalog film.

## Business Understanding

### Problem Statements (Pernyataan Masalah)
Dengan semakin banyaknya jumlah film yang tersedia di berbagai platform digital, pengguna sering kali kesulitan untuk menemukan film yang sesuai dengan preferensi mereka. Penggunaan metode pencarian manual atau berdasarkan popularitas semata tidak cukup efektif dalam memenuhi kebutuhan personalisasi pengguna. Selain itu, tidak semua pengguna memiliki waktu atau informasi yang cukup untuk menjelajahi seluruh katalog film yang tersedia.

Permasalahan lainnya adalah bagaimana membangun sistem rekomendasi yang tidak hanya mampu menyarankan film berdasarkan genre, tetapi juga mempertimbangkan preferensi pengguna berdasarkan histori rating yang telah diberikan.

### Goals (Tujuan)
Tujuan dari proyek ini adalah untuk:

1. Membangun sistem rekomendasi film yang mampu memberikan saran film secara personal kepada pengguna.

2. Menerapkan dua pendekatan utama, yaitu:
  - Content-Based Filtering untuk merekomendasikan film berdasarkan kemiripan genre.
  - Collaborative Filtering dengan model deep learning untuk merekomendasikan film berdasarkan perilaku dan preferensi pengguna lain yang serupa.

3. Mengevaluasi performa sistem menggunakan metrik seperti Precision@K untuk menilai relevansi rekomendasi.

4. Memberikan insight kepada pengembang atau pengelola platform film digital tentang bagaimana mempersonalisasi pengalaman pengguna melalui sistem rekomendasi.

## Data Understanding
### Informasi Umum Data
Proyek ini menggunakan dua dataset utama, yaitu:
- movies.csv: berisi informasi mengenai daftar film beserta genre-nya.
- ratings.csv: berisi data rating film yang diberikan oleh pengguna.

Jumlah data:
- movies.csv: terdiri dari 9.742 data film.
- ratings.csv: terdiri dari 100.836 data rating yang diberikan oleh 610 pengguna.

### Sumber Data
Dataset yang digunakan merupakan bagian dari MovieLens Dataset, yang dapat diakses dan diunduh melalui tautan berikut:
https://grouplens.org/datasets/movielens/100k/

### Penjelasan Variabel/Fitur
movies.csv
- movieId: ID unik dari masing-masing film.
- title: Judul film, disertai dengan tahun rilis.
- genres: Genre film yang dipisahkan dengan delimiter |, seperti "Action|Adventure|Animation".
ratings.csv
- userId: ID unik dari pengguna.
- movieId: ID film yang dirating oleh pengguna (berelasi dengan movieId pada movies.csv).
- rating: Nilai rating yang diberikan (skala 0.5–5.0).
- timestamp: Waktu rating diberikan dalam format UNIX timestamp (kemudian diubah menjadi kolom datetime dan year untuk analisis lebih lanjut).
  
## Data Preparation

### Deskripsi Fitur
Pada tahap ini, dilakukan serangkaian teknik persiapan data (data preparation) agar data siap digunakan dalam proses pemodelan sistem rekomendasi. Adapun langkah-langkah yang dilakukan secara berurutan adalah sebagai berikut:

1. Menghapus Duplikasi Data
   - Menghapus duplikasi pada dataset movies berdasarkan kombinasi movieId dan title.
   - Menghapus seluruh baris duplikat pada dataset ratings.
2. Memecah Kolom Genre
   - Kolom genres yang semula berupa string dengan delimiter '|' diubah menjadi list menggunakan fungsi split().
3. One-Hot Encoding Genre
   - Menggunakan MultiLabelBinarizer untuk mengubah kolom genre menjadi fitur biner (one-hot encoding) agar bisa digunakan dalam algoritma content-based filtering.
4. Konversi Timestamp
   - Mengubah kolom timestamp pada ratings.csv ke dalam format datetime (datetime) untuk analisis waktu.
   - Menambahkan kolom baru bernama rating_year dari atribut waktu tersebut untuk menganalisis tren tahunan rating film.
5. Menggabungkan Dataset
   - Melakukan merge antara dataset movies dan ratings berdasarkan movieId untuk menghasilkan dataframe gabungan yang lebih kaya informasi (movie_rating).
6. Seleksi dan Pembuatan Fitur
   - Memilih fitur-fitur yang relevan seperti movieId, title, dan hasil one-hot encoding genre untuk digunakan dalam content-based filtering.
   - Menyiapkan data userId, movieId, dan rating sebagai input untuk model collaborative filtering.
7. Normalisasi Rating
   - Melakukan normalisasi nilai rating ke dalam rentang 0–1 sebelum digunakan dalam pelatihan model neural network.
8. Pemisahan Data
   - Dataset dibagi menjadi data pelatihan (x_train, y_train) dan data validasi (x_val, y_val) menggunakan train_test_split.

## Modeling and Result

Pada tahap ini, dilakukan pembangunan dua jenis sistem rekomendasi untuk menyelesaikan permasalahan kesulitan pengguna dalam menemukan film yang sesuai dengan preferensi mereka, yaitu:

### 1. Content-Based Filtering
Metode ini merekomendasikan film berdasarkan kemiripan genre antar film. Langkah-langkah yang dilakukan adalah:
- Mengubah genre film menjadi representasi numerik dengan one-hot encoding.
- Menghitung kemiripan antar film menggunakan **cosine similarity**.
- Membuat fungsi rekomendasi yang menerima input `movieId` dan menghasilkan **Top-N film yang mirip**.

**Contoh Output Top-5 Recommendation:**  
Untuk film **"Toy Story (1995)"**, sistem merekomendasikan 5 film teratas yang memiliki genre serupa:
1. Toy Story 2  
2. Monsters, Inc.
3. Antz
4. Adventure of Rocky and Bullwinkle, The
5. Emperor's New Groove, The

---

### 2. Collaborative Filtering
Metode ini memanfaatkan rating yang diberikan pengguna untuk mempelajari preferensi mereka dan mencari kemiripan antar pengguna. Langkah-langkahnya meliputi:
- Melakukan encoding pada `userId` dan `movieId`.
- Membangun model neural network (deep learning) dengan **TensorFlow Keras**, menggunakan embedding layer untuk pengguna dan film.
- Melatih model dengan data rating yang telah dinormalisasi.
- Mengevaluasi performa model menggunakan **Root Mean Squared Error (RMSE)**.

**Hasil Evaluasi Model:**
- RMSE training dan validasi menunjukkan performa yang stabil setelah beberapa epoch.
- Model mampu menghasilkan **Top-10 rekomendasi** film untuk masing-masing pengguna.

**Contoh Output Rekomendasi untuk 1 User:**
1. Persuasion (1995)  
2. Shawshank Redemption, The (1994)  
3. Streetcar Named Desire, A (1951)
4. Kawrence if Arabia (1962)
5. Swept Away (Travolti da un insolito destino nell'azzurro mare d'agosto)
6. Jules and Jim (Jules et Jim (1961)
7. Lady Eve, The (1941)
8. Trial, The (1962)
9. Baby Driver (2017)
10. Three Billboards Outside Ebbing, Missouri (2017)


## Evaluation

### Metrik Evaluasi yang Digunakan

Dalam proyek ini digunakan dua pendekatan sistem rekomendasi, sehingga digunakan dua metrik evaluasi yang berbeda sesuai konteks masing-masing:

1. **Content-Based Filtering**
   - **Precision**: Metrik ini mengukur proporsi film yang direkomendasikan dalam Top-K yang benar-benar disukai oleh pengguna (berdasarkan histori rating ≥ 4).
   - Evaluasi dilakukan dengan mengambil sampel 100 pengguna dan menghitung rata-rata nilai Precision@5.

2. **Collaborative Filtering (Deep Learning)**
   - **Root Mean Squared Error (RMSE)**: Metrik ini mengukur selisih rata-rata antara nilai rating aktual dengan rating yang diprediksi oleh model. Semakin kecil nilai RMSE, semakin baik akurasi prediksi model.

---

### Hasil Evaluasi

- **Precision@5 (Content-Based)**  
  diperoleh nilai rata-rata **Precision sebesar 0.0920**. 

- **RMSE (Collaborative Filtering)**  
  Berdasarkan hasil training dan validasi model, nilai Root Mean Squared Error yang diperoleh adalah:  
  - **RMSE pada data training: 0.0374**
  - **RMSE pada data validasi: 0.1934** 

### Kesimpulan Evaluasi

Kedua pendekatan sistem rekomendasi menunjukkan hasil yang baik sesuai dengan metrik masing-masing. Content-based filtering efektif dalam memberikan rekomendasi berbasis kesamaan genre, sementara collaborative filtering berhasil memodelkan pola interaksi pengguna dan memberikan rekomendasi yang bersifat personal.

Pemilihan metrik evaluasi sudah sesuai dengan problem statement dan konteks data: Precision@K cocok untuk mengukur relevansi rekomendasi, dan RMSE cocok untuk menilai akurasi prediksi model dalam regresi rating.
