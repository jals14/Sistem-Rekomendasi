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
Bagian ini menyajikan gambaran umum tentang dataset yang digunakan dalam proyek sistem rekomendasi ini, merinci asal-usulnya, dimensi, kualitas, dan sifat fitur-fiturnya. Memahami aspek-aspek ini sangat penting untuk pra-pemrosesan data dan pengembangan model yang efektif.
  1. URL/Tautan Sumber Data 🔗
     Dataset yang digunakan dalam proyek ini bersumber dari [MovieLens 25M ](https://grouplens.org/datasets/movielens/100k/) Dataset. Secara khusus, kita menggunakan file movies.csv dan ratings.csv, yang masing-masing berisi metadata film dan rating pengguna.
  2. Jumlah Baris dan Kolom 📏
     Kita menggunakan dua dataset utama:
     - movies.csv: Dataset ini berisi informasi tentang film.
       - Jumlah Baris (sebelum pra-pemrosesan): 9742 baris.
       - Jumlah Kolom (sebelum pra-pemrosesan): 3 kolom.
       Penjelasan Kolom:
         - movieId: Pengidentifikasi unik untuk setiap film.
         - title: Judul film, termasuk tahun rilisnya.
         - genres: Daftar genre yang dipisahkan oleh pipa (|) yang terkait dengan film tersebut.
     - ratings.csv: Dataset ini berisi rating pengguna untuk film.
       - Jumlah Baris (sebelum pra-pemrosesan): 100836 baris.
       - Jumlah Kolom (sebelum pra-pemrosesan): 4 kolom.
       Penjelasan Kolom:
         - userId: Pengidentifikasi unik untuk setiap pengguna.
         - movieId: ID film yang diberi rating, terhubung ke dataset movies.csv.
         - rating: Rating yang diberikan oleh pengguna ke film, pada skala umumnya dari 0.5 hingga 5.0.
         - timestamp: Waktu ketika rating diberikan, direpresentasikan sebagai Unix timestamp.
  
## Data Preparation

A. Penjelasan Tahapan
1. Penghapusan Duplikat:
  - movies.drop_duplicates(subset=['movieId', 'title'], inplace=True): Baris duplikat berdasarkan kombinasi movieId dan title dihapus dari DataFrame movies.
  - rating.drop_duplicates(inplace=True): Baris duplikat dihapus dari DataFrame rating.
Tujuan: Memastikan kebersihan data dan menghindari perhitungan yang tidak akurat karena adanya entri ganda.

2. Pemrosesan Kolom 'genres':
  - movies['genres'] = movies['genres'].apply(lambda x: x.split('|')): String genre dipecah menjadi list genre untuk setiap film.
  - mlb = MultiLabelBinarizer() dan genre_matrix = pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_): MultiLabelBinarizer digunakan untuk melakukan one-hot encoding pada kolom genre, mengubah list genre menjadi kolom biner (0 atau 1) untuk setiap kategori genre yang ada.
  - movies = pd.concat([movies[['movieId', 'title']], genre_matrix], axis=1): Matriks genre yang sudah di-one-hot encode digabungkan kembali dengan kolom movieId dan title dari DataFrame movies.
Tujuan: Mengubah format genre dari string menjadi representasi numerik biner yang dapat digunakan untuk perhitungan kesamaan pada model Content-Based Filtering.

3. Konversi dan Ekstraksi Timestamp:
  - rating['datetime'] = pd.to_datetime(rating['timestamp'], unit='s'): Kolom timestamp di DataFrame rating dikonversi menjadi format datetime yang lebih mudah dioperasikan.
  - rating['rating_year'] = rating['datetime'].dt.year: Kolom baru rating_year ditambahkan, berisi tahun dari kolom datetime.
  - Tujuan: Memudahkan analisis berbasis waktu dan potensi penggunaan fitur waktu di kemudian hari, meskipun rating_year tidak secara langsung digunakan dalam model rekomendasi yang Anda tampilkan.Penggabungan Dataset movies dan ratings:
  - movie_rating = pd.merge(rating, movies, on='movieId', how='inner'): DataFrame rating dan movies digabungkan berdasarkan movieId.
Tujuan: Mengkonsolidasi informasi film dan rating ke dalam satu DataFrame untuk memudahkan analisis dan pembangunan model.

B. Tahapan Data Preparation Terpisah untuk Setiap Pendekatan

1. Data Preparation Umum (untuk kedua pendekatan):
  - Penghapusan duplikat pada movies dan rating.
  - Konversi timestamp ke datetime dan ekstraksi rating_year.
  - Penggabungan rating dan movies menjadi movie_rating.

2. Data Preparation Spesifik untuk Content-Based Filtering:
  - Pemrosesan kolom genres dengan one-hot encoding menggunakan MultiLabelBinarizer.
  - Pembuatan movie_features dan genre_matrix yang hanya berfokus pada atribut film (genre) untuk perhitungan cosine_similarity.
  - Pembuatan user_likes (film yang disukai pengguna) untuk evaluasi Precision@K.

3. Data Preparation Spesifik untuk Collaborative Filtering:
  - Membuat DataFrame df baru dari rating yang hanya berisi userId, movieId, dan rating.
  - Mapping user dan movie ke angka: user_to_encoded, movie_to_encoded, dan pembuatan kolom user serta movie terenkripsi.
  - Normalisasi rating (norm_rating).
  - Acak dan split data: df.sample(frac=1, random_state=42) dan train_test_split(x, y, test_size=0.2, random_state=42) untuk memisahkan data menjadi set pelatihan dan validasi.

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

**Contoh Output Rekomendasi untuk User 301:**
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
