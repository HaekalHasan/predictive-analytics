# Laporan Proyek Machine Learning - Haekal Hasan Thanvindra
---
## 1. Domain Proyek

### Latar Belakang

Industri properti merupakan salah satu sektor yang sangat dinamis dan dipengaruhi oleh banyak faktor seperti lokasi, ukuran bangunan, fasilitas, hingga kondisi lingkungan. Dataset yang digunakan dalam proyek ini mencakup fitur-fitur seperti luas rumah, jumlah kamar tidur dan kamar mandi, ukuran garasi, serta kualitas lingkungan, yang kesemuanya memengaruhi harga rumah secara signifikan.

Bagi agen properti, investor, dan calon pembeli, kemampuan untuk memprediksi harga rumah secara akurat sangat penting untuk pengambilan keputusan. Dengan semakin tersedianya data historis penjualan rumah dan informasi karakteristik properti yang detail, teknologi machine learning memberikan peluang untuk membangun model prediktif yang mampu memproyeksikan harga rumah berdasarkan fitur-fitur tersebut.

### Kenapa Masalah Ini Harus Diselesaikan

Menentukan harga rumah yang sesuai dengan nilai pasar sangat krusial, baik untuk menghindari kerugian karena *undervaluation* maupun untuk menjaga daya saing jika terjadi *overpricing*. Prediksi harga rumah dapat membantu:

* Calon pembeli untuk mengetahui harga pasar yang wajar
* Agen properti untuk menentukan strategi harga
* Investor untuk mengidentifikasi properti yang menguntungkan

### Referensi Pendukung

* C. R. Madhuri, G. Anuradha and M. V. Pujitha, "House Price Prediction Using Regression Techniques: A Comparative Study," 2019 International Conference on Smart Structures and Systems (ICSSS), Chennai, India, 2019, pp. 1-5, doi: 10.1109/ICSSS.2019.8882834.
  [Link ke Paper](https://ieeexplore.ieee.org/abstract/document/8882834)

---

## 2. Business Understanding

### Problem Statements

* Bagaimana memanfaatkan data historis penjualan rumah untuk memprediksi harga rumah di masa mendatang?
* Apa saja fitur (variabel) yang paling berpengaruh terhadap harga rumah?

### Goals

* Membangun model Machine Learning untuk memprediksi harga rumah berdasarkan fitur-fitur yang tersedia.
* Mengidentifikasi fitur paling penting yang memengaruhi harga rumah.

### Solution Statements

* Membangun dua model regresi menggunakan **Linear Regression** dan **Random Forest Regressor**, lalu membandingkan performa keduanya.
* Evaluasi performa model dilakukan menggunakan metrik **Root Mean Squared Error (RMSE)** dan **R-squared (R²)**.

---

## 3. Data Understanding

Dataset yang digunakan dalam proyek ini adalah **House Price Regression Dataset** dari Kaggle. Dataset ini terdiri dari 1000 baris data dan 8 fitur numerik beserta target prediksi (`House_Price`).

🔗 [Link ke Dataset](https://www.kaggle.com/datasets/prokshitha/home-value-insights)

### Variabel dalam Dataset

| Fitur                  | Deskripsi                        |
| ---------------------- | -------------------------------- |
| `Square_Footage`       | Ukuran rumah (dalam sq.ft).      |
| `Num_Bedrooms`         | Jumlah kamar tidur.              |
| `Num_Bathrooms`        | Jumlah kamar mandi.              |
| `Year_Built`           | Tahun rumah dibangun.            |
| `Lot_Size`             | Luas tanah (dalam acre).         |
| `Garage_Size`          | Kapasitas garasi.                |
| `Neighborhood_Quality` | Skor kualitas lingkungan (1–10). |
| `House_Price`          | Harga rumah (target).            |

### Insight

* Rata-rata luas rumah: 2815 sq.ft (min: 503, max: 4999)
* Rata-rata kamar tidur: \~3
* Harga rumah rata-rata: Rp 618,861
* Tidak ada **missing values**

### Visualisasi & Insight Tambahan

* **Distribusi Harga Rumah** normal dengan outlier di harga tinggi.
* **Square\_Footage** memiliki korelasi sangat kuat terhadap harga (0.99)
* **Lot\_Size** memiliki korelasi lebih lemah namun tetap positif.
* Visualisasi scatter plot menunjukkan bahwa semakin besar luas rumah, semakin tinggi harganya.

---

## 4. Data Preparation

### Langkah-langkah:

1. **Membagi dataset** menjadi training (80%) dan test (20%).
2. **Standardisasi** fitur numerik menggunakan `StandardScaler`.

   * Tujuannya agar skala fitur seragam dan model tidak bias.
3. **Tidak melakukan encoding**, karena semua fitur sudah numerik.
4. **Tidak ada missing values**, sehingga tidak perlu cleaning tambahan.

---

## 5. Modeling

Pada tahap ini, dilakukan pembuatan dan pelatihan model machine learning untuk memprediksi harga rumah. Dua algoritma digunakan:

### Model yang Digunakan:

1. **Linear Regression**
   Linear Regression adalah model linier sederhana yang cepat dan mudah diinterpretasikan. Model ini bekerja dengan mencari garis terbaik yang meminimalkan selisih kuadrat antara nilai prediksi dan aktual (*least squares method*). Model ini cocok digunakan sebagai baseline.

   **Parameter:**

   * Tidak ada parameter yang disetel secara eksplisit.
   * Parameter default yang digunakan:

     * `fit_intercept=True`: Model menghitung intercept secara otomatis.
     * `n_jobs=None`: Proses training dilakukan pada satu core CPU.

2. **Random Forest Regressor**
   Random Forest merupakan algoritma ensambel berbasis banyak pohon keputusan (decision tree). Model ini mampu menangkap hubungan non-linier dan interaksi antar fitur dengan baik.

   **Parameter:**

   * `random_state=42`: Ditentukan untuk memastikan hasil yang konsisten dan reproducible.
   * Parameter lain menggunakan nilai default:

     * `n_estimators=100`: Jumlah pohon dalam forest.
     * `max_depth=None`: Tidak ada batas kedalaman pohon.

Model dilatih menggunakan data training yang telah distandardisasi (scaling), lalu diuji menggunakan data test.

---

## 6. Evaluation

### Metrik Evaluasi:

* **Root Mean Squared Error (RMSE)**
  Mengukur rata-rata kesalahan prediksi dalam satuan harga rumah. Nilai lebih kecil menandakan performa yang lebih baik.
* **R-squared (R² Score)**
  Mengukur seberapa besar variansi harga rumah yang dapat dijelaskan oleh fitur-fitur input. Nilai mendekati 1 menunjukkan model yang sangat baik.

### Hasil Evaluasi:

```
Linear Regression  →  RMSE: 10071.48  |  R²: 1.00  
Random Forest      →  RMSE: 19853.32  |  R²: 0.99  
```

### Interpretasi Hasil:

Model **Linear Regression** menunjukkan performa yang sangat baik dengan RMSE terendah dan R² sempurna (1.00), yang berarti model mampu menjelaskan hampir seluruh variasi harga rumah. **Random Forest Regressor** juga menunjukkan performa sangat baik, namun sedikit lebih rendah dibandingkan Linear Regression.

### Evaluasi terhadap Business Understanding:

* **Problem Statement #1**
  *“Bagaimana memanfaatkan data historis penjualan rumah untuk memprediksi harga rumah di masa mendatang?”*
  ✅ *Telah terjawab.*
  Model mampu memprediksi harga rumah secara akurat berdasarkan data historis dan fitur yang tersedia.

* **Problem Statement #2**
  *“Apa saja fitur (variabel) yang paling berpengaruh terhadap harga rumah?”*
  ✅ *Terjawab sebagian.*
  Analisis korelasi menunjukkan bahwa *Square\_Footage* merupakan fitur penting. Random Forest dapat dieksplorasi lebih lanjut untuk *feature importance*.

* **Goals**
  ✅ *Tercapai.*
  Model prediksi berhasil dibangun dengan performa evaluasi yang sangat baik. Linear Regression terbukti sangat efektif untuk memproyeksikan harga rumah.

* **Solusi yang Diterapkan**
  ✅ *Berdampak positif.*
  Penggunaan dua model dan evaluasinya menunjukkan bahwa pendekatan ini berjalan sesuai rencana. Linear Regression memberikan hasil terbaik untuk kasus ini.

### Kesimpulan:

Model **Linear Regression** dipilih sebagai model terbaik karena performanya yang unggul dan efisiensi pelatihan. Model ini cocok digunakan dalam skenario nyata, seperti membantu agen properti atau pembeli dalam menentukan harga rumah secara objektif.

---