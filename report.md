# Laporan Proyek: Prediksi Kasus Demam Berdarah Dengue Menggunakan Artificial Neural Network

## 1. Penjelasan Terhadap Masalah yang Dijadikan Kasus

Demam Berdarah Dengue (DBD) merupakan penyakit infeksi yang disebabkan oleh virus dengue dan ditularkan melalui gigitan nyamuk *Aedes aegypti* dan *Aedes albopictus*. Penyakit ini menjadi salah satu masalah kesehatan masyarakat yang paling serius di wilayah tropis dan subtropis di seluruh dunia. Menurut data World Health Organization (WHO), sekitar 390 juta infeksi dengue terjadi setiap tahunnya, dengan 96 juta di antaranya menunjukkan manifestasi klinis yang signifikan. Wilayah Asia Tenggara, Amerika Latin, dan Kepulauan Pasifik merupakan daerah endemis dengan tingkat kejadian tertinggi.

Dampak kesehatan masyarakat dari demam berdarah di wilayah tropis sangat besar. Selain menyebabkan morbiditas dan mortalitas yang tinggi, epidemi dengue juga membebani sistem kesehatan, menurunkan produktivitas ekonomi, dan memerlukan alokasi sumber daya yang signifikan untuk pengendalian vektor. Di Indonesia sendiri, kasus demam berdarah dilaporkan secara konsisten setiap tahun dengan pola musiman yang berkaitan erat dengan musim hujan dan kondisi iklim lokal.

Terdapat beberapa variabel lingkungan dan iklim yang diketahui mempengaruhi penyebaran dengue. Faktor-faktor utama meliputi: (1) suhu udara rata-rata, yang mempengaruhi siklus hidup nyamuk dan periode inkubasi virus; (2) kelembaban relatif, yang berkorelasi dengan kelangsungan hidup nyamuk dewasa; (3) curah hujan, yang menciptakan habitat perkembangbiakan larva nyamuk; (4) Normalized Difference Vegetation Index (NDVI), yang merepresentasikan tutupan vegetasi sebagai indikator ketersediaan habitat nyamuk; serta (5) suhu titik embun dan variabel reanalisis atmosfer lainnya yang secara tidak langsung mempengaruhi dinamika populasi vektor.

Dalam konteks ini, penggunaan Artificial Neural Network (ANN) sebagai metode prediksi dipilih karena beberapa alasan fundamental. Pertama, hubungan antara variabel iklim dan jumlah kasus dengue bersifat non-linear dan kompleks, sehingga metode regresi linear konvensional tidak mampu menangkap pola-pola tersebut secara memadai. Kedua, ANN memiliki kemampuan untuk melakukan aproksimasi fungsi universal (universal function approximation), yang berarti model ini secara teoritis dapat memodelkan hubungan kompleks apa pun antara input dan output dengan arsitektur yang tepat. Ketiga, ANN dapat menangani interaksi antar fitur tanpa perlu spesifikasi eksplisit dari interaksi tersebut oleh peneliti, sehingga model dapat menemukan pola-pola tersembunyi yang mungkin tidak terdeteksi oleh analisis statistik tradisional.

Dataset yang digunakan dalam proyek ini berasal dari kompetisi DengAI yang diselenggarakan oleh DrivenData, mencakup data mingguan untuk dua kota: San Juan, Puerto Rico (sj) dan Iquitos, Peru (iq). Dataset ini menyediakan 24 fitur cuaca dan lingkungan per minggu beserta jumlah kasus dengue aktual sebagai variabel target, menjadikannya kasus yang ideal untuk pendekatan pembelajaran mesin berbasis regresi.

## 2. Konsep Teori dari Metode yang Digunakan

### Model Perceptron dan Arsitektur Jaringan Saraf

Perceptron merupakan unit dasar (neuron) dalam jaringan saraf tiruan. Setiap perceptron menerima satu atau lebih input, mengalikan masing-masing input dengan bobot (weight) yang bersesuaian, menjumlahkan hasil perkalian tersebut dengan bias, kemudian melewatkan hasilnya melalui fungsi aktivasi untuk menghasilkan output. Secara matematis, output sebuah neuron dinyatakan sebagai: z = f(Σ(wi * xi) + b), di mana wi adalah bobot, xi adalah input, b adalah bias, dan f adalah fungsi aktivasi.

Dalam proyek ini, jaringan saraf yang digunakan adalah feedforward neural network dengan arsitektur Sequential, artinya informasi mengalir hanya dalam satu arah dari input menuju output tanpa adanya loop atau koneksi balik.

### Fungsi Aktivasi

**ReLU (Rectified Linear Unit)** digunakan pada seluruh hidden layer. Fungsi ini didefinisikan sebagai f(x) = max(0, x). ReLU dipilih karena kemampuannya mengatasi masalah vanishing gradient yang sering terjadi pada fungsi aktivasi sigmoid dan tanh, serta efisiensi komputasinya yang tinggi. ReLU memungkinkan jaringan untuk belajar representasi non-linear yang kompleks sambil mempertahankan gradien yang stabil selama proses pelatihan.

**Aktivasi Linear** digunakan pada output layer. Karena tugas yang diselesaikan adalah regresi (prediksi jumlah kasus yang berupa bilangan kontinu), output layer menggunakan aktivasi linear f(x) = x tanpa transformasi, sehingga model dapat menghasilkan prediksi dalam rentang nilai yang tidak terbatas.

### Forward Propagation (Propagasi Maju)

Forward propagation adalah proses di mana data input dilewatkan melalui seluruh lapisan jaringan secara berurutan untuk menghasilkan prediksi. Pada setiap lapisan, input dikalikan dengan matriks bobot, ditambah bias, kemudian ditransformasi oleh fungsi aktivasi. Proses ini berlanjut dari input layer, melewati seluruh hidden layer, hingga mencapai output layer yang menghasilkan prediksi akhir. Dalam arsitektur yang digunakan, input melewati tiga hidden layer (128, 64, dan 32 neuron) sebelum menghasilkan satu nilai prediksi.

### Backpropagation (Propagasi Mundur)

Backpropagation adalah algoritma yang digunakan untuk menghitung gradien fungsi loss terhadap setiap bobot dalam jaringan. Proses ini menggunakan aturan rantai (chain rule) dari kalkulus diferensial untuk menghitung kontribusi setiap bobot terhadap kesalahan prediksi secara efisien. Gradien yang dihitung kemudian digunakan oleh optimizer untuk memperbarui bobot-bobot jaringan agar meminimalkan fungsi loss. Backpropagation memungkinkan pelatihan jaringan yang dalam (deep) secara efisien karena menghindari kebutuhan untuk menghitung gradien secara independen untuk setiap bobot.

### Fungsi Loss: Mean Squared Error (MSE)

Fungsi loss yang digunakan untuk pelatihan adalah Mean Squared Error (MSE), yang menghitung rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. MSE dipilih karena memberikan penalti yang lebih besar untuk kesalahan prediksi yang besar (outlier), mendorong model untuk menghindari prediksi yang sangat melenceng. Evaluasi model dilakukan menggunakan Mean Absolute Error (MAE) yang lebih mudah diinterpretasikan karena memiliki unit yang sama dengan variabel target (jumlah kasus).

### Optimizer Adam

Adam (Adaptive Moment Estimation) adalah optimizer berbasis gradien yang menggabungkan keunggulan dari dua metode optimasi populer: AdaGrad (yang bekerja baik pada gradien jarang) dan RMSProp (yang bekerja baik pada pengaturan non-stasioner). Adam mempertahankan rata-rata bergerak eksponensial dari momen pertama (mean) dan momen kedua (variance tidak terpusat) dari gradien, kemudian menggunakan estimasi yang dikoreksi bias untuk memperbarui bobot. Keunggulan Adam meliputi konvergensi yang cepat, ketahanan terhadap pemilihan learning rate, dan kesesuaian untuk masalah dengan data berskala besar atau parameter yang banyak.

### Dropout Regularization

Dropout adalah teknik regularisasi yang mencegah overfitting dengan secara acak menonaktifkan (mengatur output ke nol) sebagian neuron selama pelatihan. Dalam arsitektur ini, dropout dengan rate 0.3 diterapkan setelah dua hidden layer pertama, yang berarti 30% neuron pada lapisan tersebut dinonaktifkan secara acak pada setiap batch pelatihan. Teknik ini memaksa jaringan untuk belajar representasi yang lebih robust dan tidak bergantung pada neuron tertentu, sehingga meningkatkan kemampuan generalisasi model pada data yang belum pernah dilihat sebelumnya.

### Early Stopping

Early stopping adalah teknik regularisasi berbasis waktu yang menghentikan proses pelatihan ketika performa model pada data validasi tidak lagi membaik. Dalam implementasi ini, pelatihan dihentikan jika validation loss tidak mengalami penurunan selama 10 epoch berturut-turut (patience=10). Selain itu, bobot model dikembalikan ke kondisi terbaik yang tercatat selama pelatihan (restore_best_weights=True). Early stopping mencegah overfitting dengan menghindari pelatihan yang berlebihan di mana model mulai menghafal noise pada data pelatihan.

### Deskripsi Arsitektur ANN yang Digunakan

Arsitektur ANN yang diimplementasikan dalam proyek ini terdiri dari:

1. **Input Layer**: Menerima vektor fitur berdimensi sesuai jumlah fitur setelah preprocessing (23 fitur berupa city terenkode, year, weekofyear, dan 20 fitur cuaca/lingkungan).
2. **Hidden Layer 1**: Dense layer dengan 128 neuron dan aktivasi ReLU, diikuti oleh Dropout layer dengan rate 0.3.
3. **Hidden Layer 2**: Dense layer dengan 64 neuron dan aktivasi ReLU, diikuti oleh Dropout layer dengan rate 0.3.
4. **Hidden Layer 3**: Dense layer dengan 32 neuron dan aktivasi ReLU.
5. **Output Layer**: Dense layer dengan 1 neuron dan aktivasi linear, menghasilkan prediksi jumlah kasus dengue.

Total parameter model bervariasi tergantung jumlah fitur input, namun dengan 23 fitur input, arsitektur ini menghasilkan sekitar 12.000-13.000 parameter yang dapat dilatih. Model dikompilasi dengan optimizer Adam dan fungsi loss MSE, serta dilatih dengan maksimum 200 epoch dan batch size 32.

## 3. Tahapan Pemrosesan Data hingga Output

### Tahap 1: Environment Setup dan Reproducibility Seeds (Cell 1-2)

Tahap pertama meliputi impor seluruh library yang dibutuhkan (pandas, numpy, tensorflow, scikit-learn, seaborn, matplotlib) dan pengaturan random seed untuk memastikan reprodusibilitas. Seed yang ditetapkan meliputi random.seed(42), np.random.seed(42), dan tf.random.set_seed(42). Cell kedua mendokumentasikan versi Python dan seluruh library yang digunakan untuk keperluan audit reprodusibilitas.

### Tahap 2: Data Loading dan Merging (Cell 3)

Pada tahap ini, empat file CSV dari dataset DengAI dimuat menggunakan pd.read_csv(): dengue_features_train.csv, dengue_labels_train.csv, dengue_features_test.csv, dan submission_format.csv dari direktori data/. Training features dan labels digabungkan menggunakan left join pada kolom kunci (city, year, weekofyear) untuk menghasilkan df_train_merged. Output shape: df_train_merged (1456, 25), df_test (416, 24), df_submission (416, 4).

### Tahap 3: Missing Value Handling (Cell 4)

Penanganan nilai hilang dilakukan dengan strategi multi-tahap: (1) forward-fill kemudian backward-fill untuk mengisi gap interior pada kolom numerik; (2) mean imputation jika masih terdapat 5 atau lebih nilai NaN dalam suatu kolom; (3) pengisian dengan nol jika seluruh nilai dalam kolom masih NaN, disertai peringatan. Setelah imputasi, seluruh kolom numerik memiliki nol nilai null, memastikan kelengkapan data untuk input model.

### Tahap 4: Exploratory Data Analysis (Cell 5)

EDA menghasilkan tiga visualisasi utama: (1) correlation heatmap seluruh fitur numerik yang menampilkan korelasi antar variabel cuaca dan kasus dengue; (2) time-series plot untuk kota San Juan (sj) yang menampilkan tren total_cases, station_avg_temp_c, dan reanalysis_relative_humidity_percent sepanjang waktu; (3) time-series plot serupa untuk kota Iquitos (iq). Visualisasi ini mengungkap pola musiman yang kuat dan hubungan positif antara kelembaban/suhu dengan jumlah kasus.

### Tahap 5: Preprocessing dan Feature Engineering (Cell 6)

Preprocessing mencakup langkah-langkah: (1) encoding kolom city menjadi numerik biner (sj=0, iq=1); (2) penghapusan kolom week_start_date yang bersifat non-prediktif; (3) pemisahan fitur (X) dan target (y=total_cases); (4) pembagian data menjadi training set (80%) dan validation set (20%) menggunakan train_test_split dengan random_state=42; (5) normalisasi fitur menggunakan MinMaxScaler yang difit pada training split lalu ditransformasi ke kedua split. Output: X_train dan X_val berupa array dengan dimensi (n_samples, 23) dan seluruh nilai dinormalisasi ke rentang [0, 1].

### Tahap 6: ANN Model Architecture (Cell 7)

Definisi arsitektur model Sequential dengan tiga hidden layer (128-64-32 neuron, ReLU), dua Dropout layer (rate 0.3), dan satu output layer (1 neuron, linear). Model dikompilasi dengan Adam optimizer dan MSE loss. Random seed diatur ulang sebelum inisialisasi model untuk memastikan bobot awal yang identik pada setiap eksekusi. Model summary menampilkan konfigurasi layer, output shape, dan jumlah parameter.

### Tahap 7: Model Training (Cell 8)

Pelatihan dilakukan dengan EarlyStopping callback (monitor val_loss, patience=10, restore_best_weights=True), maksimum 200 epoch, batch size 32, dan validation_split=0.2 dalam Keras fit(). Grafik loss curve menampilkan penurunan training loss dan validation loss per epoch, dengan indikasi titik di mana early stopping terpicu. Output mencakup informasi epoch terakhir dan visualisasi kurva pelatihan.

### Tahap 8: Model Evaluation (Cell 9)

Evaluasi dilakukan pada validation set dengan menghitung MAE dan RMSE menggunakan sklearn.metrics. Scatter plot actual vs. predicted menampilkan distribusi prediksi terhadap nilai aktual. Evaluasi per kota (sj dan iq) memberikan pemahaman tentang performa model pada masing-masing konteks geografis. Metrik kombinasi kedua kota juga dilaporkan.

### Tahap 9: Inference (Cell 10-11)

Fungsi predict_dengue_cases menerima input berupa dictionary atau Series fitur cuaca, melakukan validasi kelengkapan input, menerapkan scaler yang sama, dan mengembalikan prediksi berupa bilangan bulat non-negatif. Fungsi generate_submission mengiterasi seluruh baris test set, menghasilkan prediksi per baris, dan menulis hasil ke submission.csv dengan format yang sesuai (kolom: city, year, weekofyear, total_cases).

### Tahap 10: Report Generation (Cell 12)

Tahap akhir menghasilkan laporan akademik report.md dalam Bahasa Indonesia yang mencakup empat seksi wajib: latar belakang masalah, konsep teori, tahapan pemrosesan, dan hasil analisis. Nilai MAE dan RMSE aktual dari evaluasi diinterpolasi langsung ke dalam seksi hasil.

## 4. Hasil dan Deskripsi/Bahasannya

### Interpretasi Nilai MAE dan RMSE

Model ANN yang dilatih menghasilkan Mean Absolute Error (MAE) sebesar 22.9294 dan Root Mean Square Error (RMSE) sebesar 51.2738 pada validation set. Nilai MAE sebesar 22.9294 menunjukkan bahwa secara rata-rata, prediksi model menyimpang sebanyak 22.9294 kasus dari jumlah kasus aktual per minggu. Sementara itu, RMSE sebesar 51.2738 yang lebih besar dari MAE mengindikasikan adanya beberapa prediksi dengan kesalahan yang lebih besar (outlier), karena RMSE memberikan penalti kuadrat untuk kesalahan yang besar.

Rasio RMSE/MAE dapat digunakan sebagai indikator variabilitas kesalahan prediksi. Rasio yang mendekati 1 menunjukkan kesalahan yang konsisten, sedangkan rasio yang jauh lebih besar dari 1 menunjukkan adanya variasi besar dalam akurasi prediksi. Dalam konteks epidemiologi, MAE memberikan gambaran yang lebih intuitif tentang akurasi prediksi mingguan yang dapat diharapkan dari model ini.

### Analisis Kurva Loss (Training vs. Validation)

Kurva training loss dan validation loss menunjukkan pola pembelajaran model selama proses pelatihan. Pada epoch-epoch awal, kedua kurva menurun secara signifikan yang menandakan model berhasil mempelajari pola-pola fundamental dalam data. Seiring bertambahnya epoch, penurunan loss melambat dan kurva mulai mendatar (plateau), mengindikasikan konvergensi model.

Jarak (gap) antara training loss dan validation loss merupakan indikator tingkat overfitting. Gap yang kecil menunjukkan bahwa model memiliki kemampuan generalisasi yang baik, sedangkan gap yang besar mengindikasikan overfitting di mana model terlalu menghafal data pelatihan. Penggunaan Dropout dan Early Stopping membantu menjaga gap ini tetap minimal. Early stopping terpicu sebelum epoch ke-200, menghentikan pelatihan pada titik optimal di mana validation loss mulai stagnasi atau meningkat.

### Analisis Plot Predicted vs. Actual

Scatter plot actual vs. predicted cases menampilkan sebaran prediksi model terhadap nilai sebenarnya. Titik-titik yang berada di dekat garis diagonal ideal (y=x) menunjukkan prediksi yang akurat. Dari visualisasi ini dapat diamati bahwa model cenderung lebih akurat untuk prediksi kasus rendah hingga sedang, namun mungkin mengalami underestimation pada kasus wabah dengan jumlah kasus yang sangat tinggi (outlier). Hal ini konsisten dengan karakteristik MSE loss yang mendominasi pelatihan dan distribusi target yang right-skewed (sebagian besar minggu memiliki kasus rendah dengan beberapa spike besar).

### Perbandingan Antar Kota

Performa model pada kedua kota menunjukkan perbedaan yang signifikan. San Juan (sj) yang memiliki jumlah kasus rata-rata lebih tinggi dan variasi musiman yang lebih ekstrem cenderung menghasilkan MAE dan RMSE yang lebih besar secara absolut dibandingkan Iquitos (iq). Hal ini disebabkan oleh: (1) rentang nilai total_cases yang lebih lebar di San Juan; (2) pola wabah periodik yang lebih intens; dan (3) proporsi data San Juan yang lebih dominan dalam training set. Evaluasi per kota penting untuk memahami konteks di mana model bekerja lebih baik dan di mana diperlukan perbaikan.

### Keterbatasan Model

Beberapa keterbatasan utama dari model yang diimplementasikan meliputi:

1. **Temporal dependency diabaikan**: Model feedforward yang digunakan tidak menangkap dependensi temporal (urutan waktu) antar observasi. Jumlah kasus dengue pada minggu tertentu kemungkinan berkorelasi kuat dengan kasus pada minggu-minggu sebelumnya, namun informasi ini tidak dieksploitasi oleh arsitektur saat ini.

2. **Tidak ada feature engineering lanjutan**: Model menggunakan fitur mentah tanpa rekayasa fitur seperti lag features, rolling averages, atau fitur interaksi yang dapat meningkatkan representasi pola temporal.

3. **Generalisasi terbatas pada dua kota**: Model hanya dilatih dan divalidasi pada data dari San Juan dan Iquitos, sehingga generalisasi ke kota lain tidak dapat dijamin tanpa pelatihan ulang.

4. **Sensitivitas terhadap outlier**: Kejadian wabah dengan jumlah kasus ekstrem sulit diprediksi karena jarang terjadi dalam data pelatihan, menyebabkan model cenderung underestimate pada saat terjadi lonjakan kasus.

5. **Pendekatan single-model**: Model yang sama diterapkan untuk kedua kota, padahal dinamika epidemiologi masing-masing kota mungkin berbeda dan memerlukan pendekatan yang terspesialisasi.

### Saran Perbaikan

Untuk meningkatkan performa model di masa depan, beberapa pendekatan dapat dipertimbangkan:

1. **Arsitektur berbasis sekuens**: Menggunakan LSTM (Long Short-Term Memory) atau GRU (Gated Recurrent Unit) yang dirancang khusus untuk menangkap dependensi temporal dalam data time-series.

2. **Feature engineering temporal**: Menambahkan fitur lag (nilai kasus dan cuaca pada minggu-minggu sebelumnya), rolling statistics (rata-rata bergerak, standar deviasi), dan fitur musiman (sine/cosine encoding untuk week_of_year).

3. **Model terpisah per kota**: Melatih model independen untuk masing-masing kota agar dapat menangkap pola epidemiologi lokal yang unik.

4. **Ensemble methods**: Menggabungkan prediksi dari beberapa model (ANN, Random Forest, XGBoost) untuk meningkatkan robustness dan akurasi.

5. **Augmentasi data dan oversampling**: Menerapkan teknik untuk menangani ketidakseimbangan distribusi target, terutama untuk kasus wabah yang jarang terjadi.

6. **Hyperparameter tuning**: Melakukan pencarian grid atau Bayesian optimization untuk menemukan kombinasi optimal dari jumlah layer, neuron, learning rate, dropout rate, dan batch size.
