---
title: Word Embedding

---

### Nama : Isnita Widyur Rahmah
### NIM : 220411100048
### Kelas : IF 7A


# Word Embedding
Word embedding adalah teknik yang mengubah kata-kata menjadi representasi vektor numerik, sehingga kata-kata yang memiliki makna atau konteks serupa akan memiliki vektor yang dekat satu sama lain. Tujuan utamanya adalah untuk mengubah kata-kata (yang berupa teks) menjadi angka sehingga dapat diproses oleh model pembelajaran mesin (machine learning) atau pembelajaran mendalam (deep learning).

## Skip-Gram
Skip-gram adalah metode yang memprediksi kata-kata konteks di sekitar kata target untuk menghasilkan representasi vektor kata dalam model Word2Vec. Kata-kata konteks ini bisa terletak di sebelah kiri atau kanan dari kata target, tergantung pada ukuran jendela (window size) yang telah ditentukan. Misalnya, jika memilih kata target w(t), model akan berusaha memprediksi kata-kata konteks di sekitar kata tersebut, seperti w(t−2), w(t−1), w(t+1), w(t+2).
![1](https://hackmd.io/_uploads/HJ88W_rRA.png)

### 1. Representasi Vektor Kata
![2](https://hackmd.io/_uploads/BkZabuBA0.png)
Korelasi antara dua kata, seperti "success" dan "achieve", bisa dihitung menggunakan jarak vektor di antara keduanya. Dalam visualisasi tiga dimensi, jarak ini menunjukkan seberapa dekat atau jauh makna kedua kata. Biasanya, jarak Cosine digunakan untuk mengukur hubungan antara vektor, meskipun dalam contoh visualisasi ini jarak Euclidean yang digunakan.
#### Contoh Analogi dengan Vektor Kata:
Misalnya, untuk mengetahui ibu kota Jerman, dapat dilakukan dengan menggunakan operasi vektor seperti berikut
\begin{align*} 
vec(\text{Germany}) & = [1.22 \quad 0.34 \quad -3.82] \\ 
vec(\text{capital}) & = [3.02 \quad -0.93 \quad 1.82] \\
vec(\text{Berlin})  & = [4.09 \quad -0.58 \quad 2.01]
\end{align*}Dengan menambahkan vektor kata Germany dan capital
\begin{align*}
    vec(\text{Germany}) + vec(\text{capital}) &= [1.22 \quad 0.34 \quad -3.82] + [3.02 \quad -0.93 \quad 1.82] \\
                                              &= [4.24 \quad -0.59 \quad -2.00]
\end{align*}Hasil ini hampir sama dengan vektor kata Berlin, sehingga model dapat menyimpulkan bahwa ibu kota Jerman adalah Berlin.

### 2. Konsep Window Size dalam model Skip-Gram
- Softmax Regression
    Softmax regression (atau multinomial logistic regression) adalah generalisasi dari logistic regression yang digunakan ketika ada lebih dari dua kelas yang ingin diklasifikasikan.
    Rumus umum untuk softmax regression adalah:
    $$J(\theta) = -\frac{1}{T} \sum^T_{t=1}\sum^K_{k=1}log \frac{exp(\theta^{(k)\top}x^{(t)})}{\sum^K_{i=1}exp(\theta^{(i)\top}x^{(t)})} \tag{10}$$T adalah jumlah sampel pelatihan.
K adalah jumlah label yang harus diklasifikasikan.
Dalam aplikasi Natural Language Processing (NLP),  𝐾 diatur sama dengan 𝑉, yaitu jumlah kata dalam kosakata. Kosakata ini bisa sangat besar, bahkan mencapai puluhan ribu kata.
- Penerapan dalam Skip-Gram
Model Skip-Gram mengadaptasi rumus softmax dengan mengganti 𝐾 dengan window size 𝐶. Window size adalah parameter hiper yang biasanya berkisar antara 1 hingga 10. Ini merepresentasikan jumlah kata konteks yang akan diprediksi di sekitar kata target. Skip-Gram tidak perlu memprediksi semua 𝑉 kata dalam korpus yang mungkin berjarak ratusan kata dari kata target. Sebaliknya, ia hanya fokus pada memprediksi 1 hingga 10 kata konteks terdekat.
- Rumus yang diadaptasi untuk Skip-Gram
$$J(\theta) = -\frac{1}{T} \sum^T_{t=1}\sum_{c\leq j \leq c,j\neq 0}log\frac{exp(\theta^{(t+j)\top}x^{(t)})}{\sum^K_{i=1}exp(\theta^{(i)\top}x^{(t)})} \tag{11}$$𝑗 menunjukkan posisi relatif kata konteks terhadap kata target. Nilai  𝑗 dapat positif atau negatif untuk menandakan posisi kiri atau kanan.
𝐾 dalam penyebut merujuk pada jumlah total kata dalam kosakata 𝑉. Ini memastikan probabilitas kata konteks valid dan terdistribusi dengan baik.

### 3. Struktur Skip-Gram
Model Skip-Gram menggunakan jaringan saraf untuk mempelajari representasi vektor kata sehingga dapat memprediksi kata-kata konteks di sekitar kata target. Model ini berupaya memahami hubungan antara kata target dan kata-kata konteksnya untuk mengoptimalkan pembelajaran vektor kata.
![3](https://hackmd.io/_uploads/HymY8dH00.png)
Dalam contoh ini, terdapat 10 kata (T=10) dan 8 kata unik (V=8). Di asumsikan bahwa kata "passes" adalah kata target, dan kata "who" serta "the" adalah kata konteks. Window size pada contoh tersebut ditetapkan sebesar 1. Ini berarti hanya satu kata di kiri dan satu kata di kanan dari kata target yang dipertimbangkan sebagai kata konteks.
![4](https://hackmd.io/_uploads/r1GsudH0R.png)
Mengingat ada 8 kata unik (V=8) dan 3 neuron (N=3):
Matriks bobot input (Winput) akan berukuran 8×3, yang berarti setiap kata dalam korpus direpresentasikan oleh vektor 3 dimensi di lapisan tersembunyi.
Matriks bobot output (Woutput) berukuran 3×8, yang menghubungkan lapisan tersembunyi dengan representasi kata-kata konteks yang diprediksi oleh model.
```
from gensim.models import Word2Vec

model = Word2Vec(corpus, size=3, window=1)
```
Code tersebut menjelaskan bagaimana model Word2Vec dapat diimplementasikan menggunakan gensim dengan size=3 untuk jumlah neuron dan window=1 untuk window size.

#### 3.1 Forward Propagation dalam Skip-Gram
Forward propagation adalah bagian dari proses pelatihan jaringan saraf, di mana model menghasilkan distribusi probabilitas kata konteks (y_pred) berdasarkan kata target yang diberikan. Matriks embedding kata yang digunakan dalam model Skip-Gram adalah W_input dan W_output. Matriks ini terus dioptimalkan selama pelatihan untuk menangkap hubungan yang lebih baik antara kata-kata.
![5](https://hackmd.io/_uploads/BkihjOHRC.png)
Lapisan input adalah vektor one-hot berdimensi V, di mana V adalah jumlah kata unik (vocabulary) dalam korpus. Setiap elemen dari vektor ini adalah nol, kecuali satu elemen yang menunjukkan kata target (input). Karena vektor input adalah one-hot, hanya satu baris dari W_input yang diambil, yang sesuai dengan kata target tersebut. Hal ini membuat W_input berfungsi seperti tabel lookup (pencarian) untuk kata target.

#### 3.2 Matriks Embedding Skip-Gram
Dengan memprediksi kata-kata konteks secara efektif, model mampu memperbaiki matriks embedding (Winput dan Woutput) sehingga representasi kata-kata tersebut menjadi lebih bermakna dalam ruang vektor. Winput dan Woutput adalah matriks bobot yang dioptimalkan selama proses pelatihan. Setiap baris dalam matriks ini mewakili sebuah vektor kata (word vector).
![6](https://hackmd.io/_uploads/HyR4fYH00.png)
Kata "passes" memiliki vektor [0.1, 0.2, 0.7] dan kata "should" memiliki vektor [-2, 0.2, 0.8]. Karena ukuran vektor ditetapkan menjadi 3 (size=3), kata-kata tersebut direpresentasikan dalam ruang vektor tiga dimensi (3D).
![7](https://hackmd.io/_uploads/B1eLQKrCA.png)
Optimasi matriks embedding ini bertujuan untuk merepresentasikan kata-kata secara lebih bermakna dalam ruang vektor. Hasilnya, model mampu menangkap hubungan antara kata-kata berdasarkan konteks yang mereka bagikan.

#### 3.3 Hidden Layer Skip-Gram
![8](https://hackmd.io/_uploads/ryP2mKS0R.png)
Vektor tersembunyi ℎ dihasilkan dari perkalian antara matriks embedding kata input (𝑊𝑖𝑛𝑝𝑢𝑡) dan vektor input satu-hot (𝑥)
$$h = W_{input}^T \cdot x  \in \mathbb{R}^{N} \tag{12}$$Hidden Layer h menyimpan representasi vektor yang lebih kompak dari kata target dan berfungsi sebagai input untuk layer output. Dengan optimasi yang tepat, layer ini dapat memberikan representasi yang baik untuk kata-kata dalam ruang vektor, sehingga model dapat belajar untuk memprediksi kata-kata konteks dengan lebih akurat

#### 3.4 Layer Output Softmax dalam Model Skip-Gram
Layer output dalam model Skip-Gram menghasilkan distribusi probabilitas 𝑉-dimensional dari semua kata unik dalam korpus, berdasarkan kata t. Dalam statistik, probabilitas kondisional dari 𝐴 diberikan 𝐵 dilambangkan sebagai 𝑝(𝐴∣𝐵). Probabilitas kondisional ini dihitung menggunakan fungsi softmax
$$p(w_{context}|w_{center}) = \frac{exp(W_{output_{(context)}} \cdot h)}{\sum^V_{i=1}exp(W_{output_{(i)}} \cdot h)} \in \mathbb{R}^{1} \tag{13}$$ Di mana 𝑊𝑜𝑢𝑡𝑝𝑢𝑡(𝑖) adalah vektor baris ke-𝑖 dari matriks embedding output, dan 𝑊𝑜𝑢𝑡𝑝𝑢𝑡(𝑐𝑜𝑛𝑡𝑒𝑥𝑡) adalah vektor baris yang sesuai dengan kata konteks. Dan 𝑉 adalah jumlah kata unik dalam korpus, dan ℎ adalah layer tersembunyi yang memiliki dimensi (𝑁×1).
Proses ini diulang sebanyak 𝑉 kali untuk mendapatkan distribusi probabilitas kondisional untuk setiap kata unik dalam korpus, berdasarkan kata t
$$\left[ \begin{array}{c} p(w_{1}|w_{center}) \\ p(w_{2}|w_{center}) \\ p(w_{3}|w_{center}) \\ \vdots \\ p(w_{V}|w_{center}) \end{array} \right] = \frac{exp(W_{output} \cdot h)}{\sum^V_{i=1}exp(W_{output_{(i)}} \cdot h)} \in \mathbb{R}^{V}\tag{14}$$
Dalam persamaan ini, 𝑊𝑜𝑢𝑡𝑝𝑢𝑡 di penyebut memiliki ukuran 𝑉×𝑁. Mengalikan 𝑊𝑜𝑢𝑡𝑝𝑢𝑡 dengan ℎ yang berukuran 𝑁×1 akan menghasilkan vektor produk titik berukuran 𝑉×1.
![9](https://hackmd.io/_uploads/BkBdItHCA.png)

#### 3.5 Backward Propagation dalam Model Skip-Gram
Backward propagation dalam model Skip-Gram bertujuan untuk menghitung kesalahan prediksi dan memperbarui matriks bobot 𝜃 untuk mengoptimalkan representasi vektor kata.
![10](https://hackmd.io/_uploads/S1K5DKBAA.png) Kesalahan prediksi adalah perbedaan antara distribusi probabilitas kata yang dihitung dari layer output softmax (𝑦𝑝𝑟𝑒𝑑) dan distribusi probabilitas yang sebenarnya (𝑦𝑡𝑟𝑢𝑒) dari kata konteks ke-𝑐.
![11](https://hackmd.io/_uploads/SJiawKBA0.png) Kesalahan prediksi untuk semua kata konteks 
𝐶 dijumlahkan untuk menghitung gradien bobot untuk memperbarui matriks bobot, sesuai dengan persamaan (18) dan (19).
![12](https://hackmd.io/_uploads/Hkdf_tBAR.png) Seiring dengan pengoptimalan matriks bobot, kesalahan prediksi untuk semua kata dalam vektor kesalahan prediksi $\sum_{(c=1)}^C e_c$ akan berkonvergensi menuju 0.

### 4. Numerical demonstration
#### 4.1 Menghitung Hidden (Projection) Layer dalam Model Skip-Gram
![13](https://hackmd.io/_uploads/ByLzFFHAC.png) Dalam contoh ini, kata target yang dipilih adalah "passes". Dengan ukuran jendela yang ditetapkan sebesar 1, maka kata-kata konteks yang relevan adalah "the" dan "who".

#### 4.2 Softmax Output Layer dalam Model Skip-Gram
![14](https://hackmd.io/_uploads/rkOsFKBC0.png) 
Fungsi softmax menghitung probabilitas kata konteks berdasarkan kata target. Probabilitas ini mencerminkan seberapa besar kemungkinan kata konteks muncul ketika kata target digunakan. Output dari fungsi softmax akan menghasilkan vektor probabilitas berukuran 1×𝑉 yang menunjukkan kemungkinan setiap kata unik dalam korpus menjadi kata konteks untuk kata target yang diberikan.

#### 4.3 Menjumlahkan prediction error dari kata konteks
![15](https://hackmd.io/_uploads/ByNroFH0R.png)
Kesalahan prediksi (prediction errors) untuk setiap kata konteks dihitung sebagai selisih antara probabilitas yang diprediksi (ypred) dan probabilitas yang sebenarnya (ytrue) untuk setiap kata konteks. Setelah menghitung kesalahan untuk masing-masing kata konteks, kesalahan tersebut dijumlahkan. Proses ini bertujuan untuk mendapatkan gradien yang akan digunakan untuk memperbarui bobot matriks (weight matrices) dalam model

#### 4.4 Perhitungan Gradien untuk Matriks Bobot Input (∇𝑊input)
Gradien dari matriks bobot input, dinyatakan sebagai  $\frac{\partial J}{\partial W_{output}}$, dihitung menggunakan persamaan $\frac{\partial J}{\partial W_{\text{input}}} = x \cdot \left( W_{\text{output}}^T \sum_{c=1}^{C} e_c \right)$
Gradien ini digunakan untuk memperbarui bobot input, sehingga model dapat belajar dari kesalahan yang dibuat selama prediksi kata konteks.
![17](https://hackmd.io/_uploads/S1I6ntr0A.png)
Penambahan $W_{output}^T \sum^C_{c=1} e_c$ membantu dalam memperhitungkan semua kesalahan yang terjadi dalam jendela konteks, yang memberikan informasi tentang bagaimana bobot harus diperbarui.
Dengan mengalikan $W_{output}^T \sum^C_{c=1} e_c$  dengan vektor one-hot encoded 𝑥, pembaruan bobot akan fokus hanya pada vektor kata yang sesuai dengan kata t.

#### 4.5 perhitungan gradien untuk matriks bobot output (∇𝑊output)
Gradien dari matriks bobot output, dinyatakan sebagai $\frac{\partial J}{\partial W_{output}}$, dihitung menggunakan persamaan $\frac{\partial J}{\partial W_{\text{output}}} = h \cdot \sum_{c=1}^{C} e_c$
Gradien ini menunjukkan bagaimana setiap elemen dalam matriks bobot output harus diperbarui berdasarkan kesalahan prediksi yang dihasilkan untuk setiap kata konteks.
![16](https://hackmd.io/_uploads/H1Ff0tSCA.png)
Pembaruan dilakukan untuk seluruh matriks bobot output, mencerminkan bahwa semua kata dalam kosakata dipertimbangkan dalam konteks kesalahan prediksi. Hal ini penting karena model perlu menyesuaikan hubungan antar kata di seluruh kosakata untuk meningkatkan akurasi prediksi kata konteks yang diharapkan.

#### 4.6 Pembaruan Matriks Bobot (Weight Matrices) melalui Backward Propagation
![18](https://hackmd.io/_uploads/Bkdwb9SCA.png)
![19](https://hackmd.io/_uploads/SkenZqBAR.png)
Semua bobot dalam 𝑊output diperbarui, yang berarti setiap vektor kata dalam matriks ini disesuaikan. Namun, dalam 𝑊input, hanya satu baris vektor yang diperbarui, yaitu yang sesuai dengan kata target.
Proses pembaruan matriks bobot adalah langkah penting dalam pelatihan model Skip-Gram. Pembaruan dilakukan secara terpisah untuk 𝑊input dan 𝑊output, dengan fokus pada satu kata target pada setiap iterasi. Metodologi ini memanfaatkan SGD untuk efisiensi dalam pembaruan bobot, memungkinkan model untuk belajar dari setiap contoh secara individual, sehingga memperbaiki representasi kata di ruang vektor.
 
## Perhitungan Manual Skip-Gram
### Contoh kata = ibu beli sayur di pasar
Kosakata: ["ibu", "beli", "sayur", "di", "pasar"]
Indeks: "ibu": 0, "beli": 1, "sayur": 2, "di": 3, "pasar": 4
Misalkan bobot awal pada layer embedding (layer pertama) adalah sebagai berikut (dimensi 5 untuk kosakata, dan 5 untuk embedding): $$W = \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.5 & 0.4 & 0.3 & 0.2 & 0.1 \\
0.2 & 0.3 & 0.4 & 0.5 & 0.6 \\
0.6 & 0.5 & 0.4 & 0.3 & 0.2 \\
0.1 & 0.1 & 0.1 & 0.1 & 0.1
\end{bmatrix}$$
#### Iterasi-1
Dari pasangan ("beli","ibu"):
- Input (X): One-hot encoding untuk "beli" menjadi [0,1,0,0,0]
- Output (y): One-hot encoding untuk "ibu" menjadi [1,0,0,0,0]
1. Hitung Aktivasi Layer Pertama
$$z = X \cdot W = [0, 1, 0, 0, 0] \cdot W $$$$z= [0.5, 0.4, 0.3, 0.2, 0.1]$$
2. Hitung skor output sebelum softmax
$$z' = z \cdot W' \quad (\text W' \text{ adalah bobot dari layer kedua})$$Misalkan bobot layer kedua adalah $$W' =
\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
\end{bmatrix}$$ sehingga $$z' = [0.35, 0.35, 0.35, 0.35, 0.35]$$
3. Hitung Softmax untuk 𝑧′
$$e^{z'} = [1.419067, 1.419067, 1.419067, 1.419067, 1.419067]$$ $$\sum_{j} e^{z'_j} = 1.419067 + 1.419067 + 1.419067 + 1.419067 + 1.419067 = 7.095335$$ $$\text{softmax}(z_i') = \frac{e^{z_i'}}{\sum_{j} e^{z_j'}}$$ Jadi, hasil dari softmax untuk 𝑧′ adalah $$\hat{y} = \text{softmax}(z') = [0.2, 0.2, 0.2, 0.2, 0.2]$$

#### Iterasi 2
Dari pasangan ("beli","sayur"):
- Input (X): One-hot encoding untuk "beli" menjadi [0,1,0,0,0]
- Output (y): One-hot encoding untuk "sayur" menjadi [0,0,1,0,0]
1. Hitung Aktivasi Layer Pertama
$$z = X \cdot W = [0, 1, 0, 0, 0] \cdot W $$$$z= [0.55, 0.45, 0.35, 0.25, 0.15]$$
2. Hitung skor output sebelum softmax
$$z' = z \cdot W' \quad (\text W' \text{ adalah bobot dari layer kedua})$$$$z' = [0.425, 0.425, 0.425, 0.425, 0.425]$$
3. Hitung Softmax untuk 𝑧′
$$e^{z'} = [1.528,1.528,1.528,1.528,1.528]$$ $$\sum_{j} e^{z'_j} = 1.528+1.528+1.528+1.528+1.528=7.64$$ $$\text{softmax}(z_i') = \frac{e^{z_i'}}{\sum_{j} e^{z_j'}}$$ Jadi, hasil dari softmax untuk 𝑧′ adalah $$\hat{y} = \text{softmax}(z') = [0.200,0.200,0.200,0.200,0.200]$$

## Code Perhitungan
```
import tensorflow as tf
import numpy as np

# 1. Definisikan kosakata
vocab = ["ibu", "beli", "sayur", "di", "pasar"]
vocab_size = len(vocab)
word_to_index = {word: idx for idx, word in enumerate(vocab)}
print("Kosakata dan Indeks:", word_to_index)

# 2. Buat pasangan (target, context)
pairs = [
    ("beli", "ibu"),
    ("beli", "sayur"),
    ("sayur", "beli"),
    ("sayur", "di"),
    ("di", "sayur"),
    ("di", "pasar"),
    ("pasar", "di")
]
print("Pasangan Target dan Context:", pairs)

# 3. One-Hot Encoding
def one_hot(word):
    vector = np.zeros(vocab_size)
    vector[word_to_index[word]] = 1
    return vector

# Mengubah pasangan menjadi bentuk input dan output
X = np.array([one_hot(target) for target, context in pairs])
y = np.array([one_hot(context) for target, context in pairs])
print("Input (X):\n", X)
print("Output (y):\n", y)

# 4. Membangun Model
embedding_dim = 5  # Dimensi embedding
model = tf.keras.Sequential([
    tf.keras.layers.Dense(embedding_dim, input_shape=(vocab_size,), activation='linear'),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model telah dibangun dan disusun.")

# 5. Melatih Model
# Meningkatkan epoch dan menampilkan hasil setiap 50 epoch untuk pemantauan
history = model.fit(X, y, epochs=500, verbose=0)

# Menampilkan hasil pelatihan
for epoch in range(0, 500, 50):
    print(f"Epoch {epoch + 1} - Loss: {history.history['loss'][epoch]:.4f}, Accuracy: {history.history['accuracy'][epoch]:.4f}")

# 6. Menampilkan Embedding
embedding_layer = model.layers[0]
embeddings = embedding_layer.get_weights()[0]

print("Embedding:")
for word, idx in word_to_index.items():
    print(f"Embedding untuk '{word}': {embeddings[idx]}")

```
