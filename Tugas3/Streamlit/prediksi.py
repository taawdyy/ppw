import pandas as pd
import re
from tqdm import tqdm
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st
import pickle
from sklearn import preprocessing
import requests
from bs4 import BeautifulSoup

# Fungsi untuk crawling artikel menggunakan requests dan BeautifulSoup
def crawl_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Memastikan permintaan berhasil
        soup = BeautifulSoup(response.content, 'html.parser')

        # Mengambil judul
        title_element = soup.find('h1', class_='text-cnn_black')
        title = title_element.get_text().strip() if title_element else 'Judul tidak ditemukan'

        # Mengambil Isi
        content_div = soup.find('div', class_='detail-text')
        content = "\n".join([p.get_text().strip() for p in content_div.find_all('p')]) if content_div else 'Isi artikel tidak ditemukan'

        # Mengambil tanggal
        date_div = soup.find('div', class_='text-cnn_grey text-sm mb-4')
        date_text = date_div.text.strip() if date_div else 'Tanggal tidak ditemukan'

        # Mengambil kategori
        category_meta = soup.find("meta", attrs={'name': 'dtk:namakanal'})
        category = category_meta['content'].strip() if category_meta and 'content' in category_meta.attrs else 'Kategori tidak ditemukan'

        return {'Title': title, 'Content': content, 'Date': date_text, 'Category': category}
    except requests.RequestException as e:
        print(f"Error fetching article: {e}")
        return None

# Fungsi untuk membersihkan teks
def clean_lower(text):
    return text.lower() if isinstance(text, str) else text

def clean_punct(text):
    clean_patterns = re.compile(r'[0-9]|[/(){}\[\]\|@,;_]|[^a-z ]')
    text = clean_patterns.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _normalize_whitespace(text):
    corrected = re.sub(r'\s+', ' ', text)
    return corrected.strip()

def clean_stopwords(text):
    stopword = set(stopwords.words('indonesian'))
    text = ' '.join(word for word in text.split() if word not in stopword)
    return text.strip()

def sastrawistemmer(text):
    factory = StemmerFactory()
    st = factory.create_stemmer()
    return ' '.join(st.stem(word) for word in tqdm(text.split()) if word in text)

# Streamlit User Interface
def main():
    st.title("Prediksi Kategori Berita Online")
    st.write("Masukkan URL artikel dari situs berita untuk memprediksi kategorinya.")

    url_input = st.text_input("Masukkan URL Artikel", "")
    if st.button("Prediksi"):
        if url_input:
            article_data = crawl_article(url_input)

            if article_data and article_data['Content']:
                df = pd.DataFrame([article_data])

                # Proses pembersihan teks
                df['lower case'] = df['Content'].apply(clean_lower)
                df['tanda baca'] = df['lower case'].apply(clean_punct)
                df['spasi'] = df['tanda baca'].apply(_normalize_whitespace)
                df['stopwords'] = df['spasi'].apply(clean_stopwords)
                df['stemming'] = df['stopwords'].apply(sastrawistemmer)

                # Membuat VSM
                filename_vectorizer = 'tfidf_vectorizer.sav'
                tfidf_vectorizer = pickle.load(open(filename_vectorizer, 'rb'))
                corpus = df['stemming'].tolist()
                x_tfidf = tfidf_vectorizer.transform(corpus)
                feature_names = tfidf_vectorizer.get_feature_names_out()
                tfidf_df = pd.DataFrame(x_tfidf.toarray(), columns=feature_names)
                cat_df = df["Category"]
                tfidf_df['Category'] = cat_df.values

                # Encode label kategori
                label_encoder = preprocessing.LabelEncoder()
                tfidf_df['Category'] = label_encoder.fit_transform(tfidf_df['Category'])

                # Load model dan prediksi
                filename_model = 'lr_model.sav'
                lr_model = pickle.load(open(filename_model, 'rb'))

                y_pred = lr_model.predict(tfidf_df.drop(['Category'], axis=1))
                
                # Mengubah pemetaan kategori
                category_map = {0: 'ekonomi', 1: 'olahraga'}
                
                y_pred_labels = [category_map[pred] for pred in y_pred]

                st.write(f"Hasil prediksi kategori berita: **{y_pred_labels[0]}**")
            else:
                st.write("Konten artikel tidak dapat diambil. Pastikan URL yang dimasukkan benar.")
        else:
            st.write("Silakan masukkan URL artikel terlebih dahulu.")

if __name__ == "__main__":
    st.set_page_config(page_title="News Classification", page_icon="📰")
    main()
