import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend sebelum import pyplot
import matplotlib.pyplot as plt
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google_play_scraper import Sort, reviews as gp_reviews
import numpy as np
import re

# Login ke HF (ambil token dari secrets.toml)
# Pastikan Anda memiliki file .streamlit/secrets.toml dengan token
if 'huggingface' in st.secrets:
    login(token=st.secrets.huggingface.token, add_to_git_credential=False)

st.set_page_config(page_title="ABSA LINE Reviews", layout="wide")
st.title("Analisis Sentimen Berbasis Aspek — LINE Reviews")
st.markdown("### Buka Sidebar Untuk Memulai Analisis")

# ==========================================
# 1. RESOURCE & PREPROCESSING
# ==========================================

@st.cache_data
def get_text_resources():
    """Memuat dictionary normalisasi dan stopwords"""
    norm_dict = {
        "gak": "tidak", "ga": "tidak", "gk": "tidak", "tdk": "tidak", "enggak": "tidak", 
        "nggak": "tidak", "kagak": "tidak",
        "gw": "saya", "gua": "saya", "aku": "saya", "sy": "saya", "gue": "saya",
        "lu": "kamu", "lo": "kamu", "km": "kamu", "kalian": "kamu",
        "yg": "yang", "kalo": "kalau", "klo": "kalau", 
        "krn": "karena", "karna": "karena", 
        "utk": "untuk", "untk": "untuk", 
        "dgn": "dengan", "dlm": "dalam", 
        "sdh": "sudah", "udh": "sudah", "udah": "sudah",
        "blm": "belum", 
        "jg": "juga", "jga": "juga", 
        "tp": "tetapi", "tapi": "tetapi", 
        "aja": "saja", "aj": "saja", 
        "bgt": "sangat", "banget": "sangat", 
        "knp": "kenapa", "napa": "kenapa", 
        "gmn": "bagaimana", "gimana": "bagaimana", 
        "bs": "bisa", "bisaa": "bisa", "gabisa": "tidak bisa", "ga bisa": "tidak bisa",
        "trus": "terus", "trs": "terus", 
        "jd": "jadi", "jdi": "jadi", 
        "pdhl": "padahal", 
        "bnyk": "banyak", 
        "sm": "sama", 
        "lbh": "lebih", 
        "dr": "dari",
        "eror": "error", "erorr": "error", "errorr": "error", 
        "apk": "aplikasi", "apps": "aplikasi", "aplikasine": "aplikasi",
        "hp": "ponsel", "handphone": "ponsel", 
        "download": "unduh", "donlot": "unduh", 
        "update": "pembaruan", "updet": "pembaruan", 
        "login": "masuk", "log in": "masuk", "sign in": "masuk",
        "log out": "keluar", 
        "notif": "notifikasi", 
        "verif": "verifikasi", 
        "no": "nomor", "nmr": "nomor", 
        "pw": "kata sandi", "password": "kata sandi", "sandi": "kata sandi",
        "chat": "pesan", "chatting": "pesan", 
        "call": "panggilan", "nelpon": "panggilan", "telfon": "panggilan",
        "voom": "timeline", "line voom": "timeline",
        "tolong": "mohon", "pls": "mohon", "please": "mohon", "plis": "mohon",
        "balikin": "kembalikan", 
        "ilang": "hilang", 
        "tau": "tahu", 
        "liat": "lihat", 
        "cuman": "hanya", "cuma": "hanya",
        "makasih": "terima kasih", "tq": "terima kasih",
        "mulu": "terus", "melulu": "terus",
        "bener": "benar",
        "sampe": "sampai",
        "kapan": "kapan",
        "kocakk": "kocak"
    }

    custom_stopwords = {
        "dan", "atau", "tetapi", "tapi", "juga", 
        "yang", "di", "ke", "dari", "pada", "dalam", "untuk", "bagi", "kepada", "oleh",
        "ini", "itu", "tersebut", "disini", "disitu", 
        "saya", "aku", "kamu", "dia", "mereka", "kita", "kami", "anda", "kalian",
        "bisa", "dapat", "akan", "sedang", "sudah", "telah", "masih", "belum",
        "ada", "adalah", "ialah", "merupakan", "sebagai", "seperti",
        "sih", "dong", "deh", "kok", "lah", "mah", "kan", "pun", "doang",
        "nya", 
        "saja", "aja", 
        "padahal", "walaupun", "meskipun", 
        "karena", "sebab", "akibat", "sehingga", "maka",
        "terus", "lalu", "kemudian", "akhirnya",
        "sangat", "banget", "sekali", "terlalu",
        "mohon", "tolong", "harap", "silakan", 
        "terima", "kasih", "makasih",
        "tanya", "tahu", "kasih", "banyak", "sedikit", "kurang", "lebih",
        "hari", "tanggal", "bulan", "tahun", "jam", "waktu",
        "kali", "tiap", "setiap", 
        "apa", "kenapa", "bagaimana", "dimana", "siapa", 
        "halo", "hai", "min", "admin", "kak", "gan", "sis", "bro"
    }
    
    return norm_dict, custom_stopwords

def local_clean_text(text):
    """
    Preprocessing lengkap: Cleaning -> Normalization -> Stopword Removal
    """
    if not isinstance(text, str):
        return ""
    
    # Load resources (Cached)
    norm_dict, custom_stopwords = get_text_resources()

    # 1. Lowercase
    text = text.lower()
    
    # 2. Hapus URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # 3. Penanganan Normalisasi Frasa (Sebelum Split)
    # Mengubah "ga bisa" -> "tidak bisa" agar tidak terpisah
    phrase_fixes = {
        "ga bisa": "tidak bisa",
        "gabisa": "tidak bisa",
        "log in": "masuk",
        "sign in": "masuk",
        "log out": "keluar",
        "line voom": "timeline"
    }
    for phrase, replacement in phrase_fixes.items():
        text = text.replace(phrase, replacement)

    # 4. Hapus Angka & Simbol (Hanya sisakan huruf dan spasi)
    # Pattern ini menghapus semua kecuali huruf a-z dan spasi
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 5. Hapus whitespace berlebih
    text = re.sub(r"\s+", " ", text).strip()
    
    # 6. Tokenisasi (Split)
    words = text.split()
    
    # 7. Loop Normalisasi & Stopword Removal
    cleaned_words = []
    for word in words:
        # Normalisasi: Cek di dictionary, jika tidak ada pakai kata asli
        word = norm_dict.get(word, word)
        
        # Stopword Removal:
        # Masukkan kata jika: (BUKAN stopword DAN panjang > 1) ATAU (kata adalah "tidak")
        # Kata "tidak" dikecualikan dari stopword removal karena penting untuk sentimen negatif
        if (word not in custom_stopwords and len(word) > 1) or word == "tidak":
            cleaned_words.append(word)
            
    # 8. Gabungkan kembali
    return " ".join(cleaned_words)

# ==========================================
# 2. LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():
    """Load the three fine-tuned models from HuggingFace"""
    models = {}
    model_names = {
        "Topic_1": "Cheruno/Topic_1",
        "Topic_2": "Cheruno/Topic_2", 
        "Topic_3": "Cheruno/Topic_3"
    }
    
    for topic, model_name in model_names.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            models[topic] = pipeline("text-classification", 
                                   model=model, 
                                   tokenizer=tokenizer,
                                   return_all_scores=True)
            st.success(f"Model {topic} berhasil dimuat")
        except Exception as e:
            st.error(f"Gagal memuat model {topic}: {str(e)}")
    
    return models

# ==========================================
# 3. PREDICTION FUNCTIONS
# ==========================================

def predict_sentiment_for_topic(text, model_pipeline):
    """Predict sentiment using the loaded model pipeline"""
    try:
        # Jika teks kosong setelah preprocessing, anggap Netral
        if not text or text.strip() == "":
            return "Netral"

        results = model_pipeline(text)
        # Get the highest score prediction
        best_pred = max(results[0], key=lambda x: x['score'])
        
        # Map numerical labels to text labels
        label_mapping = {
            'LABEL_0': 'Negatif',
            'LABEL_1': 'Netral', 
            'LABEL_2': 'Positif'
        }
        
        predicted_label = best_pred['label']
        return label_mapping.get(predicted_label, predicted_label)
        
    except Exception as e:
        # Jika error, return Netral agar aplikasi tidak crash
        return "Netral"

def predict_all_topics(text, models):
    """Predict sentiment for all three topics with LOCAL preprocessing"""
    # 1. Gunakan preprocessing lokal yang sudah diperbarui
    cleaned_text = local_clean_text(text)
    
    predictions = {}
    for topic in ["Topic_1", "Topic_2", "Topic_3"]:
        if topic in models:
            predictions[topic] = predict_sentiment_for_topic(cleaned_text, models[topic])
        else:
            predictions[topic] = "Netral"
    
    return predictions, cleaned_text

# ==========================================
# 4. SIDEBAR & MAIN APP LOGIC
# ==========================================

st.sidebar.header("Pilih Jumlah Ulasan Yang Akan Di Analisis")
n = st.sidebar.selectbox("Jumlah ulasan:", [10, 50, 100, 500, 1000], index=2)

if st.sidebar.button("🚀 Jalankan Analisis", type="primary"):
    
    # Load sentiment analysis models
    with st.spinner("Memuat model sentimen..."):
        models = load_models()
    
    if not models:
        st.error("Gagal memuat model sentimen. Silakan coba lagi.")
        st.stop()
    
    # Scraping function
    @st.cache_data
    def scrape_reviews(n):
        """Scrape reviews from Google Play Store"""
        try:
            with st.spinner(f"Mengambil {n} ulasan terbaru..."):
                data, _ = gp_reviews(
                    "jp.naver.line.android", 
                    lang="id", 
                    country="id",
                    sort=Sort.NEWEST, 
                    count=n
                )
                df = pd.DataFrame(data)
                
                # Mapping Columns
                available_cols = []
                col_mapping = {
                    'content': ['content', 'review', 'reviewText', 'text'],
                    'score': ['score', 'rating', 'stars'],
                    'at': ['at', 'date', 'reviewCreatedVersion', 'time']
                }
                
                final_df = pd.DataFrame()
                
                # Map content column
                content_col = next((col for col in col_mapping['content'] if col in df.columns), None)
                if content_col:
                    final_df['content'] = df[content_col]
                else:
                    return pd.DataFrame()
                
                # Map score column
                score_col = next((col for col in col_mapping['score'] if col in df.columns), None)
                final_df['score'] = df[score_col] if score_col else 5
                
                # Map date column
                date_col = next((col for col in col_mapping['at'] if col in df.columns), None)
                final_df['at'] = df[date_col] if date_col else pd.Timestamp.now()
                
                return final_df
                
        except Exception as e:
            st.error(f"Gagal mengambil ulasan: {str(e)}")
            return pd.DataFrame()

    # Scrape reviews
    df = scrape_reviews(n)
    
    if df.empty:
        st.warning("Gagal mengambil ulasan dari Google Play Store. Menggunakan data sample untuk demo...")
        sample_reviews = [
            "Line sangat bagus untuk chat dengan teman-teman. Fiturnya lengkap dan mudah digunakan.",
            "Aplikasi sering crash ketika membuka sticker store. Sangat mengganggu.",
            "Login kadang bermasalah, harus coba beberapa kali baru bisa masuk.",
            "Fitur video call jernih banget, cocok untuk meeting online.",
            "Registrasi akun mudah, tapi verifikasi nomor HP agak lama.",
            "Timeline sering lag, bikin frustasi saat scrolling.",
            "Chat backup tidak berfungsi dengan baik, data hilang.",
            "Sticker gratis banyak pilihan, anak-anak suka banget.",
            "Notifikasi kadang tidak muncul, jadi miss chat penting.",
            "Voice message clear, bagus untuk yang malas ketik."
        ]
        
        # Repeat samples to fill 'n'
        df = pd.DataFrame({
            'content': sample_reviews * (n // len(sample_reviews) + 1),
            'score': np.random.randint(1, 6, (n // len(sample_reviews) + 1) * len(sample_reviews)),
            'at': pd.date_range(start='2024-01-01', periods=(n // len(sample_reviews) + 1) * len(sample_reviews), freq='D')
        })
        df = df.head(n)
        st.info(f"Menggunakan {len(df)} ulasan sample untuk demonstrasi")

    st.success(f"Berhasil mengambil {len(df)} ulasan!")
    
    # Show sample reviews
    with st.expander("Lihat contoh ulasan mentah"):
        st.dataframe(df.head(10))

    # Perform ABSA inference
    with st.spinner("Melakukan preprocessing & analisis sentimen..."):
        progress_bar = st.progress(0)
        
        # Initialize result columns
        topic_mapping = {
            "Topic_1": "Topic_1_Pengalaman_Umum_Penggunaan_LINE",
            "Topic_2": "Topic_2_Fitur_Tambahan", 
            "Topic_3": "Topic_3_Login_dan_Registrasi_Akun"
        }
        
        for topic in topic_mapping.values():
            df[topic] = ""
        
        df["cleaned_content"] = ""
        
        # Process each review
        for idx, row in df.iterrows():
            # Panggil prediksi (yang didalamnya memanggil local_clean_text)
            predictions, cleaned_text = predict_all_topics(row["content"], models)
            
            # Store cleaned text
            df.at[idx, "cleaned_content"] = cleaned_text
            
            # Store predictions
            for topic, sentiment in predictions.items():
                readable_topic = topic_mapping[topic]
                df.at[idx, readable_topic] = sentiment
            
            # Update progress
            progress_bar.progress((idx + 1) / len(df))
        
        progress_bar.empty()

    # Topic columns for results
    topic_columns = [
        "Topic_1_Pengalaman_Umum_Penggunaan_LINE",
        "Topic_2_Fitur_Tambahan", 
        "Topic_3_Login_dan_Registrasi_Akun"
    ]

    st.success("Analisis selesai!")

    # ==========================================
    # 5. DISPLAY RESULTS & VISUALIZATION
    # ==========================================
    
    st.subheader("📊 Hasil Analisis")
    
    # Show dataframe with results
    result_df = df[["content", "cleaned_content"] + topic_columns].copy()
    
    with st.expander("Perbandingan Teks Asli vs Teks Bersih (Preprocessing)", expanded=True):
        st.write("Perhatikan bagaimana 'gak', 'log in', dan stopwords ditangani:")
        comparison_df = df[["content", "cleaned_content"]].head(10)
        comparison_df.columns = ["Teks Asli", "Teks Bersih"]
        st.dataframe(comparison_df, use_container_width=True)
    
    st.dataframe(result_df, use_container_width=True)

    # Analisis per Topik
    st.subheader("📈 Analisis per Topik")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, topic in enumerate(topic_columns):
        with [col1, col2, col3][idx]:
            st.markdown(f"**{topic.replace('_', ' ')}**")
            
            sentiment_counts = df[topic].value_counts()
            
            if not sentiment_counts.empty:
                fig, ax = plt.subplots(figsize=(6, 6))
                
                colors = {'Positif': '#28a745', 'Negatif': '#dc3545', 'Netral': '#ffc107'}
                sentiment_colors = [colors.get(x, '#6c757d') for x in sentiment_counts.index]
                
                wedges, texts, autotexts = ax.pie(
                    sentiment_counts.values, 
                    labels=sentiment_counts.index,
                    autopct='%1.1f%%',
                    colors=sentiment_colors,
                    startangle=90
                )
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.set_title(f"Distribusi Sentimen", fontsize=10, fontweight='bold')
                st.pyplot(fig)
                plt.close()
                
                st.write("Detail:")
                for sentiment, count in sentiment_counts.items():
                    st.write(f"• {sentiment}: {count}")
            else:
                st.write("Tidak ada data.")

    # Overall sentiment distribution
    st.subheader("📊 Distribusi Sentimen Keseluruhan")
    
    all_sentiments = []
    for topic in topic_columns:
        all_sentiments.extend(df[topic].tolist())
    
    all_sentiments = [s for s in all_sentiments if s and s.strip()]
    
    if all_sentiments:
        overall_counts = pd.Series(all_sentiments).value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = {'Positif': '#28a745', 'Negatif': '#dc3545', 'Netral': '#ffc107'}
        sentiment_colors = [colors.get(x, '#6c757d') for x in overall_counts.index]

        wedges, texts, autotexts = ax.pie(
            overall_counts.values, 
            labels=overall_counts.index,
            autopct='%1.1f%%',
            colors=sentiment_colors,
            startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        ax.set_title("Distribusi Sentimen Keseluruhan", fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

        # Summary statistics
        st.subheader("📋 Ringkasan Statistik")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Ulasan", len(df))
            st.metric("Rata-rata Rating", f"{df['score'].mean():.2f}/5")
        
        with col2:
            positive_ratio = (pd.Series(all_sentiments) == 'Positif').sum() / len(all_sentiments) * 100
            negative_ratio = (pd.Series(all_sentiments) == 'Negatif').sum() / len(all_sentiments) * 100
            st.metric("Sentimen Positif", f"{positive_ratio:.1f}%")
            st.metric("Sentimen Negatif", f"{negative_ratio:.1f}%")
    else:
        st.warning("Tidak ada data sentimen untuk ditampilkan")

    # Download results
    st.subheader("💾 Unduh Hasil")
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 Unduh hasil sebagai CSV",
        data=csv,
        file_name=f"absa_line_reviews_{n}.csv",
        mime="text/csv"
    )

# Information about the app
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Tentang Aplikasi")
st.sidebar.markdown("""
Aplikasi ini melakukan Analisis Sentimen Berbasis Aspek (ABSA) 
pada ulasan LINE Messenger menggunakan model IndoBERT yang 
telah di-fine-tune untuk 3 topik:

1. **Pengalaman Umum Penggunaan**
2. **Fitur Tambahan** 3. **Login dan Registrasi Akun**

Preprocessing Custom:
- Normalisasi kata alay (gak -> tidak)
- Stopword removal (kecuali 'tidak')
- Cleaning simbol & URL
""")
