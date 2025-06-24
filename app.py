import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend sebelum import pyplot
import matplotlib.pyplot as plt
from huggingface_hub import login, hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google_play_scraper import Sort, reviews as gp_reviews
import numpy as np
import importlib.util
import sys
import re

# Login ke HF (ambil token dari secrets.toml)
login(token=st.secrets.huggingface.token, add_to_git_credential=False)

st.set_page_config(page_title="ABSA LINE Reviews", layout="wide")
st.title("Analisis Sentimen Berbasis Aspek — LINE Reviews")

# Local fallback preprocessing function
def local_clean_text(text):
    """Local fallback preprocessing function"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# Load preprocessing function from HuggingFace
@st.cache_resource
def load_preprocessing_function():
    """Load preprocessing function from HuggingFace Hub"""
    try:
        # Download file preprocessing dari HuggingFace Hub
        preprocessing_file = hf_hub_download(
            repo_id="Cheruno/text_preprocessor",
            filename="modeling_text_preprocessor.py",
            repo_type="model"
        )
        
        # Load module dari file
        spec = importlib.util.spec_from_file_location("preprocessing", preprocessing_file)
        preprocessing_module = importlib.util.module_from_spec(spec)
        sys.modules["preprocessing"] = preprocessing_module
        spec.loader.exec_module(preprocessing_module)
        
        return preprocessing_module.clean_text
    except Exception as e:
        st.warning(f"Gagal memuat preprocessing dari HuggingFace: {str(e)}")
        st.info("Menggunakan preprocessing lokal sebagai fallback...")
        return local_clean_text

# Load models
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

# Prediction functions
def predict_sentiment_for_topic(text, model_pipeline):
    """Predict sentiment using the loaded model pipeline"""
    try:
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
        st.error(f"Error in prediction: {str(e)}")
        return "Netral"

def predict_all_topics(text, models, preprocessor=None):
    """Predict sentiment for all three topics with preprocessing"""
    # Preprocess text first
    if preprocessor:
        cleaned_text = preprocessor(text)  # Call the function directly
    else:
        # Fallback preprocessing
        cleaned_text = local_clean_text(text)
    
    predictions = {}
    for topic in ["Topic_1", "Topic_2", "Topic_3"]:
        if topic in models:
            predictions[topic] = predict_sentiment_for_topic(cleaned_text, models[topic])
        else:
            predictions[topic] = "Netral"
    
    return predictions, cleaned_text

# Sidebar
st.sidebar.header("Pengaturan Analisis")
n = st.sidebar.selectbox("Jumlah ulasan:", [10, 50, 100, 500, 1000], index=2)

if st.sidebar.button("🚀 Jalankan Analisis", type="primary"):
    
    # Load preprocessing function
    with st.spinner("Memuat fungsi preprocessing..."):
        preprocessor = load_preprocessing_function()  # Fixed function name
    
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
                
                # Debug: Show available columns
                st.write("Debug - Available columns:", df.columns.tolist())
                st.write("Debug - Data shape:", df.shape)
                
                # Check which columns are available and map them accordingly
                available_cols = []
                col_mapping = {
                    'content': ['content', 'review', 'reviewText', 'text'],
                    'score': ['score', 'rating', 'stars'],
                    'at': ['at', 'date', 'reviewCreatedVersion', 'time']
                }
                
                final_df = pd.DataFrame()
                
                # Map content column
                content_col = None
                for col in col_mapping['content']:
                    if col in df.columns:
                        content_col = col
                        break
                
                if content_col:
                    final_df['content'] = df[content_col]
                else:
                    st.error("Kolom konten tidak ditemukan dalam data")
                    return pd.DataFrame()
                
                # Map score column
                score_col = None
                for col in col_mapping['score']:
                    if col in df.columns:
                        score_col = col
                        break
                
                if score_col:
                    final_df['score'] = df[score_col]
                else:
                    final_df['score'] = 5  # Default score
                
                # Map date column
                date_col = None
                for col in col_mapping['at']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col:
                    final_df['at'] = df[date_col]
                else:
                    final_df['at'] = pd.Timestamp.now()  # Default timestamp
                
                return final_df
                
        except Exception as e:
            st.error(f"Gagal mengambil ulasan: {str(e)}")
            return pd.DataFrame()

    # Scrape reviews
    df = scrape_reviews(n)
    
    if df.empty:
        st.warning("Gagal mengambil ulasan dari Google Play Store. Menggunakan data sample untuk demo...")
        # Create sample data for demonstration
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
        
        df = pd.DataFrame({
            'content': sample_reviews * (n // len(sample_reviews) + 1),
            'score': np.random.randint(1, 6, n),
            'at': pd.date_range(start='2024-01-01', periods=n, freq='D')
        })
        df = df.head(n)  # Limit to requested number
        
        st.info(f"Menggunakan {len(df)} ulasan sample untuk demonstrasi")

    st.success(f"Berhasil mengambil {len(df)} ulasan!")
    
    # Show sample reviews
    with st.expander("Lihat contoh ulasan"):
        st.dataframe(df.head(10))

    # Perform ABSA inference
    with st.spinner("Melakukan analisis sentimen..."):
        progress_bar = st.progress(0)
        
        # Initialize result columns
        topic_mapping = {
            "Topic_1": "Topic_1_Pengalaman_Umum_Penggunaan_LINE",
            "Topic_2": "Topic_2_Fitur_Tambahan", 
            "Topic_3": "Topic_3_Login_dan_Registrasi_Akun"
        }
        
        for topic in topic_mapping.values():
            df[topic] = ""
        
        # Add cleaned text column
        df["cleaned_content"] = ""
        
        # Process each review
        for idx, row in df.iterrows():
            predictions, cleaned_text = predict_all_topics(row["content"], models, preprocessor)
            
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

    # Display results
    st.subheader("📊 Hasil Analisis")
    
    # Show dataframe with results (including cleaned text for comparison)
    result_df = df[["content", "cleaned_content"] + topic_columns].copy()
    
    # Show comparison between original and cleaned text
    with st.expander("Perbandingan Teks Asli vs Teks Bersih"):
        comparison_df = df[["content", "cleaned_content"]].head(10)
        comparison_df.columns = ["Teks Asli", "Teks Bersih"]
        st.dataframe(comparison_df, use_container_width=True)
    
    st.dataframe(result_df, use_container_width=True)

    # Create aggregation for each topic
    st.subheader("📈 Analisis per Topik")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, topic in enumerate(topic_columns):
        with [col1, col2, col3][idx]:
            st.write(f"**{topic.replace('_', ' ')}**")
            
            # Count sentiments for this topic
            sentiment_counts = df[topic].value_counts()
            
            # Create pie chart only if there's data
            if not sentiment_counts.empty:
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Define colors for each sentiment
                colors = {'Positif': '#28a745', 'Negatif': '#dc3545', 'Netral': '#ffc107'}
                
                # Create color list based on actual sentiments present
                sentiment_colors = []
                for sentiment in sentiment_counts.index:
                    if sentiment in colors:
                        sentiment_colors.append(colors[sentiment])
                    else:
                        sentiment_colors.append('#6c757d')  # Default gray for unknown sentiments
                
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    sentiment_counts.values, 
                    labels=sentiment_counts.index,
                    autopct='%1.1f%%',
                    colors=sentiment_colors,
                    startangle=90
                )
                
                # Style the text
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.set_title(f"Distribusi Sentimen\n{topic.replace('_', ' ')}", 
                           fontsize=10, fontweight='bold')
                st.pyplot(fig)
                plt.close()
                
                # Show sentiment counts as well
                st.write("Detail:")
                for sentiment, count in sentiment_counts.items():
                    st.write(f"• {sentiment}: {count}")
            else:
                st.write("Tidak ada data untuk ditampilkan")

    # Overall sentiment distribution
    st.subheader("📊 Distribusi Sentimen Keseluruhan")
    
    # Combine all sentiments
    all_sentiments = []
    for topic in topic_columns:
        all_sentiments.extend(df[topic].tolist())
    
    # Remove empty values
    all_sentiments = [s for s in all_sentiments if s and s.strip()]
    
    if all_sentiments:
        overall_counts = pd.Series(all_sentiments).value_counts()
        
        # Create overall pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = {'Positif': '#28a745', 'Negatif': '#dc3545', 'Netral': '#ffc107'}
        
        # Create color list based on actual sentiments present
        sentiment_colors = []
        for sentiment in overall_counts.index:
            if sentiment in colors:
                sentiment_colors.append(colors[sentiment])
            else:
                sentiment_colors.append('#6c757d')  # Default gray

        wedges, texts, autotexts = ax.pie(
            overall_counts.values, 
            labels=overall_counts.index,
            autopct='%1.1f%%',
            colors=sentiment_colors,
            startangle=90
        )
        
        # Style the text
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
            
        # Show detailed counts
        st.subheader("Detail Distribusi Sentimen")
        for sentiment, count in overall_counts.items():
            percentage = (count / len(all_sentiments)) * 100
            st.write(f"**{sentiment}**: {count} ulasan ({percentage:.1f}%)")
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
2. **Fitur Tambahan** 
3. **Login dan Registrasi Akun**

Model yang digunakan:
- Cheruno/Topic_1
- Cheruno/Topic_2  
- Cheruno/Topic_3
""")
