import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend sebelum import pyplot
import matplotlib.pyplot as plt
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google_play_scraper import Sort, reviews as gp_reviews
import numpy as np

# Login ke HF (ambil token dari secrets.toml)
login(token=st.secrets.huggingface.token, add_to_git_credential=False)

st.set_page_config(page_title="ABSA LINE Reviews", layout="wide")
st.title("Analisis Sentimen Berbasis Aspek — LINE Reviews")

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
        return best_pred['label']
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Netral"

def predict_all_topics(text, models, preprocessor=None):
    """Predict sentiment for all three topics with preprocessing"""
    # Preprocess text first
    if preprocessor:
        cleaned_text = preprocessor.preprocess(text)
    else:
        # Fallback preprocessing
        import re
        cleaned_text = text.lower() if isinstance(text, str) else ""
        cleaned_text = re.sub(r"http\S+|www\S+|https\S+", "", cleaned_text)
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        cleaned_text = re.sub(r"\d+", "", cleaned_text)
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
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
    
    # Load preprocessing model
    with st.spinner("Memuat model preprocessing..."):
        preprocessor = load_preprocessing_model()
    
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
                    "com.linecorp.line", 
                    lang="id", 
                    country="id",
                    sort=Sort.NEWEST, 
                    count=n
                )
                df = pd.DataFrame(data)
                # Keep only content and some metadata
                return df[["content", "score", "at"]].copy()
        except Exception as e:
            st.error(f"Gagal mengambil ulasan: {str(e)}")
            return pd.DataFrame()

    # Scrape reviews
    df = scrape_reviews(n)
    
    if df.empty:
        st.error("Gagal mengambil ulasan. Silakan coba lagi.")
        st.stop()

    st.success(f"Berhasil mengambil {len(df)} ulasan!")
    
    # Show sample reviews
    with st.expander("Lihat contoh ulasan"):
        st.dataframe(df.head(10))

    # Perform ABSA inference
    with st.spinner("Melakukan analisis sentimen..."):
        progress_bar = st.progress(0)
        
        # Initialize result columns
        for topic in ["Topic_1", "Topic_2", "Topic_3"]:
            df[f"{topic}_Sentiment"] = ""
        
        # Add cleaned text column
        df["cleaned_content"] = ""
        
        # Process each review
        for idx, row in df.iterrows():
            predictions, cleaned_text = predict_all_topics(row["content"], models)
            
            # Store cleaned text
            df.at[idx, "cleaned_content"] = cleaned_text
            
            # Map predictions to readable format
            topic_mapping = {
                "Topic_1": "Topic_1_Pengalaman_Umum_Penggunaan_LINE",
                "Topic_2": "Topic_2_Fitur_Tambahan", 
                "Topic_3": "Topic_3_Login_dan_Registrasi_Akun"
            }
            
            for topic, sentiment in predictions.items():
                readable_topic = topic_mapping[topic]
                df.at[idx, f"{readable_topic}"] = sentiment
            
            # Update progress
            progress_bar.progress((idx + 1) / len(df))
        
        progress_bar.empty()

    # Rename columns to match your data format
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
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = {'Positif': '#28a745', 'Negatif': '#dc3545', 'Netral': '#ffc107'}
            sentiment_colors = [colors.get(sentiment, '#6c757d') for sentiment in sentiment_counts.index]
            
            ax.pie(sentiment_counts.values, 
                   labels=sentiment_counts.index,
                   autopct='%1.1f%%',
                   colors=sentiment_colors,
                   startangle=90)
            ax.set_title(f"Distribusi Sentimen\n{topic.replace('_', ' ')}")
            st.pyplot(fig)
            plt.close()

    # Overall sentiment distribution
    st.subheader("📊 Distribusi Sentimen Keseluruhan")
    
    # Combine all sentiments
    all_sentiments = []
    for topic in topic_columns:
        all_sentiments.extend(df[topic].tolist())
    
    overall_counts = pd.Series(all_sentiments).value_counts()
    
    # Create overall pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {'Positif': '#28a745', 'Negatif': '#dc3545', 'Netral': '#ffc107'}
    sentiment_colors = [colors.get(sentiment, '#6c757d') for sentiment in overall_counts.index]

    ax.pie(overall_counts.values, 
           labels=overall_counts.index,
           autopct='%1.1f%%',
           colors=sentiment_colors,
           startangle=90)
    ax.set_title("Distribusi Sentimen Keseluruhan")
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
