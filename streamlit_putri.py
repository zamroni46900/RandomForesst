import streamlit as st
import time
import pickle
import pandas as pd
import requests
from streamlit_lottie import st_lottie  # Pastikan sudah install streamlit-lottie

# --- SET PAGE CONFIG (Harus paling atas) ---
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="🦋",
    layout="wide"
)

# Fungsi untuk memuat animasi Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- HALAMAN DIMULAI ---

with st.spinner("Memuat Streamlit Putri Saleha🦋"):
    # Simulasi loading singkat
    time.sleep(10)  # Tambahkan delay kecil (opsional)

    # Muat animasi Lottie
    lottie_loading = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9tFPv.json ")
    if lottie_loading:
        st_lottie(lottie_loading, height=200, key="loading")

st.success("Horee Sudah Selesai")

# Fungsi untuk menampilkan icon kupu-kupu dengan animasi
def butterfly_icon():
    butterfly_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="32" height="32">
        <g transform="translate(0, 0)">
            <path fill="#FF69B4" d="M256 16C123.5 16 16 123.5 16 256s107.5 240 240 240 240-107.5 240-240S388.5 16 256 16zm-64 352c-44.1 0-80-35.9-80-80s35.9-80 80-80c13.6 0 26.3 3.4 37.5 9.3-4.9 13.4-7.5 27.9-7.5 42.7 0 23.3 9.1 44.5 24 60.2-9.3 8.1-21.3 12.8-34 12.8zm128 0c-12.7 0-24.7-4.7-34-12.8 14.9-15.7 24-36.9 24-60.2 0-14.8-2.6-29.3-7.5-42.7 11.2-5.9 23.9-9.3 37.5-9.3 44.1 0 80 35.9 80 80s-35.9 80-80 80z">
                <animateTransform attributeName="transform"
                    attributeType="XML"
                    type="rotate"
                    from="0 256 256"
                    to="360 256 256"
                    dur="5s"
                    repeatCount="indefinite"/>
            </path>
        </g>
    </svg>
    """
    st.markdown(f'<div style="text-align: center; margin-bottom: 20px;">{butterfly_svg}</div>', unsafe_allow_html=True)

# Load model dan data
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Import font Quicksand + Styling CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand :wght@400;600&display=swap');

    body {
        font-family: 'Quicksand', sans-serif;
    }

    .main {
        background: linear-gradient(to bottom right, #FFE4E1, #FFF0F5);
        padding: 20px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #DB7093;
        animation: fadeInDown 1s ease-out;
    }

    /* Animasi Header */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stMetric {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(255, 105, 180, 0.2);
        transition: transform 0.2s ease-in-out;
    }
    .stMetric:hover {
        transform: translateY(-8px);
    }
    .stMetric label {
        font-weight: bold !important;
        color: #C71585 !important;
    }
    .stMetric div {
        font-size: 24px !important;
        color: #DB7093 !important;
    }

    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(255, 105, 180, 0.2);
    }

    .stTab {
        background-color: #fff0f5;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(255, 105, 180, 0.1);
        margin-bottom: 20px;
    }

    .css-1aumxhk {
        background-color: #DB7093;
        color: white;
    }

    .footer {
        text-align: center;
        color: #999;
        margin-top: 40px;
        animation: blink 2s infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    </style>
""", unsafe_allow_html=True)

# Header dengan animasi
st.markdown("""
<div class="fadeInDown" style="
    background: linear-gradient(to right, #FFB6C1, #FF69B4);
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 30px;">
    🦋 Analisis Sentimen dengan Random Forest
</div>
<div style='text-align: center; color: #999; margin-bottom: 30px;'>Visualisasi Hasil Model Klasifikasi Sentimen</div>
""", unsafe_allow_html=True)

# Tab untuk navigasi
tab1, tab2, tab3 = st.tabs(["📊 Evaluasi Model", "🔧 Skenario 1", "⚙️ Skenario 2"])

with tab1:
    st.header("📈 Evaluasi Model Utama")
    butterfly_icon()
    
    st.subheader("📋 Metrics Utama")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Accuracy", f"{model_data['accuracy']:.4f}", help="Proporsi prediksi yang benar dari semua prediksi")
    with col2:
        st.metric("🔍 Recall (Positif)", f"{model_data['classification_report']['positif']['recall']:.4f}", 
                help="Kemampuan model menemukan semua sampel positif")
    with col3:
        st.metric("⚖️ F1-Score (Positif)", f"{model_data['classification_report']['positif']['f1-score']:.4f}", 
                help="Rata-rata harmonik precision dan recall")
    
    st.subheader("📑 Classification Report")
    report_df = pd.DataFrame(model_data['classification_report']).transpose()
    st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Reds'), use_container_width=True)

with tab2:
    st.header("🔧 Skenario 1 - Variasi n_estimators")
    butterfly_icon()
    
    for i, result in enumerate(model_data['scenario1_results']):
        st.subheader(f"🔄 Model dengan n_estimators = {result['params']['n_estimators']}")
        
        cols = st.columns([1,1,1,2])
        with cols[0]:
            st.metric("🎯 Accuracy", f"{result['accuracy']:.4f}")
        with cols[1]:
            st.metric("🔍 Recall", f"{result['recall']:.4f}")
        with cols[2]:
            st.metric("⚖️ F1-Score", f"{result['f1']:.4f}")
        with cols[3]:
            params_str = "\n".join([f"{k}: {v}" for k, v in result['params'].items()])
            st.text_area("Parameter Model", value=params_str, height=100, key=f"params1_{i}")
        
        if i < len(model_data['scenario1_results'])-1:
            st.markdown("---")

with tab3:
    st.header("⚙️ Skenario 2 - Variasi max_depth")
    butterfly_icon()
    
    for i, result in enumerate(model_data['scenario2_results']):
        st.subheader(f"📏 Model dengan max_depth = {result['params']['max_depth']}")
        
        cols = st.columns([1,1,1,2])
        with cols[0]:
            st.metric("🎯 Accuracy", f"{result['accuracy']:.4f}")
        with cols[1]:
            st.metric("🔍 Recall", f"{result['recall']:.4f}")
        with cols[2]:
            st.metric("⚖️ F1-Score", f"{result['f1']:.4f}")
        with cols[3]:
            params_str = "\n".join([f"{k}: {v}" for k, v in result['params'].items()])
            st.text_area("Parameter Model", value=params_str, height=100, key=f"params2_{i}")
        
        if i < len(model_data['scenario2_results'])-1:
            st.markdown("---")

# Footer
st.markdown("<div class='footer'>🦋 Aplikasi Analisis Sentimen © 2023</div>", unsafe_allow_html=True)