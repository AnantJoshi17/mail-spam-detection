import streamlit as st
import pandas as pd
import tempfile
import os
import time
from src.pipeline.prediction_pipeline import PredictionPipeline

# Page config
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
.big-title {font-size: 40px; font-weight: bold; color: #4CAF50;}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e1e1e;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def get_pipeline():
    return PredictionPipeline(load_models=True)

pipeline = get_pipeline()

# Header
st.markdown('<div class="big-title">📧 Spam Email Classifier</div>', unsafe_allow_html=True)
st.caption("Smart ML-based email classification system")

# Sidebar
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Choose Mode", ["Single Email", "Batch Processing"])

# ---------------- SINGLE EMAIL ----------------
if mode == "Single Email":
    st.subheader("✉️ Analyze Email")

    email_text = st.text_area(
        "Paste Email Content",
        height=250,
        placeholder="Type or paste email here..."
    )

    col1, col2 = st.columns([1,1])

    with col1:
        if st.button("🔍 Analyze", use_container_width=True):
            if email_text.strip():
                with st.spinner("Analyzing email..."):
                    result = pipeline.predict_single_email(email_text)
                    prediction = result['prediction']
                    confidence = result.get('confidence', 0)

                    if prediction == "Spam":
                        st.error(f"🚨 SPAM DETECTED")
                    else:
                        st.success(f"✅ SAFE EMAIL (HAM)")

                    st.progress(int(confidence))
                    st.write(f"Confidence: {confidence:.2f}%")
            else:
                st.warning("Enter email content first")

    with col2:
        st.markdown("### 💡 Tips")
        st.info("Spam emails often contain:\n- Urgent language\n- Suspicious links\n- Money offers")

# ---------------- BATCH MODE ----------------
else:
    st.subheader("📂 Batch Email Processing")

    uploaded_file = st.file_uploader("Upload MBOX file", type=['mbox','txt'])

    if uploaded_file:
        if st.button("🚀 Process", use_container_width=True):
            with st.spinner("Processing file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                df = pipeline.predict_mbox_file(tmp_path)

                spam_count = len(df[df['Prediction'] == 'Spam'])
                ham_count = len(df[df['Prediction'] == 'Ham'])

                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(df))
                col2.metric("Spam", spam_count)
                col3.metric("Ham", ham_count)

                st.dataframe(df.head(20), use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results", csv, "results.csv")

                os.unlink(tmp_path)
