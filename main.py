import streamlit as st
import io
import fitz
import nltk
from langdetect import detect, LangDetectException
from model_loader import load_model
from summarizer import semantic_summarize
from translator import translate_to_indonesian


@st.cache_resource
def get_model_cached():
    """Memuat model menggunakan cache Streamlit."""
    return load_model()

with st.spinner("Memuat model summarizer (MiniLM)..."):
    model = get_model_cached()

def extract_text_from_pdf(file):
    """Mengambil teks dari seluruh halaman PDF"""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

# ======================
# FRONTEND
# ======================
st.set_page_config(page_title="Semantic Summarizer (MiniLM)", layout="centered")

st.title("📄 Semantic Text Summarizer")
st.write("""
Aplikasi ini menggunakan **SentenceTransformer** (`paraphrase-multilingual-MiniLM-L12-v2`)
untuk membuat ringkasan berbasis makna (semantik).
""")

# Input text area
input_text = st.text_area("Masukkan teks di sini:", height=250)

# Upload file
uploaded_file = st.file_uploader("Atau unggah file (.txt atau .pdf)", type=["txt", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with st.spinner("Mengekstrak teks dari PDF..."):
            input_text = extract_text_from_pdf(uploaded_file)
    else:
        # Asumsikan encoding utf-8 untuk file txt
        try:
            input_text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("Gagal membaca file .txt. Pastikan file menggunakan encoding UTF-8.")
            input_text = ""

# Pilihan panjang ringkasan
summary_length = st.selectbox(
    "Pilih panjang ringkasan:",
    options=["Pendek (3 kalimat)", "Sedang (5 kalimat)", "Panjang (8 kalimat)"],
    index=1
)
length_map = {"Pendek (3 kalimat)": 3, "Sedang (5 kalimat)": 5, "Panjang (8 kalimat)": 8}
num_sentences = length_map[summary_length]

# Tombol ringkas
if st.button("🔍 Ringkas Teks"):
    if input_text.strip():
        try:
            # 1. Deteksi Bahasa
            lang = detect(input_text[:500]) # Ambil 500 karakter pertama untuk deteksi
        except LangDetectException:
            lang = "en" # Default ke bahasa Inggris jika deteksi gagal
            st.warning("Gagal mendeteksi bahasa, diasumsikan Bahasa Inggris.")

        # 2. Meringkas Teks (Secara Semantik)
        with st.spinner("Menganalisis dan meringkas teks..."):
            summary = semantic_summarize(input_text, model, num_sentences=num_sentences)

        # 3. Menerjemahkan (HANYA JIKA BUKAN 'id')
        if lang == 'id':
            summary_final = summary
            st.success("Ringkasan selesai!")
        else:
            with st.spinner(f"Menerjemahkan dari '{lang}' ke Bahasa Indonesia..."):
                summary_final = translate_to_indonesian(summary)
            st.success(f"Ringkasan (diterjemahkan dari '{lang}') selesai!")

        # 4. Tampilkan Hasil
        st.subheader("Hasil Ringkasan (Bahasa Indonesia):")
        st.text_area("Output:", summary_final, height=250)

        # Unduh hasil ringkasan
        buffer = io.BytesIO()
        buffer.write(summary_final.encode("utf-8"))
        buffer.seek(0)
        st.download_button(
            label="Unduh Ringkasan (.txt)",
            data=buffer,
            file_name="ringkasan.txt",
            mime="text/plain"
        )
    else:
        st.warning("Masukkan teks terlebih dahulu sebelum meringkas.")

st.markdown("---")
st.caption("Model: paraphrase-multilingual-MiniLM-L12-v2 (Semantic TextRank) + GoogleTranslator.")
