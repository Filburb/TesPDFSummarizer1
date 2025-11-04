import re
import nltk
import numpy as np
import networkx as nx
from sentence_transformers import util
import os  # <-- Tambahkan import ini

# ==========================================================
# PERBAIKAN FINAL NLTK:
# ==========================================================
# 1. Tentukan path LOKAL di dalam proyek kita
#    Ini akan membuat folder 'nltk_data' di sebelah summarizer.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_NLTK_DATA = os.path.join(SCRIPT_DIR, 'nltk_data')

# 2. Beri tahu NLTK untuk MENCARI data di path lokal kita
if LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.append(LOCAL_NLTK_DATA)

# 3. Periksa apakah data 'punkt' sudah ada DI LOKASI KITA
#    Kita periksa file spesifik yang gagal (english.pickle)
punkt_path = os.path.join(LOCAL_NLTK_DATA, 'tokenizers', 'punkt', 'english.pickle')

if not os.path.exists(punkt_path):
    print(f"MENGUNDUH: Data NLTK 'punkt' tidak ditemukan. Mengunduh ke {LOCAL_NLTK_DATA}")
    # 4. Jika tidak ada, UNDUH secara paksa ke direktori lokal kita
    nltk.download('punkt', download_dir=LOCAL_NLTK_DATA)
else:
    print("Data NLTK 'punkt' sudah ada.")
# ==========================================================


def clean_text(text):
    """
    Membersihkan teks secara lebih agresif.
    """
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"\s?\([^)]*?(et al\.|[0-9]{4})[^)]*?\)", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"(studi ablasi|eksperimen|hasil lengkap|detail|lihat) (dapat ditemukan|terdapat) di (Lampiran|Appendix|Tabel|Bagian).*?\.", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(\d+(\.\d+)*|[A-Z]\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    return text.strip()


def semantic_summarize(text, model, num_sentences=5):
    """
    Meringkas teks menggunakan SentenceTransformer embeddings.
    """
    cleaned_text = clean_text(text)
    
    # 2. Pecah Teks menjadi Kalimat
    #    Sekarang ini akan 100% menemukan data 'punkt'
    sentences = nltk.sent_tokenize(cleaned_text)
    
    if len(sentences) <= num_sentences:
        return cleaned_text 

    # 3. Buat Embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # 4. Hitung Matriks Similaritas
    sim_matrix = util.cos_sim(embeddings, embeddings)
    sim_matrix_np = sim_matrix.cpu().numpy()

    # 5. Buat Graf dan jalankan PageRank
    nx_graph = nx.from_numpy_array(sim_matrix_np)
    try:
        scores = nx.pagerank(nx_graph)
    except nx.PowerIterationFailedConvergence:
        print("PERINGATAN: Analisis PageRank gagal, menggunakan fallback sederhana.")
        scores = {i: np.sum(sim_matrix_np[i]) for i in range(len(sentences))}

    # 6. Urutkan Kalimat berdasarkan Skor
    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)

    # 7. Ambil N kalimat teratas
    top_sentences = ranked_sentences[:num_sentences]
    
    # 8. Urutkan kembali kalimat teratas berdasarkan urutan aslinya
    top_sentences_sorted_by_index = sorted(top_sentences, key=lambda x: x[2])
    
    # 9. Gabungkan kalimat-kalimat tersebut
    summary = " ".join([s[1] for s in top_sentences_sorted_by_index])
    
    return summary
