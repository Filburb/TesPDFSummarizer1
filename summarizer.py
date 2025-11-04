import re
import nltk
import numpy as np
import networkx as nx
from sentence_transformers import util


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("MENGUNDUH: Data NLTK 'punkt' tidak ditemukan. Mengunduh...")
    nltk.download('punkt')


def clean_text(text):
    """
    Membersihkan teks secara lebih agresif, 
    khususnya untuk teks ilmiah atau copy-paste.
    """
    
    # 1. Ganti newline, tab, dan carriage return dengan spasi tunggal
    text = re.sub(r"[\n\t\r]+", " ", text)
    
    # 2. Hapus sitasi dalam kurung, cth: (Author et al., 2017) atau (Nama, 2018)
    text = re.sub(r"\s?\([^)]*?(et al\.|[0-9]{4})[^)]*?\)", "", text)
    
    # 3. Hapus URL dan email (jika ada)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    
    # 4. Hapus referensi ke lampiran/tabel
    text = re.sub(r"(studi ablasi|eksperimen|hasil lengkap|detail|lihat) (dapat ditemukan|terdapat) di (Lampiran|Appendix|Tabel|Bagian).*?\.", ".", text, flags=re.IGNORECASE)
    
    # 5. Hapus nomor bagian di awal kalimat, cth "5.1 " atau "C. "
    text = re.sub(r"^\s*(\d+(\.\d+)*|[A-Z]\.)\s+", "", text, flags=re.MULTILINE)
    
    # 6. Hapus spasi ganda yang mungkin muncul setelah penghapusan
    text = re.sub(r"\s+", " ", text)
    
    # 7. Hapus spasi sebelum tanda baca
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    
    return text.strip()


def semantic_summarize(text, model, num_sentences=5):
    """
    Meringkas teks menggunakan SentenceTransformer embeddings dan algoritma TextRank (PageRank).
    """
    
    # 1. Bersihkan Teks
    cleaned_text = clean_text(text)
    
    # 2. Pecah Teks menjadi Kalimat
    # Kita gunakan 'english' secara eksplisit karena ini yang diunduh 'punkt'
    # Tokenizer ini cukup baik untuk çoğu bahasa (termasuk Indonesia)
    sentences = nltk.sent_tokenize(cleaned_text, language='english')
    
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
