import re
import numpy as np
import networkx as nx
import nltk
from sentence_transformers import util

nltk.data.path.append("nltk_data")

def clean_text(text):
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"\s?\([^)]*?(et al\.|[0-9]{4})[^)]*?\)", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"(studi|hasil lengkap|lihat) (terdapat|dapat ditemukan) di (Lampiran|Appendix|Tabel).*?\.", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(\d+(\.\d+)*|[A-Z]\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    return text.strip()

def semantic_summarize(text, model, num_sentences=5):
    cleaned_text = clean_text(text)
    sentences = nltk.sent_tokenize(cleaned_text)
    if len(sentences) <= num_sentences:
        return cleaned_text

    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    nx_graph = nx.from_numpy_array(sim_matrix)
    try:
        scores = nx.pagerank(nx_graph)
    except nx.PowerIterationFailedConvergence:
        print("PERINGATAN: Analisis PageRank gagal, fallback sederhana digunakan.")
        scores = {i: np.sum(sim_matrix[i]) for i in range(len(sentences))}

    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[2])
    summary = " ".join([s[1] for s in top_sentences])
    return summary
