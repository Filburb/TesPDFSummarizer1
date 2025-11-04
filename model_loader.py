from sentence_transformers import SentenceTransformer

def load_model():
    """
    Memuat model multilingual ringan untuk embedding & pemrosesan bahasa.
    """
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    return model