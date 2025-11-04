from deep_translator import GoogleTranslator

def translate_to_indonesian(text: str) -> str:
    """
    Menerjemahkan teks apa pun ke Bahasa Indonesia
    menggunakan GoogleTranslator (tanpa API key).
    """
    if not text.strip():
        return ""
    try:
        return GoogleTranslator(source="auto", target="id").translate(text)
    except Exception as e:
        print("Terjadi kesalahan saat menerjemahkan:", e)
        return text
