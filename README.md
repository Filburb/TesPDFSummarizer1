# Text Summarizer (MiniLM)

## Deskripsi Singkat
** Text Summarizer (MiniLM)** adalah aplikasi berbasis **Streamlit** yang dapat meringkas teks panjang atau dokumen ilmiah dengan pendekatan **semantic similarity**.  
Aplikasi ini memanfaatkan model **paraphrase-multilingual-MiniLM-L12-v2** untuk memahami makna antar kalimat, bukan sekadar potongan teks secara statistik.

Pengguna dapat:
- Memasukkan teks secara langsung atau mengunggah file `.txt` / `.pdf`.
- Memilih panjang ringkasan (pendek, sedang, panjang).
- Mendapatkan hasil ringkasan otomatis dalam **Bahasa Indonesia** (dengan terjemahan otomatis jika teks sumber bukan berbahasa Indonesia).
- Mengunduh hasil ringkasan dalam format `.txt`.

Kelompok 13:
221112207-Filbert Wijaya
221113189-Kevin Wijaya

url aplikasi
https://tespdfsummarizer1-pdpvlufzgavco6rfvbws4a.streamlit.app/

##  Fitur Utama
- **Multilingual Support:** Dapat meringkas teks dalam berbagai bahasa (Inggris, Indonesia, dll).  
- **Semantic-based Summarization:** Menggunakan SentenceTransformer dan PageRank .  
- **Auto Translation:** Otomatis menerjemahkan hasil ringkasan ke Bahasa Indonesia.  
- **Streamlit UI:** Antarmuka web yang interaktif dan ringan.  
- **File Upload Support:** Menerima input teks maupun file `.pdf` / `.txt`.  

---

## Petunjuk Instalasi

environment  python 3.10

1. Clone Repositori
- git clone https://github.com/Filburb/TesPDFSummarizer1.git

2. Buat dan Aktifkan Virtual Environment
- python -m venv venv
- source venv/bin/activate        # di Linux/Mac
- venv\Scripts\activate           # di Windows

3. Instal Dependensi
- pip install -r requirements.txt

4. Jalankan Aplikasi
- streamlit run main.py

5. Akses Aplikasi
- http://localhost:8501





