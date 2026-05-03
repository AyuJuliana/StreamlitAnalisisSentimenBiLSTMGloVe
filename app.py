import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import os
import json
import io
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Analisis Sentimen Bank Digital",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main gradient orange */
    .gradient-orange {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 1rem;
    }
    .card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        
        min-height: 180px;
    }
    .card-border-orange {
        border-top: 4px solid #f97316;
    }
    .card-border-yellow {
        border-top: 4px solid #eab308;
    }
    .metric-box {
        background: #fff7ed;
        border: 2px solid #fed7aa;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-box-yellow {
        background: #fefce8;
        border: 2px solid #fef08a;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-box-green {
        background: #f0fdf4;
        border: 2px solid #bbf7d0;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-box-red {
        background: #fef2f2;
        border: 2px solid #fecaca;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    .badge-positif {
        background: #dcfce7;
        color: #166534;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .badge-negatif {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .step-box-orange {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .step-box-yellow {
        background: #fefce8;
        border-left: 4px solid #eab308;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .sidebar-logo {
        font-size: 1.5rem;
        font-weight: bold;
        color: #f97316;
        margin-bottom: 1rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff7ed 0%, #ffffff 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        border: none;
        border-radius: 0.75rem;
        font-weight: bold;
        padding: 0.75rem 2rem;
        width: 100%;
        font-size: 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ea580c, #c2410c);
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(249,115,22,0.4);
    }
    .stTextArea textarea {
        border: 2px solid #fed7aa !important;
        border-radius: 0.75rem !important;
    }
    .stTextArea textarea:focus {
        border: 2px solid #f97316 !important;
        box-shadow: 0 0 0 4px rgba(249,115,22,0.15) !important;
    }
    h1, h2, h3 { color: #1f2937; }
    .stAlert { border-radius: 0.75rem; }
    /* Progress bar color */
    .stProgress > div > div { background: linear-gradient(90deg, #f97316, #ea580c); }
</style>
""", unsafe_allow_html=True)

# ============================================
# DOWNLOAD NLTK DATA
# ============================================
@st.cache_resource
def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk()

# ============================================
# CUSTOM STOPWORDS CLASS
# ============================================
class CustomStopwords:
    def __init__(self):
        factory = StopWordRemoverFactory()
        self.default_stopwords = factory.get_stop_words()
        self.sentiment_words = {
            'tidak', 'bukan', 'belum', 'jangan', 'tanpa', 'tak', 'tiada', 'nihil', 'gak', 'ga', 'nggak',
            'sangat', 'sekali', 'banget', 'amat', 'terlalu', 'paling', 'lebih', 'kurang', 'agak', 'cukup',
            'bagus', 'baik', 'senang', 'puas', 'suka', 'mantap', 'keren', 'oke', 'recommended', 'mudah',
            'cepat', 'lancar', 'praktis', 'efisien',
            'buruk', 'jelek', 'kecewa', 'lambat', 'lama', 'susah', 'sulit', 'ribet', 'rumit', 'error',
            'gagal', 'macet', 'lemot', 'berat'
        }
        self.banking_terms = {
            'transfer', 'saldo', 'bunga', 'admin', 'biaya', 'transaksi', 'rekening', 'tabungan', 'deposito',
            'kartu', 'debit', 'kredit', 'virtual', 'account', 'pocket', 'kantong', 'limit', 'nominal',
            'withdraw', 'tarik', 'tunai', 'setor', 'kirim', 'terima', 'mutasi', 'riwayat', 'notifikasi',
            'verifikasi', 'pin', 'password', 'otp', 'keamanan', 'security'
        }
        self.jago_features = {'jago', 'syariah', 'pocket', 'pockets', 'gopay', 'bibit', 'flip', 'qris'}
        self.additional_stopwords = {
            'yg', 'utk', 'dgn', 'dll', 'dsb', 'dst', 'krn', 'spy', 'ttg', 'tsb', 'yth',
            'nya', 'ku', 'mu', 'kah', 'lah', 'pun', 'kita', 'kami', 'mereka', 'dia',
            'apakah', 'mengapa', 'kapan', 'dimana', 'kemana', 'bagaimana'
        }
        self.stopwords = self._create_final_stopwords()

    def _create_final_stopwords(self):
        final_stopwords = set(self.default_stopwords)
        final_stopwords.update(self.additional_stopwords)
        words_to_keep = self.sentiment_words | self.banking_terms | self.jago_features
        final_stopwords = final_stopwords - words_to_keep
        return final_stopwords

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stopwords]


# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    # Load GloVe
    glove_path = os.path.join(models_dir, "glove_model (2).pkl")
    with open(glove_path, 'rb') as f:
        glove_model = pickle.load(f)

    if isinstance(glove_model, dict):
        GLOVE_WORD2IDX = glove_model.get('word2idx', glove_model.get('word_to_idx', {}))
        GLOVE_WORD_VECTORS = glove_model.get('word_vectors', glove_model.get('embeddings', np.array([])))
        GLOVE_EMBEDDING_DIM = glove_model.get('embedding_dim', GLOVE_WORD_VECTORS.shape[1] if len(GLOVE_WORD_VECTORS.shape) > 1 else 200)
        GLOVE_VOCAB_SIZE = glove_model.get('vocab_size', len(GLOVE_WORD2IDX))
    else:
        GLOVE_WORD2IDX = getattr(glove_model, 'word2idx', getattr(glove_model, 'word_to_idx', {}))
        GLOVE_WORD_VECTORS = getattr(glove_model, 'word_vectors', getattr(glove_model, 'embeddings', np.array([])))
        GLOVE_EMBEDDING_DIM = getattr(glove_model, 'embedding_dim', GLOVE_WORD_VECTORS.shape[1] if len(GLOVE_WORD_VECTORS.shape) > 1 else 200)
        GLOVE_VOCAB_SIZE = getattr(glove_model, 'vocab_size', len(GLOVE_WORD2IDX))

    # Load BiLSTM
    bilstm_path = os.path.join(models_dir, "best_bilstm_model_kfold (3).pkl")
    with open(bilstm_path, 'rb') as f:
        model_data = pickle.load(f)

    # Load slang words
    slang_path = os.path.join(models_dir, "slangwords.json")
    if os.path.exists(slang_path):
        with open(slang_path, 'r', encoding='utf-8') as f:
            SLANG_DICT = json.load(f)
    else:
        # Try txt fallback
        slang_txt_path = os.path.join(models_dir, "slangwords.txt")
        if os.path.exists(slang_txt_path):
            SLANG_DICT = {}
            with open(slang_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        SLANG_DICT[parts[0]] = parts[1]
        else:
            SLANG_DICT = {}

    custom_sw = CustomStopwords()

    BILSTM_PARAMS = model_data.get('bilstm_params', {})
    LABEL_MAP = model_data.get('label_map', {0: 'Negatif', 1: 'Positif'})

    return {
        'GLOVE_WORD2IDX': GLOVE_WORD2IDX,
        'GLOVE_WORD_VECTORS': GLOVE_WORD_VECTORS,
        'GLOVE_EMBEDDING_DIM': GLOVE_EMBEDDING_DIM,
        'GLOVE_VOCAB_SIZE': GLOVE_VOCAB_SIZE,
        'model_data': model_data,
        'BILSTM_PARAMS': BILSTM_PARAMS,
        'LABEL_MAP': LABEL_MAP,
        'SLANG_DICT': SLANG_DICT,
        'custom_stopwords': custom_sw,
    }


# ============================================
# BILSTM PREDICTOR
# ============================================
class BiLSTMPredictor:
    def __init__(self, model_data):
        self.params = model_data['bilstm_params']
        self.weights = model_data['bilstm_weights']
        self.label_map = model_data['label_map']
        self.input_dim = self.params['input_dim']
        self.hidden_dim = self.params['hidden_dim']
        self.output_dim = self.params['output_dim']

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def lstm_cell_forward(self, x, h_prev, c_prev, direction='forward'):
        if direction == 'forward':
            Wf, Uf, bf = self.weights['Wf_f'], self.weights['Uf_f'], self.weights['bf_f']
            Wi, Ui, bi = self.weights['Wi_f'], self.weights['Ui_f'], self.weights['bi_f']
            Wc, Uc, bc = self.weights['Wc_f'], self.weights['Uc_f'], self.weights['bc_f']
            Wo, Uo, bo = self.weights['Wo_f'], self.weights['Uo_f'], self.weights['bo_f']
        else:
            Wf, Uf, bf = self.weights['Wf_b'], self.weights['Uf_b'], self.weights['bf_b']
            Wi, Ui, bi = self.weights['Wi_b'], self.weights['Ui_b'], self.weights['bi_b']
            Wc, Uc, bc = self.weights['Wc_b'], self.weights['Uc_b'], self.weights['bc_b']
            Wo, Uo, bo = self.weights['Wo_b'], self.weights['Uo_b'], self.weights['bo_b']

        f = self.sigmoid(np.dot(x, Wf) + np.dot(h_prev, Uf) + bf)
        i = self.sigmoid(np.dot(x, Wi) + np.dot(h_prev, Ui) + bi)
        c_tilde = self.tanh(np.dot(x, Wc) + np.dot(h_prev, Uc) + bc)
        c = f * c_prev + i * c_tilde
        o = self.sigmoid(np.dot(x, Wo) + np.dot(h_prev, Uo) + bo)
        h = o * self.tanh(c)
        return h, c

    def predict(self, X):
        seq_length = X.shape[0]
        h_f = np.zeros((1, self.hidden_dim))
        c_f = np.zeros((1, self.hidden_dim))
        h_b = np.zeros((1, self.hidden_dim))
        c_b = np.zeros((1, self.hidden_dim))

        h_forwards = []
        for t in range(seq_length):
            x_t = X[t:t+1]
            h_f, c_f = self.lstm_cell_forward(x_t, h_f, c_f, 'forward')
            h_forwards.append(h_f)

        h_backwards = []
        for t in range(seq_length - 1, -1, -1):
            x_t = X[t:t+1]
            h_b, c_b = self.lstm_cell_forward(x_t, h_b, c_b, 'backward')
            h_backwards.insert(0, h_b)

        h_final = np.concatenate([h_forwards[-1], h_backwards[0]], axis=1)
        Wy = self.weights['Wy']
        by = self.weights['by']
        logits = np.dot(h_final, Wy) + by
        output = self.softmax(logits)
        pred_idx = np.argmax(output[0])
        confidence = float(output[0][pred_idx])
        return pred_idx, output[0], confidence


# ============================================
# PREPROCESSING FUNCTIONS
# ============================================
def clean_content(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def normalize_slang(tokens, slang_dict):
    if not slang_dict:
        return tokens
    return [slang_dict.get(token, token) for token in tokens]

def preprocess_text(text, slang_dict, custom_sw):
    text = clean_content(text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = normalize_slang(tokens, slang_dict)
    tokens = custom_sw.remove_stopwords(tokens)
    return ' '.join(tokens)

def text_to_glove_sequence(text, word2idx, word_vectors):
    tokens = text.lower().split()
    embeddings = []
    for token in tokens:
        if token in word2idx:
            idx = word2idx[token]
            embedding = word_vectors[idx]
        else:
            embedding = np.mean(word_vectors, axis=0)
        embeddings.append(embedding)
    if len(embeddings) == 0:
        embeddings = [np.mean(word_vectors, axis=0)]
    return np.array(embeddings)

def predict_sentiment(text, models):
    cleaned_text = preprocess_text(text, models['SLANG_DICT'], models['custom_stopwords'])
    if not cleaned_text.strip():
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': 'Netral',
            'confidence': 0.0,
            'probabilities': {'Negatif': 0.0, 'Positif': 0.0}
        }
    glove_seq = text_to_glove_sequence(cleaned_text, models['GLOVE_WORD2IDX'], models['GLOVE_WORD_VECTORS'])
    predictor = BiLSTMPredictor(models['model_data'])
    pred_idx, probs, confidence = predictor.predict(glove_seq)
    sentiment = models['LABEL_MAP'][pred_idx]
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 2),
        'probabilities': {
            models['LABEL_MAP'][i]: round(float(prob) * 100, 2)
            for i, prob in enumerate(probs)
        }
    }


# ============================================
# LOAD MODELS WITH SPINNER
# ============================================
with st.spinner("⏳ Memuat model GloVe dan BiLSTM..."):
    try:
        models = load_models()
        models_loaded = True
    except Exception as e:
        models_loaded = False
        load_error = str(e)


# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Analisis Sentimen Bank Digital — BiLSTM + GloVe</div>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigasi",
        options=["📊 Dashboard", "💬 Input Teks", "📁 Upload File"],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()
    if models_loaded:
        st.success("✅ Model siap digunakan")
        m = models['BILSTM_PARAMS']
        st.caption(f"GloVe Vocab: {models['GLOVE_VOCAB_SIZE']:,}")
        st.caption(f"Embedding Dim: {models['GLOVE_EMBEDDING_DIM']}")
        st.caption(f"BiLSTM Hidden: {m.get('hidden_dim', 32)}")
    else:
        st.error("❌ Model gagal dimuat")


# ============================================
# PAGE: DASHBOARD
# ============================================
if page == "📊 Dashboard":

    # Welcome
    st.markdown("""
    <div class="gradient-orange" style="margin-bottom:1.5rem">
        <h2 style="color:white;margin:0">Selamat Datang! 👋</h2>
        <p style="color:rgba(255,255,255,0.95);margin-top:0.75rem;line-height:1.7">
            Website ini menggunakan <strong>Custom Bidirectional LSTM</strong> yang dilatih from scratch
            dengan <strong>GloVe Word Embedding (200 dimensi)</strong> untuk menganalisis sentimen ulasan
            Bank Jago. Model menggunakan preprocessing khusus dengan <strong>normalisasi slang words</strong>
            dan <strong>custom stopwords removal</strong>, serta dilengkapi dengan <strong>SMOTE</strong>
            untuk mengatasi imbalanced data dan validasi menggunakan <strong>5-Fold Cross Validation</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Info Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card card-border-orange">
            <h4>📊 Dashboard</h4>
            <p style="color:#6b7280;font-size:0.9rem">Informasi lengkap dataset, hasil SMOTE, GloVe training,
            hyperparameter tuning, dan K-Fold Cross Validation</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card card-border-yellow">
            <h4>💬 Input Teks</h4>
            <p style="color:#6b7280;font-size:0.9rem">Analisis sentimen real-time Binary Classification (Positif/Negatif) dengan
            preprocessing otomatis dan confidence score</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card card-border-orange">
            <h4>📁 Upload File</h4>
            <p style="color:#6b7280;font-size:0.9rem">Batch processing secara massal hingga 1000 ulasan sekaligus dengan statistik detail
            dan export hasil ke CSV</p>
        </div>""", unsafe_allow_html=True)

    # SMOTE Results
    st.markdown("---")
    st.markdown("### ⚖️ SMOTE Class Balancing")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:linear-gradient(to bottom right,#fff1f2,#fff7ed);border:2px solid #fca5a5;border-radius:0.75rem;padding:1.25rem">
            <h4 style="color:#374151">⚠️ Before SMOTE (Imbalanced)</h4>
        </div>""", unsafe_allow_html=True)
        st.markdown("**Class 0 (Negatif):** 2,015 (45.06%)")
        st.progress(0.4506)
        st.markdown("**Class 1 (Positif):** 2,457 (54.94%)")
        st.progress(0.5494)
        st.caption("Total: **4,472 samples**")

    with col2:
        st.markdown("""
        <div style="background:linear-gradient(to bottom right,#f0fdf4,#ecfdf5);border:2px solid #86efac;border-radius:0.75rem;padding:1.25rem">
            <h4 style="color:#374151">✅ After SMOTE (Balanced) — K-Fold CV</h4>
        </div>""", unsafe_allow_html=True)
        st.markdown("**Class 0 (Negatif):** 2,706 (49.44%)")
        st.progress(0.4944)
        st.markdown("**Class 1 (Positif):** 2,767 (50.56%)")
        st.progress(0.5056)
        st.caption("Total K-Fold: **5,473 samples** ↑ +1,001 (+22.39%)")

    # GloVe Training
    st.markdown("---")
    st.markdown("### 🧠 GloVe Word Embedding Training")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-box"><div style="font-size:2rem;font-weight:bold;color:#f97316">5,244</div><div style="color:#6b7280">Vocabulary Size</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box-yellow"><div style="font-size:2rem;font-weight:bold;color:#ca8a04">200</div><div style="color:#6b7280">Embedding Dimension</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><div style="font-size:2rem;font-weight:bold;color:#f97316">0.0134</div><div style="color:#6b7280">Final Loss (Epoch 50)</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Training progress
    col_left, col_right = st.columns([1, 1])
    with col_left:
        with st.container():
            st.markdown("**Training Progress (50 Epochs)**")
            progress_data = {
                'Epoch': [10, 20, 30, 40, 50],
                'Loss': [0.0415, 0.0291, 0.0215, 0.0167, 0.0134]
            }
            df_prog = pd.DataFrame(progress_data)
            st.dataframe(df_prog, use_container_width=True, hide_index=True)

    with col_right:
        # GloVe training loss chart
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_dir, "static", "glove_training_loss.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Grafik penurunan loss GloVe selama 50 epoch — konvergen dari ~0.084 menuju 0.0134", use_container_width=True)

    # Hyperparameter Tuning
    st.markdown("---")
    st.markdown("### 🎛️ Hyperparameter Tuning (Grid Search)")
    st.markdown("""
    <div style="background:linear-gradient(to right,#f0fdf4,#ecfdf5);border:2px solid #86efac;border-radius:0.75rem;padding:1.25rem;margin-bottom:1rem">
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
            <span style="font-size:2rem">🏆</span>
            <div>
                <h4 style="margin:0;color:#374151">Best Configuration</h4>
                <p style="margin:0;color:#6b7280;font-size:0.9rem">Ranked #1 dari 9 kombinasi parameter</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-box-green"><div style="font-size:2rem;font-weight:bold;color:#16a34a">32</div><div style="color:#6b7280">Hidden Units</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box-green"><div style="font-size:2rem;font-weight:bold;color:#16a34a">0.0</div><div style="color:#6b7280">Dropout Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box-green"><div style="font-size:2rem;font-weight:bold;color:#16a34a">89.09%</div><div style="color:#6b7280">Val Accuracy</div></div>', unsafe_allow_html=True)

    # Best Fold
    st.markdown("")
    st.markdown("""
    <div style="background:linear-gradient(to right,#f0fdf4,#ecfdf5);border:2px solid #86efac;border-radius:0.75rem;padding:1.25rem">
        <h4 style="color:#374151">⭐ Best Fold: Fold 2 — Model Tersimpan</h4>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-box-green"><div style="font-size:1.8rem;font-weight:bold;color:#16a34a">90.22%</div><div style="color:#6b7280">Validation Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box-green"><div style="font-size:1.8rem;font-weight:bold;color:#16a34a">0.2458</div><div style="color:#6b7280">Validation Loss</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box-green"><div style="font-size:1.8rem;font-weight:bold;color:#16a34a">0.9022</div><div style="color:#6b7280">Macro F1-Score</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:0.5rem;padding:1rem">
            <b>Negatif (Fold 2)</b><br>
            Precision: <b>89.76%</b><br>
            Recall: <b>90.59%</b><br>
            F1-Score: <b>90.17%</b>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:0.5rem;padding:1rem">
            <b>Positif (Fold 2)</b><br>
            Precision: <b>90.68%</b><br>
            Recall: <b>89.86%</b><br>
            F1-Score: <b>90.26%</b>
        </div>""", unsafe_allow_html=True)

    # Final Test Results
    st.markdown("---")
    st.markdown("### ✅ Final Test Set Evaluation (Best Model — Fold 2)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-box"><div style="font-size:1.8rem;font-weight:bold;color:#f97316">88.55%</div><div style="color:#6b7280">Test Accuracy</div><div style="color:#9ca3af;font-size:0.8rem">495/559</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box-yellow"><div style="font-size:1.8rem;font-weight:bold;color:#ca8a04">88.46%</div><div style="color:#6b7280">Macro Precision</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><div style="font-size:1.8rem;font-weight:bold;color:#f97316">88.59%</div><div style="color:#6b7280">Macro Recall</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box-yellow"><div style="font-size:1.8rem;font-weight:bold;color:#ca8a04">88.51%</div><div style="color:#6b7280">Macro F1-Score</div></div>', unsafe_allow_html=True)

    st.info("**Test Loss: 0.2707** — Model disimpan sebagai `best_bilstm_model_kfold.pkl` (Hidden=32, Dropout=0.0, Vocab=5,244)")

    # Confusion Matrix
    st.markdown("#### Confusion Matrix (Test Set)")
    cm_data = {
        '': ['Aktual Negatif', 'Aktual Positif'],
        'Pred. Negatif': ['✅ 231 (True Negative)', '❌ 36 (False Negative)'],
        'Pred. Positif': ['❌ 28 (False Positive)', '✅ 264 (True Positive)']
    }
    st.dataframe(pd.DataFrame(cm_data).set_index(''), use_container_width=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:linear-gradient(to bottom right,#fff1f2,#fff7ed);border:2px solid #fca5a5;border-radius:0.75rem;padding:1rem">
            <b>Negatif Class (Test)</b><br>
            Precision: <b>86.52%</b><br>
            Recall: <b>89.19%</b><br>
            F1-Score: <b>87.83%</b><br>
            <span style="color:#9ca3af;font-size:0.8rem">Support: 259 samples</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:linear-gradient(to bottom right,#f0fdf4,#ecfdf5);border:2px solid #86efac;border-radius:0.75rem;padding:1rem">
            <b>Positif Class (Test)</b><br>
            Precision: <b>90.41%</b><br>
            Recall: <b>88.00%</b><br>
            F1-Score: <b>89.19%</b><br>
            <span style="color:#9ca3af;font-size:0.8rem">Support: 300 samples</span>
        </div>""", unsafe_allow_html=True)

    # Metodologi
    st.markdown("---")
    st.markdown("### 🔬 Metodologi Penelitian (10 Langkah)")

    steps = [
        ("orange", "1", "Pengumpulan Data",
         "Web scraping ulasan aplikasi Bank Jago dari Google Play Store menggunakan `google_play_scraper` — 5.850 ulasan (02 Jan 2024 – 19 Jun 2024)"),
        ("yellow", "2", "Text Preprocessing",
         "Cleaning → Case folding → Tokenisasi → Normalisasi slang words → Custom stopwords removal → Stemming (Sastrawi)"),
        ("orange", "3", "Labelling Data",
         "Binary labelling berdasarkan rating: rating 1–3 → Negatif (0), rating 4–5 → Positif (1). Divalidasi oleh pakar domain"),
        ("yellow", "4", "Splitting Dataset",
         "Proporsi 80:10:10 — Training: 4.472, Validasi: 559, Testing: 559. Total: 5.590 sampel (random seed=42)"),
        ("orange", "5", "SMOTE Data Training",
         "Oversampling +442 sampel sintetik pada data training — Negatif: 2.015→2.457, Total: 4.472→4.914 (50:50). Hanya pada data training"),
        ("yellow", "6", "GloVe Word Embedding",
         "Custom GloVe 200D dilatih dari scratch — Vocab: 5.244, window size: 5, x_max: 50, α: 0.75, learning rate: 0.05, 50 epochs. Final loss: 0.0134"),
        ("orange", "7", "Baseline Model Bi-LSTM",
         "Training awal: hidden=64, dropout=0.0, lr=0.001, 20 epoch — Val acc baseline: 88.73%, val loss: 0.3028 (epoch 19)"),
        ("yellow", "8", "Hyperparameter Tuning (Grid Search)",
         "9 kombinasi: hidden units (32/64/128) × dropout (0.0/0.1/0.2). Best config: hidden=32, dropout=0.0, val acc=89.09%, val loss=0.2990"),
        ("orange", "9", "5-Fold Cross Validation",
         "Best config dievaluasi dengan 5-Fold CV pada 5.473 sampel (SMOTE K-Fold). Mean val acc: 89.13% (±0.62%), best fold: Fold 2 (val acc: 90.22%)"),
        ("yellow", "10", "Evaluasi Final (Test Set)",
         "Best model (Fold 2) dievaluasi pada test set — Test acc: 88.55% (495/559), test loss: 0.2707, macro F1: 88.51%. Disimpan sebagai `best_bilstm_model_kfold.pkl`"),
    ]

    col1, col2 = st.columns(2)
    for i, (color, num, title, desc) in enumerate(steps):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            bg = "#fff7ed" if color == "orange" else "#fefce8"
            border = "#f97316" if color == "orange" else "#eab308"
            st.markdown(f"""
            <div style="background:{bg};border-left:4px solid {border};border-radius:0.5rem;padding:1rem;margin-bottom:0.75rem;display:flex;gap:1rem;align-items:flex-start">
                <div style="background:{'#fed7aa' if color=='orange' else '#fef08a'};border-radius:9999px;width:2rem;height:2rem;display:flex;align-items:center;justify-content:center;font-weight:bold;color:{'#9a3412' if color=='orange' else '#713f12'};flex-shrink:0">{num}</div>
                <div>
                    <b style="color:#1f2937">{title}</b>
                    <p style="color:#6b7280;font-size:0.85rem;margin:0.25rem 0 0 0">{desc}</p>
                </div>
            </div>""", unsafe_allow_html=True)


# ============================================
# PAGE: INPUT TEKS
# ============================================
elif page == "💬 Input Teks":
    st.markdown("## 💬 Analisis Sentimen Teks")
    st.markdown("**Binary Classification: Positif atau Negatif**")

    if not models_loaded:
        st.error(f"❌ Model gagal dimuat: {load_error}")
        st.stop()

    # Example texts
    examples = [
        "Aplikasi sangat user-friendly dan proses verifikasi cepat sekali. Fitur pocket memudahkan saving! Puas banget pakai Bank Jago!",
        "Aplikasi sering error dan customer service lambat respon. Transaksi gagal terus, sangat mengecewakan! Tidak recommended!",
        "Transfer pakai QRIS gampang banget dan cepet lagi. Biaya admin murah, cocok buat nabung digital. Recommended deh!"
    ]

    with st.container():
        st.markdown("""
        <div style="background:linear-gradient(to right,#fff7ed,#fefce8);border:2px solid #fed7aa;border-radius:0.75rem;padding:1rem;margin-bottom:1rem">
            <b>💡 Contoh Ulasan:</b>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        for i, (col, ex, icon) in enumerate(zip([col1, col2, col3], examples, ["😊", "😞", "😊"])):
            with col:
                if st.button(f"{icon} Contoh {i+1}", key=f"ex_{i}", use_container_width=True):
                    st.session_state['input_text'] = ex

    input_text = st.text_area(
        "✏️ Masukkan Ulasan Bank Digital",
        value=st.session_state.get('input_text', ''),
        placeholder="Contoh: Aplikasi Bank Jago sangat mudah digunakan, transfer cepat dan aman. Fitur pocket juga membantu mengelola keuangan!",
        height=160,
        help="Teks akan otomatis di-preprocessing (normalisasi slang, hapus stopwords)"
    )
    st.caption("ℹ️ Teks akan otomatis di-preprocessing (normalisasi slang, hapus stopwords)")

    analyze_btn = st.button("📊 Analisis Sentimen", use_container_width=True)

    if analyze_btn:
        if not input_text or len(input_text.strip()) < 10:
            st.warning("⚠️ Mohon masukkan minimal 10 karakter.")
        else:
            with st.spinner("🔍 Menganalisis..."):
                result = predict_sentiment(input_text, models)

            st.markdown("---")
            st.markdown("### 📈 Hasil Analisis")

            sentiment = result['sentiment']
            confidence = result['confidence']

            col1, col2 = st.columns([1, 2])
            with col1:
                if sentiment == 'Positif':
                    st.markdown('<div style="text-align:center;padding:1.5rem;background:#f0fdf4;border:2px solid #86efac;border-radius:1rem"><div style="font-size:3rem">😊</div><div class="badge-positif" style="display:inline-block;margin-top:0.5rem">POSITIF</div></div>', unsafe_allow_html=True)
                elif sentiment == 'Negatif':
                    st.markdown('<div style="text-align:center;padding:1.5rem;background:#fef2f2;border:2px solid #fca5a5;border-radius:1rem"><div style="font-size:3rem">😞</div><div class="badge-negatif" style="display:inline-block;margin-top:0.5rem">NEGATIF</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align:center;padding:1.5rem;background:#fefce8;border:2px solid #fef08a;border-radius:1rem"><div style="font-size:3rem">😐</div><br><b>{sentiment.upper()}</b></div>', unsafe_allow_html=True)

            with col2:
                st.markdown(f"**Confidence Score: {confidence}%**")
                st.progress(confidence / 100)

                st.markdown("**Detail Probabilitas:**")
                for label, prob in result['probabilities'].items():
                    color = "green" if label == 'Positif' else "red"
                    st.markdown(f"**{label}:** {prob}%")
                    st.progress(prob / 100)

            with st.expander("🔍 Detail Preprocessing"):
                st.markdown("**Teks setelah preprocessing:**")
                st.code(result['cleaned_text'] or "(kosong setelah preprocessing)", language=None)


# ============================================
# PAGE: UPLOAD FILE
# ============================================
elif page == "📁 Upload File":
    st.markdown("## 📁 Upload File CSV/Excel")
    st.markdown("**Batch processing untuk multiple reviews**")

    if not models_loaded:
        st.error(f"❌ Model gagal dimuat: {load_error}")
        st.stop()

    # Format guide
    with st.expander("ℹ️ Format File yang Diperlukan", expanded=True):
        st.markdown("""
        - ✅ File harus memiliki kolom: `ulasan`, `review`, `text`, `content`, `komentar`, `feedback`, atau `comment`
        - ✅ Format: CSV (.csv) atau Excel (.xlsx, .xls)
        - ✅ Maksimal 1000 baris per upload
        - ✅ Preprocessing otomatis (slang normalization + stopwords removal)
        """)

    uploaded_file = st.file_uploader(
        "Pilih file CSV atau Excel",
        type=["csv", "xlsx", "xls"],
        help="Drag & drop atau klik untuk upload"
    )

    if uploaded_file is not None:
        st.success(f"✅ File dipilih: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        process_btn = st.button("⚙️ Proses File", use_container_width=True)

        if process_btn:
            # Read file
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
                st.stop()

            # Find text column
            text_column = None
            possible_columns = ['ulasan', 'review', 'text', 'content', 'komentar', 'feedback', 'comment']
            for col in possible_columns:
                if col in df.columns:
                    text_column = col
                    break

            if text_column is None:
                st.error(f"❌ File harus memiliki salah satu kolom: {', '.join(possible_columns)}")
                st.info(f"Kolom yang ditemukan: {', '.join(df.columns.tolist())}")
                st.stop()

            # Limit rows
            if len(df) > 1000:
                df = df.head(1000)
                st.warning("⚠️ File dipotong menjadi 1000 baris pertama.")

            st.info(f"📊 Memproses **{len(df)} ulasan** dari kolom `{text_column}`...")

            # Process with progress
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in df.iterrows():
                text = str(row[text_column])
                if text and text.strip() and text.lower() != 'nan':
                    pred = predict_sentiment(text, models)
                    results.append({
                        'No': len(results) + 1,
                        'Ulasan': text[:200],
                        'Sentimen': pred['sentiment'],
                        'Confidence (%)': pred['confidence']
                    })
                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Memproses {idx + 1}/{len(df)}...")

            progress_bar.empty()
            status_text.empty()

            # Results
            st.success(f"✅ Analisis selesai! Berhasil menganalisis **{len(results)} ulasan**")

            results_df = pd.DataFrame(results)

            # Summary
            positif = sum(1 for r in results if r['Sentimen'] == 'Positif')
            negatif = sum(1 for r in results if r['Sentimen'] == 'Negatif')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-box-green"><div style="font-size:2rem;font-weight:bold;color:#16a34a">{positif}</div><div style="color:#6b7280">Positif</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-box-red"><div style="font-size:2rem;font-weight:bold;color:#dc2626">{negatif}</div><div style="color:#6b7280">Negatif</div></div>', unsafe_allow_html=True)
            with col3:
                pct = round(positif / len(results) * 100, 1) if results else 0
                st.markdown(f'<div class="metric-box"><div style="font-size:2rem;font-weight:bold;color:#f97316">{pct}%</div><div style="color:#6b7280">% Positif</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📋 Hasil Analisis")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Download
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="⬇️ Download Hasil (CSV)",
                data=csv_data,
                file_name=f"hasil_analisis_sentimen_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div style="border:3px dashed #fed7aa;border-radius:1.5rem;padding:3rem;text-align:center;background:#fff7ed">
            <div style="font-size:4rem">📂</div>
            <p style="font-size:1.1rem;color:#374151;font-weight:600">Drag & drop file atau gunakan tombol di atas</p>
            <p style="color:#9ca3af">Format: CSV atau Excel (.xlsx, .xls)</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: linear-gradient(135deg, #fff7ed, #ffffff);
    border-top: 1px solid #fed7aa;
    text-align: center;
    padding: 12px 10px;
    font-size: 13px;
    color: #374151;
    z-index: 999;
}
.footer strong {
    color: #f97316;
}
</style>

<div class="footer">
    <div><strong>Analisis Sentimen Bank Digital © 2026</strong></div>
    <div>Powered by GloVe + Bi-LSTM Deep Learning Model</div>
    <div>Ni Komang Ayu Juliana (Informatika, Universitas Udayana)</div>
</div>
""", unsafe_allow_html=True)

# spacer biar konten gak ketutup footer
st.markdown("<br><br><br>", unsafe_allow_html=True)