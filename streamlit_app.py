
import os
import streamlit as st
from rag_chatbot_pipeline import ClassBasedRAGChatbot

# 0. Page config — must be the very first Streamlit command
st.set_page_config(
    page_title="Stella — Your Mercedes-Benz Chatbot",
    page_icon="⭐",
    layout="centered"
)

# Resolve paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Mercedes-Benz Design System ───────────────────────────────────────────────
# Palette:  Obsidian #0A0A0A  |  Steel  #1C1C1E  |  Silver #C0C0C0
#           Platinum #E8E8E8  |  Star   #FFFFFF   |  Red    #CC0000
# Type:     Display → "Helvetica Neue", Weight 300 (light)
#           Body    → system-ui / Inter
# Signature element: the three-pointed star motif rendered as a pure CSS
#           clip-path badge — no image dependency.
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Reset & base ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [data-testid="stApp"], .stApp {
        background-color: #0A0A0A !important;
        color: #E8E8E8 !important;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }

    /* ── Header / Star badge ───────────────────────────────────────────────── */
    .mb-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.6rem;
        padding: 2.2rem 0 1.6rem;
    }

    /* Three-pointed star — pure CSS, no image */
    .mb-star {
        width: 56px;
        height: 56px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .mb-star::before {
        content: "★";
        font-size: 3rem;
        background: linear-gradient(135deg, #C0C0C0 0%, #FFFFFF 50%, #888 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 8px rgba(192,192,192,0.4));
    }
    .mb-star-ring {
        position: absolute;
        inset: 0;
        border: 1.5px solid rgba(192,192,192,0.35);
        border-radius: 50%;
    }

    /* App title */
    .mb-title {
        font-family: 'Helvetica Neue', Helvetica, 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.9rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #FFFFFF;
        text-align: center;
    }
    .mb-title span {
        background: linear-gradient(90deg, #C0C0C0 0%, #FFFFFF 55%, #C0C0C0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .mb-subtitle {
        font-size: 0.75rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #666;
        text-align: center;
        margin-top: -0.3rem;
    }

    /* Thin rule under header */
    .mb-rule {
        width: 100%;
        max-width: 480px;
        height: 1px;
        background: linear-gradient(90deg, transparent, #C0C0C033, #C0C0C0, #C0C0C033, transparent);
        margin: 0.4rem auto 1.8rem;
    }

    /* ── Chat messages ─────────────────────────────────────────────────────── */
    div[data-testid="stChatMessage"] {
        background-color: #111113 !important;
        border: 1px solid #2A2A2E !important;
        border-radius: 6px !important;
        padding: 1.1rem 1.3rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
        transition: border-color 0.25s ease !important;
    }
    div[data-testid="stChatMessage"]:hover {
        border-color: #C0C0C055 !important;
    }

    /* ── Sender label ──────────────────────────────────────────────────────── */
    .mb-sender {
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #C0C0C0;
        margin-bottom: 0.45rem;
        font-weight: 500;
    }

    /* ── Source expander ───────────────────────────────────────────────────── */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background-color: #1C1C1E !important;
        border: 1px solid #2A2A2E !important;
        border-radius: 4px !important;
        color: #C0C0C0 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stExpanderDetails"] {
        background-color: #0F0F11 !important;
        border: 1px solid #1E1E22 !important;
        border-top: none !important;
        border-radius: 0 0 4px 4px !important;
    }

    /* ── Chat input ────────────────────────────────────────────────────────── */
    div[data-testid="stChatInput"] textarea {
        background-color: #111113 !important;
        border: 1px solid #2A2A2E !important;
        border-radius: 6px !important;
        color: #E8E8E8 !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #C0C0C066 !important;
        box-shadow: 0 0 0 1px #C0C0C033 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #555 !important;
    }

    /* ── Alert / info boxes ────────────────────────────────────────────────── */
    .stAlert {
        background-color: #1C1C1E !important;
        border-color: #2A2A2E !important;
        color: #E8E8E8 !important;
        border-radius: 4px !important;
    }

    /* ── Spinner ───────────────────────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #C0C0C0 !important;
    }

    /* ── Scrollbar ─────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0A0A0A; }
    ::-webkit-scrollbar-thumb { background: #2A2A2E; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #C0C0C066; }

    /* ── Reference cards ───────────────────────────────────────────────────── */
    .ref-card {
        background: #141416;
        border-left: 2px solid #C0C0C066;
        border-radius: 0 4px 4px 0;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.82rem;
        color: #AAA;
        line-height: 1.55;
    }
    .ref-label {
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #C0C0C0;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mb-header">
    <div class="mb-star"><div class="mb-star-ring"></div></div>
    <div class="mb-title"><span>Stella</span></div>
    <div class="mb-subtitle">Mercedes-Benz · Annual Report Assistant</div>
</div>
<div class="mb-rule"></div>
""", unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────────────
gemini_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not gemini_key:
    gemini_key = os.environ.get("GEMINI_API_KEY")

if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
else:
    st.error("🔑 GEMINI_API_KEY is not set. Configure it in Streamlit secrets or as an environment variable.")
    st.stop()

# ── PDF & DB paths ────────────────────────────────────────────────────────────
HARDCODED_PDF = os.path.join(SCRIPT_DIR, "Mercedes-Benz-Group-Report-2024-en.pdf")
PDF_BASE_NAME = os.path.splitext(os.path.basename(HARDCODED_PDF))[0]
DB_DIR = os.path.join(SCRIPT_DIR, f"rag_db_{PDF_BASE_NAME}")


# ── Cached chatbot ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_chatbot():
    bot = ClassBasedRAGChatbot(db_dir=DB_DIR, llm_model="gemini-2.5-flash")
    if os.path.exists(HARDCODED_PDF):
        bot.ingest_pdf(HARDCODED_PDF)
        st.success("✅ Report ingested successfully.")
    else:
        st.error("❌ PDF not found. Place `Mercedes-Benz-Group-Report-2024-en.pdf` in the app directory.")
    return bot


chatbot = get_chatbot()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Guten Tag. I'm Stella, your Mercedes-Benz Annual Report assistant. Ask me anything about the 2024 Group Report — financials, strategy, sustainability, or vehicle programmes.",
            "sources": []
        }
    ]


# ── Render a single message ────────────────────────────────────────────────────
def render_message(message):
    is_assistant = message["role"] == "assistant"
    label = "Stella · Mercedes-Benz" if is_assistant else "You"

    with st.chat_message(message["role"]):
        st.markdown(f'<div class="mb-sender">{label}</div>', unsafe_allow_html=True)
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander(f"View {len(sources)} source reference{'s' if len(sources) > 1 else ''}"):
                for idx, src in enumerate(sources, 1):
                    doc = src["document"]
                    page = doc.metadata.get("page", "?")
                    score = src["score"]
                    st.markdown(
                        f'<div class="ref-card">'
                        f'<div class="ref-label">Reference {idx} — Page {page} &nbsp;·&nbsp; Score {score:.4f}</div>'
                        f'{doc.page_content}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


# ── Display history ────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    render_message(message)

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about strategy, financials, sustainability…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        st.markdown('<div class="mb-sender">Stella · Mercedes-Benz</div>', unsafe_allow_html=True)
        with st.spinner("Retrieving context…"):
            result = chatbot.ask(prompt, initial_k=20, final_k=3)
            answer = result["answer"]
            sources = result["sources"]

        st.markdown(answer)

        if sources:
            with st.expander(f"View {len(sources)} source reference{'s' if len(sources) > 1 else ''}"):
                for idx, src in enumerate(sources, 1):
                    doc = src["document"]
                    page = doc.metadata.get("page", "?")
                    score = src["score"]
                    st.markdown(
                        f'<div class="ref-card">'
                        f'<div class="ref-label">Reference {idx} — Page {page} &nbsp;·&nbsp; Score {score:.4f}</div>'
                        f'{doc.page_content}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
