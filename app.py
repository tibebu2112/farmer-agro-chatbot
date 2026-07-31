"""
AgroQA Farmer Chatbot - Streamlit App
Deploy on https://share.streamlit.io
"""

from __future__ import annotations

import pickle
from pathlib import Path

import streamlit as st

from chatbot_engine import AgroChatbot
from voice_utils import transcribe_audio_file

# ---------------------------------------------------------------------------
# Page config - simple, farmer-friendly UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Farmer Agro Advisor",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4ade80;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #e5e7eb;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background-color: #ffffff !important;
        color: #14532d !important;
        border: 2px solid #16a34a;
        border-left: 8px solid #15803d;
        padding: 1.4rem 1.6rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: 600;
        line-height: 1.7;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    .stButton > button[kind="primary"] {
        background-color: #15803d !important;
        color: #ffffff !important;
        border: 2px solid #166534 !important;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.75rem 1.4rem;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #166534 !important;
        border-color: #14532d !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
CSV_CANDIDATES = [
    BASE_DIR / "AgroQA Dataset.csv",
    BASE_DIR / "AgroQA_Dataset.csv",
    BASE_DIR / "data" / "AgroQA Dataset.csv",
]
MODEL_PATH = BASE_DIR / "models" / "agro_chatbot.pkl"


def _find_csv_path() -> Path | None:
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner="Loading farming knowledge base...")
def load_chatbot() -> AgroChatbot:
    if MODEL_PATH.exists():
        return AgroChatbot.load(MODEL_PATH)

    csv_path = _find_csv_path()
    if csv_path is None:
        raise FileNotFoundError(
            "Dataset not found. Upload 'AgroQA Dataset.csv' to your GitHub repo root."
        )
    bot = AgroChatbot(csv_path)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bot.save(MODEL_PATH)
    return bot


def main() -> None:
    st.markdown('<p class="main-title">🌾 Farmer Agro Advisor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Ask by voice or text — get farming advice in simple language.</p>',
        unsafe_allow_html=True,
    )

    try:
        bot = load_chatbot()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    stats = bot.stats

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        crop_filter = st.selectbox(
            "Select your crop (optional)",
            options=bot.crop_options,
            format_func=lambda x: "All crops" if x == "all" else x.title(),
            index=0,
        )
        st.divider()
        st.subheader("About this app")
        st.write(f"**{stats['total_qa_pairs']:,}** farming Q&A pairs loaded.")
        for crop, count in sorted(stats["crops"].items()):
            st.caption(f"• {crop.title()}: {count:,} questions")
        st.divider()
        st.info(
            "Tip: Speak clearly in English. Example: "
            '"How can I control cassava mosaic disease?"'
        )

    tab_voice, tab_text = st.tabs(["🎤 Voice Question", "⌨️ Text Question"])

    question_text = ""
    transcribed = False

    with tab_voice:
        st.subheader("Record your question")
        st.write("Press the microphone button below, speak your question, then stop recording.")
        audio_bytes = st.audio_input("Tap to record your farming question")

        if audio_bytes is not None:
            with st.spinner("Listening and understanding your question..."):
                text, err = transcribe_audio_file(audio_bytes.getvalue())
            if err:
                st.warning(err)
            elif text:
                question_text = text
                transcribed = True
                st.success(f'I heard: **"{text}"**')

    with tab_text:
        st.subheader("Type your question")
        typed = st.text_area(
            "Write your farming question here",
            placeholder="Example: When should I harvest my beans?",
            height=100,
            label_visibility="collapsed",
        )
        if typed.strip():
            question_text = typed.strip()

    ask = st.button("Get Farming Advice", type="primary", use_container_width=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    if ask and question_text:
        answer, matched_crop, score, matched_q = bot.get_answer(
            question_text,
            crop_filter=crop_filter,
        )

        st.session_state.history.insert(
            0,
            {
                "question": question_text,
                "answer": answer,
                "crop": matched_crop,
            },
        )

        st.markdown("---")
        st.subheader("Answer")
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True,
        )

        meta_cols = st.columns(3)
        with meta_cols[0]:
            if matched_crop:
                st.caption(f"Crop: **{matched_crop.title()}**")
        with meta_cols[1]:
            st.caption(f"Match confidence: **{score:.0%}**")
        with meta_cols[2]:
            st.caption("Input: **Voice**" if transcribed else "Input: **Text**")

        if matched_q:
            with st.expander("Similar question in our database"):
                st.write(matched_q)

    elif ask:
        st.warning("Please record or type a question first.")

    if st.session_state.history:
        st.markdown("---")
        st.subheader("Recent questions")
        for item in st.session_state.history[:5]:
            with st.expander(f"Q: {item['question'][:80]}..."):
                st.write(item["answer"])
                if item["crop"]:
                    st.caption(f"Crop: {item['crop'].title()}")


if __name__ == "__main__":
    main()
