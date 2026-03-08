import streamlit as st
import speech_ui
import handwriting_ui

st.set_page_config(
    page_title="Vocal Pen",
    layout="wide"
)

# Custom CSS for equal width tabs
st.markdown("""
<style>
button[data-baseweb="tab"] {
    flex-grow: 1;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Centered Heading
st.markdown(
    """
    <h1 style='text-align: center;'>🎙 Vocal Pen ✍</h1>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Speech Recognition", "Handwriting Output"])

with tab1:
    speech_ui.speech_interface()

with tab2:
    handwriting_ui.handwriting_interface()