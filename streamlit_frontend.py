import streamlit as st
import speech_ui
import handwriting_ui

st.set_page_config(
    page_title="Speech → Handwriting Generator",
    layout="wide"
)

st.title("🎙 Speech → ✍ Handwriting Generator")

tab1, tab2 = st.tabs(["Speech Recognition", "Handwriting Output"])

with tab1:
    speech_ui.speech_interface()

with tab2:
    handwriting_ui.handwriting_interface()