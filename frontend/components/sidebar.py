import streamlit as st

def display_sidebar():
    """Displays the sidebar and returns the text input."""
    st.sidebar.header("Input Text")
    text_input = st.sidebar.text_area("Enter text to extract entities:", height=200)
    extract_button = st.sidebar.button("Extract Entities", use_container_width=True)
    return text_input, extract_button
