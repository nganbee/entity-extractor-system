import streamlit as st

def display_header():
    """Displays the application header."""
    st.title("✒️ Entity Extractor System")
    st.markdown(
        """
        This application uses a DeBERTa model to extract named entities from text.
        Enter some text in the sidebar to see the model in action.
        """
    )
