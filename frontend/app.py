import streamlit as st
import requests
import pandas as pd

# --- MOCK DATA FOR TESTING ---
# This data simulates a response from the backend API.
# It's useful for frontend development without a running backend.
MOCK_DATA = {
    "tokens": [
        "The", "European", "Commission", "said", "on", "Thursday", "that", "a", "group",
        "of", "seven", "countries", "had", "agreed", "to", "co-ordinate", "their",
        "economic", "and", "monetary", "policies", "."
    ],
    "labels": [
        "O", "B-ORG", "I-ORG", "O", "O", "B-MISC", "O", "O", "O", "O", "O",
        "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "O"
    ]
}

# --- CONFIGURATION ---
# Define colors for each entity type.
ENTITY_COLORS = {
    "PER": "#1f77b4",  # Muted Blue
    "ORG": "#2ca02c",  # Cooked Asparagus Green
    "LOC": "#ff7f0e",  # Safety Orange
    "MISC": "#9467bd", # Muted Purple
    "O": "transparent" # Default, no color
}
# API
API_URL = "http://backend:8000/predict"

# --- HELPER FUNCTIONS ---

def render_ner(tokens, labels):
    """
    Renders NER tags as highlighted text using HTML.
    Each entity is wrapped in a span with a specific background color and style.
    """
    # Start with a container div
    html_content = "<div style='line-height: 2.5; padding: 1em; border: 1px solid #ddd; border-radius: 5px;'>"
    current_entity = None
    entity_text = ""

    # A more robust way to handle B- and I- tags
    for i, (token, label) in enumerate(zip(tokens, labels)):
        entity_type = label.split('-')[-1] if label != 'O' else 'O'
        
        # Beginning of a new entity span
        if label.startswith('B-'):
            # If there was a previous entity, close its span
            if current_entity is not None:
                color = ENTITY_COLORS.get(current_entity, "#ccc")
                html_content += (
                    f'<span style="background-color: {color}; color: white; '
                    f'padding: 0.3em 0.6em; margin: 0 0.25em; border-radius: 0.3em; '
                    f'font-weight: bold;">{entity_text.strip()} '
                    f'<span style="font-size: 0.8em; font-weight: normal; '
                    f'opacity: 0.7;">{current_entity}</span></span>'
                )
            # Start new entity
            current_entity = entity_type
            entity_text = token + " "
        # Continuation of the current entity
        elif label.startswith('I-') and current_entity == entity_type:
            entity_text += token + " "
        # End of an entity or a non-entity token
        else:
            # If an entity was just being tracked, close it
            if current_entity is not None:
                color = ENTITY_COLORS.get(current_entity, "#ccc")
                html_content += (
                    f'<span style="background-color: {color}; color: white; '
                    f'padding: 0.3em 0.6em; margin: 0 0.25em; border-radius: 0.3em; '
                    f'font-weight: bold;">{entity_text.strip()} '
                    f'<span style="font-size: 0.8em; font-weight: normal; '
                    f'opacity: 0.7;">{current_entity}</span></span>'
                )
            # Reset for the next token
            entity_text = ""
            current_entity = None
            # Add the non-entity token
            html_content += token + " "

    # Handle any entity left at the very end of the sentence
    if current_entity is not None and entity_text:
        color = ENTITY_COLORS.get(current_entity, "#ccc")
        html_content += (
            f'<span style="background-color: {color}; color: white; '
            f'padding: 0.3em 0.6em; margin: 0 0.25em; border-radius: 0.3em; '
            f'font-weight: bold;">{entity_text.strip()} '
            f'<span style="font-size: 0.8em; font-weight: normal; '
            f'opacity: 0.7;">{current_entity}</span></span>'
        )

    html_content += "</div>"
    return html_content


def call_api(text):
    """
    Calls the backend FastAPI to get NER predictions.
    Handles potential connection errors.
    """
    try:
        response = requests.post(API_URL, json={"text": text}, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend service: {e}")
        st.info("Displaying mock data instead.")
        return None

# --- UI LAYOUT ---

def main():
    """
    Main function to run the Streamlit application.
    """
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="NER with DeBERTa",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Transformer-based NLP Application")
        st.markdown("---")
        st.header("Project Information")
        st.info(
            """
            **Project:** Named Entity Recognition (NER)
            **Student:** Lê Trung Thành Đạt
            **Dataset:** CoNLL-2003
            **Model:** DeBERTa
            """
        )
        st.markdown("---")

    # --- MAIN CONTENT ---
    st.title("Named Entity Recognition (NER)")
    st.markdown(
        "Enter English text below to identify and classify named entities like "
        "persons, organizations, and locations."
    )

    # --- TEXT INPUT AREA ---
    user_input = st.text_area(
        "Enter your text here:",
        "The European Commission said on Thursday it is ready to work with the new government in Baghdad.",
        height=150,
        key="text_input"
    )

    # --- EXTRACT BUTTON ---
    if st.button("Extract Entities", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter some text to extract entities.")
        else:
            # --- API CALL AND PROCESSING ---
            with st.spinner("Extracting entities..."):
                result = call_api(user_input)

                # If API call fails or returns no data, use mock data for demo
                if not result:
                    result = MOCK_DATA
                    st.sidebar.warning("Using Mock Data for Demonstration")

                tokens = result.get("tokens", [])
                labels = result.get("labels", [])

                if not tokens or not labels:
                    st.error("Received empty or invalid data from the backend.")
                else:

                    # --- DISPLAY RESULTS ---
                    st.success("Extraction Complete!")

                    col1, col2 = st.columns([3, 2])

                    with col1:
                        # 1. Highlighted Text
                        st.subheader("Highlighted Entities")
                        highlighted_text = render_ner(tokens, labels)
                        st.markdown(highlighted_text, unsafe_allow_html=True)

                    with col2:
                        # 2. DataFrame
                        st.subheader("Detailed Results")
                        df = pd.DataFrame({"Token": tokens, "Label": labels})
                        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()

