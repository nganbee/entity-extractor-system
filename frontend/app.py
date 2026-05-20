import streamlit as st
import requests
import pandas as pd
# --- CONFIGURATION ---
# Define colors for each entity type for visualization.
ENTITY_COLORS = {
    "PER": "#1f77b4",    # Muted Blue
    "ORG": "#2ca02c",    # Cooked Asparagus Green
    "LOC": "#ff7f0e",    # Safety Orange
    "ANIM": "#d62728",   # Brick Red
    "BIO": "#9467bd",    # Muted Purple
    "CEL": "#8c564b",    # Chestnut Brown
    "DIS": "#e377c2",    # Pink
    "EVE": "#7f7f7f",    # Middle Gray
    "FOOD": "#bcbd22",   # Curry Yellow-Green
    "INST": "#17becf",   # Blue-Teal
    "MEDIA": "#aec7e8",  # Light Blue
    "MYTH": "#ffbb78",   # Light Orange
    "PLANT": "#98df8a",  # Light Green
    "TIME": "#ff9896",   # Light Red
    "VEHI": "#c5b0d5",   # Light Purple
    "O": "transparent"   # Default, no color
}
# API endpoint for the backend
API_URL = "http://backend:8000/predict"

# --- HELPER FUNCTIONS ---

def render_ner_from_entities(text, entities):
    """
    Renders NER tags as highlighted text using HTML from a list of entity dictionaries.
    Each entity is wrapped in a span with a specific background color and style.
    """
    # Sort entities by their start index to process them in order
    entities = sorted(entities, key=lambda x: x['start'])
    
    html_content = "<div style='line-height: 2.5; padding: 1em; border: 1px solid #ddd; border-radius: 5px;'>"
    last_end = 0
    
    for entity in entities:
        start, end = entity['start'], entity['end']
        entity_group = entity['entity_group']
        word = entity['word']
        
        # Add the text between the last entity and this one
        html_content += text[last_end:start]
        
        # Add the highlighted entity
        color = ENTITY_COLORS.get(entity_group, "#ccc")
        html_content += (
            f'<span style="background-color: {color}; color: white; '
            f'padding: 0.3em 0.6em; margin: 0 0.25em; border-radius: 0.3em; '
            f'font-weight: bold;">{word} '
            f'<span style="font-size: 0.8em; font-weight: normal; '
            f'opacity: 0.7;">{entity_group}</span></span>'
        )
        
        last_end = end
        
    # Add any remaining text after the last entity
    html_content += text[last_end:]
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
        st.info("Please ensure the backend is running.")
        return None

# --- UI LAYOUT ---

def main():
    """
    Main function to run the Streamlit application.
    """
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="Entity Extractor System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Entity Extractor System")
        st.markdown("---")
        st.header("Project Information")
        st.info(
            """
            **Project:** Named Entity Recognition (NER)
            **Student:** Lê Trung Thành Đạt
            **Model:** `imbee510/finetuned_ner_xlm_roberta`
            """
        )
        st.markdown("---")
        st.header("Entity Legend")
        for entity, color in ENTITY_COLORS.items():
            if entity != "O":
                st.markdown(f'<span style="background-color:{color}; color:white; padding:2px 8px; border-radius:5px; margin-right:5px;">{entity}</span>', unsafe_allow_html=True)


    # --- MAIN CONTENT ---
    st.title("Named Entity Recognition (NER)")
    st.markdown(
        "Enter English text below to identify and classify named entities. "
        "The system uses a fine-tuned XLM-RoBERTa model to extract entities from the text."
    )

    # --- TEXT INPUT AREA ---
    user_input = st.text_area(
        "Enter your text here:",
        "The European Commission, led by Ursula von der Leyen, announced new regulations from their headquarters in Brussels. Meanwhile, a rare Sumatran tiger was spotted near the Amazon River, and scientists are studying the effects of the Spanish Flu.",
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
                entities = call_api(user_input)

                if entities is None:
                    # Error message is already shown by call_api
                    pass
                elif not entities:
                    st.success("Extraction Complete!")
                    st.info("No entities were found in the provided text.")
                else:
                    # --- DISPLAY RESULTS ---
                    st.success("Extraction Complete!")

                    col1, col2 = st.columns([3, 2])

                    with col1:
                        # 1. Highlighted Text
                        st.subheader("Highlighted Entities")
                        highlighted_text = render_ner_from_entities(user_input, entities)
                        st.markdown(highlighted_text, unsafe_allow_html=True)

                    with col2:
                        # 2. DataFrame
                        st.subheader("Detailed Results")
                        # Create a more readable DataFrame from the entities
                        df_data = {
                            "Entity": [e['word'] for e in entities],
                            "Label": [e['entity_group'] for e in entities],
                            "Start": [e['start'] for e in entities],
                            "End": [e['end'] for e in entities]
                        }
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()

