import streamlit as st

def display_results(entities):
    """Displays the extracted entities."""
    if entities:
        st.write("### Extracted Entities")
        for entity_type, entity_value in entities:
            st.info(f"**{entity_type}:** {entity_value}")
    else:
        st.info("No entities were extracted from the text.")

def display_mock_results():
    """Displays mock results for UI testing."""
    st.write("### Extracted Entities (Mock Data)")
    mock_entities = [
        ("PERSON", "John Doe"),
        ("LOCATION", "New York"),
        ("ORGANIZATION", "Acme Corp"),
        ("MISC", "DeBERTa")
    ]
    for entity_type, entity_value in mock_entities:
        st.success(f"**{entity_type}:** {entity_value}")
