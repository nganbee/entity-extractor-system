from transformers import pipeline
import torch

class NERModel:
    """
    Encapsulates the fine-tuned XLM-RoBERTa model for the NER task.
    Uses 'aggregation_strategy' to automatically group sub-tokens.
    """
    def __init__(self):
        """
        Initializes and loads the model and tokenizer from the Hugging Face Hub.
        """
        repo_id = "imbee510/finetuned_ner_xlm_roberta"
        
        # Determine the device (GPU if available, otherwise CPU)
        device = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading model from '{repo_id}' on device: {'cuda' if device == 0 else 'cpu'}...")
        # Use a pipeline to simplify prediction and automatically handle token aggregation
        self.ner_pipeline = pipeline(
            "ner",
            model=repo_id,
            tokenizer=repo_id,
            aggregation_strategy="simple",  # Groups sub-tokens into complete words
            device=device
        )
        print("Model loaded successfully!")

    def predict(self, text: str):
        """
        Predicts named entities in a given text.

        Args:
            text (str): The input text for prediction.

        Returns:
            list: A list of found entities. Each entity is a dictionary with keys
                  like 'entity_group', 'score', 'word', and includes start/end character indices.
        """
        if not text or not text.strip():
            return []
            
        # The pipeline with simple aggregation returns a list of dicts
        # e.g., [{'entity_group': 'ORG', 'score': 0.99, 'word': 'HCMUS', 'start': 21, 'end': 26}]
        entities = self.ner_pipeline(text)
        return entities

# Example of how to use the class (for testing purposes)
if __name__ == '__main__':
    print("Running NERModel in standalone mode for testing...")
    ner_model = NERModel()
    sample_text = "I am a student in HCMUS and my friend is from University of Science."
    results = ner_model.predict(sample_text)
    print(f"\nPrediction results for: '{sample_text}'")
    print(results)
