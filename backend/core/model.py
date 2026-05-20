from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class NERModel:
    """
    Encapsulates the fine-tuned XLM-RoBERTa model for the NER task.
    This version uses pure PyTorch for inference instead of the Hugging Face pipeline.
    """
    def __init__(self):
        """
        Initializes and loads the model and tokenizer directly.
        Sets up the device and puts the model in evaluation mode.
        """
        repo_id = "imbee510/finetuned_multiner_xlm_roberta"
        
        # Define the new label mapping
        label2id = {
            "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6,
            "B-ANIM": 7, "I-ANIM": 8, "B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12,
            "B-DIS": 13, "I-DIS": 14, "B-EVE": 15, "I-EVE": 16, "B-FOOD": 17, "I-FOOD": 18,
            "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23, "I-MYTH": 24,
            "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30
        }
        self.id2label = {v: k for k, v in label2id.items()}
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model and tokenizer from '{repo_id}' on device: {self.device}...")

        # Load tokenizer and model directly
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForTokenClassification.from_pretrained(repo_id)
        
        # Move model to the specified device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print("Model and tokenizer loaded successfully in eval mode!")

    def predict(self, text: str):
        """
        Predicts named entities in a given text using raw PyTorch inference.
        It processes tokens, aggregates sub-tokens, and formats the output
        to match the previous pipeline's structure.
        """
        if not text or not text.strip():
            return []

        # Tokenize the input text and get offsets
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512 # Ensure input is not too long
        )
        offset_mapping = inputs.pop("offset_mapping").squeeze().cpu().numpy()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get the most likely label for each token
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
        
        entities = []
        current_entity = None

        for i, (token_pred, offset) in enumerate(zip(predictions, offset_mapping)):
            # Ignore special tokens like <s> and </s> which have offset (0, 0)
            if offset[0] == offset[1]:
                continue

            label = self.id2label[token_pred]
            token_str = self.tokenizer.convert_ids_to_tokens([inputs["input_ids"][0, i].item()])[0]
            
            # Clean the special character from the token
            clean_token_str = token_str.replace(' ', '')

            if label.startswith("B-"):
                # If there was a previous entity, save it
                if current_entity:
                    entities.append(current_entity)
                
                # Start a new entity
                current_entity = {
                    'entity_group': label[2:],
                    'word': clean_token_str,
                    'start': offset[0],
                    'end': offset[1]
                }
            elif label.startswith("I-"):
                # If an entity is in progress and the label matches, extend it
                if current_entity and current_entity['entity_group'] == label[2:]:
                    current_entity['word'] += clean_token_str
                    current_entity['end'] = offset[1]
                else:
                    # If an I- tag appears without a B- tag, treat it as a B- tag
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'entity_group': label[2:],
                        'word': clean_token_str,
                        'start': offset[0],
                        'end': offset[1]
                    }
            else: # label is "O"
                # If an entity was being built, save it before moving on
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
        
        # Add the last entity if it exists
        if current_entity:
            entities.append(current_entity)
            
        return entities

# Example of how to use the class (for testing purposes)
if __name__ == '__main__':
    print("Running NERModel in standalone mode for testing...")
    ner_model = NERModel()
    sample_text = "I am a student in HCMUS and my friend is from University of Science."
    results = ner_model.predict(sample_text)
    print(f"\nPrediction results for: '{sample_text}'")
    print(results)
    # Expected output format: [{'entity_group': 'ORG', 'word': 'HCMUS', 'start': 21, 'end': 26}, {'entity_group': 'ORG', 'word': 'University of Science', 'start': 48, 'end': 69}]
