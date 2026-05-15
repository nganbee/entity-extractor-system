# Logic chính của Backend
# Hàm dự đoán sử dụng model DeBERTa
def predict(text: str):
    # Dummy prediction
    print(f"Predicting entities for: {text}")
    return [("Person", "John Doe"), ("Location", "New York")]
