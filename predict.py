"""
predict.py

Use the trained and saved models (plural) to make predictions on a single PDF file
applying the decision rule used during training.
"""

import pickle
import sys
from preprocessing import process_pdf

def main(pdf_path, model_threshold=1):
    # Load all models and vectorizer
    with open('model_files/all_models.pkl', 'rb') as f:
        all_models = pickle.load(f)
    with open('model_files/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Preprocess the PDF text
    preprocessed_text = process_pdf(pdf_path)

    # Transform using the previously fit vectorizer
    X = vectorizer.transform([preprocessed_text])

    # Get predictions from each model
    model_predictions = []
    for model_name, model in all_models.items():
        pred = model.predict(X)
        model_predictions.append(pred[0])

    # Apply the decision rule
    # The rule here is that if at least model_threshold models predict 1, final decision is 1
    model_predictions = list(model_predictions)
    num_ones = sum(p == 1 for p in model_predictions)
    final_decision = 1 if num_ones >= model_threshold else 0

    print(f"Individual model predictions: {model_predictions}")
    print(f"Final decision (threshold={model_threshold}): {final_decision}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_pdf_file")
        sys.exit(1)
    pdf_path = sys.argv[1]
    main(pdf_path)
