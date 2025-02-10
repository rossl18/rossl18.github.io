"""
train.py

This script orchestrates the training process. It:
- Loads data from CSV files and a PDF folder.
- Trains multiple models and retains all of them.
- Saves all models and the vectorizer for future inference with a decision rule.
"""

import pickle
import os
from data_loaders import process_csv_files, process_folder
from models import train_and_select_model

def main():
    # Ensure directory exists
    os.makedirs('model_files', exist_ok=True)

    # Adjust these paths as needed
    folder_path = 'C:/Users/rossl/syllabi'
    csv_file_class0 = 'C:/Users/rossl/nonsyllabi.csv'
    csv_file_class1 = 'C:/Users/rossl/syllabi.csv'

    # Load CSV data
    csv_texts, csv_labels = process_csv_files(
        csv_file_class0=csv_file_class0,
        csv_file_class1=csv_file_class1,
        text_column='Syllabus'  # Adjust if needed
    )

    # Load PDF data
    preprocessed_texts_pdf, filenames = process_folder(folder_path)

    # Provide labels for PDFs (ensure this matches number of PDFs)
    labels_pdf = [0, 0, 0, 0, 0, 1, 1, 1, 1]

    # Combine all texts and labels
    all_texts = preprocessed_texts_pdf + csv_texts
    labels = labels_pdf + csv_labels

    results = train_and_select_model(texts=all_texts, labels=labels, model_threshold=1)
    all_models = results['all_models']
    vectorizer = results['vectorizer']

    # Save all models and vectorizer
    with open('syllabus-classifier/model_files/all_models.pkl', 'wb') as f:
        pickle.dump(all_models, f)

    with open('syllabus-classifier/model_files/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("\nModel training complete. All models and vectorizer saved.")

if __name__ == "__main__":
    main()
