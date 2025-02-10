"""
data_loaders.py

This module handles loading and preprocessing textual data from various sources.
It includes functions for processing all PDF files in a folder and reading
labelled texts from CSV files.
"""

import os
import pandas as pd
from preprocessing import process_pdf, preprocess_text

def process_folder(folder_path):
    """
    Process all PDFs in the specified folder and return a list of preprocessed texts and filenames.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing PDF files.

    Returns
    -------
    list of str
        Preprocessed texts extracted from each PDF.
    list of str
        Filenames corresponding to each processed PDF.
    """
    preprocessed_texts = []
    filenames = []
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return [], []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            preprocessed_text = process_pdf(pdf_path)
            if preprocessed_text:
                preprocessed_texts.append(preprocessed_text)
                filenames.append(filename)
            else:
                print(f"Skipping {filename} due to no extracted text.")
    if not preprocessed_texts:
        print("No texts were processed from PDFs.")
    return preprocessed_texts, filenames

def process_csv_files(csv_file_class0, csv_file_class1, text_column='text'):
    """
    Load texts from two CSV files representing two classes (0 and 1).
    Preprocess each text and create label arrays.

    Parameters
    ----------
    csv_file_class0 : str
        Path to the CSV file for class 0.
    csv_file_class1 : str
        Path to the CSV file for class 1.
    text_column : str, optional
        Column name containing text data, by default 'text'.

    Returns
    -------
    list of str
        Preprocessed texts from both CSV files.
    list of int
        Labels corresponding to each text (0 or 1).
    """
    preprocessed_texts = []
    labels = []

    #Non-syllabus data (class 0)
    df_class0 = pd.read_csv(csv_file_class0)
    if text_column not in df_class0.columns:
        print(f"Column '{text_column}' not found in {csv_file_class0}. Columns: {df_class0.columns.tolist()}")
        return preprocessed_texts, labels
    for text in df_class0[text_column].astype(str).tolist():
        preprocessed_texts.append(preprocess_text(text))
        labels.append(0)

    #Syllabus data (class 1)
    df_class1 = pd.read_csv(csv_file_class1)
    if text_column not in df_class1.columns:
        print(f"Column '{text_column}' not found in {csv_file_class1}. Columns: {df_class1.columns.tolist()}")
        return preprocessed_texts, labels
    for text in df_class1[text_column].astype(str).tolist():
        preprocessed_texts.append(preprocess_text(text))
        labels.append(1)

    return preprocessed_texts, labels
