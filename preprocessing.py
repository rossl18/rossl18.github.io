"""
preprocessing.py

This module provides functionality for text preprocessing, including
tokenization, stemming, and cleaning. It also offers a function to
process and extract text from PDF files.
"""

import re
import nltk
import os
from pdfminer.high_level import extract_text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Preprocess a given text by lowercasing, removing non-alphabetic characters,
    tokenizing, removing stopwords, and stemming.
    
    Parameters
    ----------
    text : str
        The raw text to preprocess.

    Returns
    -------
    str
        The preprocessed text as a single string.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def process_pdf(pdf_path):
    """
    Extract and preprocess text from a PDF file.
    
    Parameters
    ----------
    pdf_path : str
        File path to the PDF file.

    Returns
    -------
    str
        The preprocessed text extracted from the PDF.
    """
    raw_text = extract_text(pdf_path)
    if not raw_text:
        print(f"No text extracted from {pdf_path}.")
        return ""
    return preprocess_text(raw_text)
