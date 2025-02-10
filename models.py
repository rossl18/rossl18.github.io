"""
models.py

This module focuses on model training, evaluation, and decision-making.
It includes functions to train multiple models, evaluate their performance,
and apply a final decision rule.
"""

import numpy as np
import pandas as pd
from better_profanity import profanity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def train_and_select_model(texts, labels, model_threshold=1):
    """
    Train multiple classifiers on a given dataset and apply a decision rule
    across all models' predictions to evaluate performance.

    Parameters
    ----------
    texts : list of str
        Preprocessed text samples.
    labels : list of int
        Labels corresponding to each text sample.
    model_threshold : int, optional
        The minimum number of models that need to predict "1" for the final
        decision to be "1". By default 1.

    Returns
    -------
    dict
        A dictionary containing:
        - 'best_model': The best performing model
        - 'vectorizer': The fitted TfidfVectorizer
        - 'final_decision_accuracy_test': The final decision accuracy on the test set
        - 'performance_df': DataFrame with model performance
        - 'X_test_texts': Test texts
        - 'y_test': Test labels
    """
    profanity.load_censor_words()

    if len(texts) != len(labels):
        print("Number of texts does not match number of labels.")
        return {}

    # Split data into train/test
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Vectorize on training data
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Support Vector Machine': SVC(probability=True, class_weight='balanced'),
        'Naïve Bayes': MultinomialNB()
    }

    performance_data = []
    model_test_predictions = []
    best_acc = 0
    best_model = None

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Check if this model is the 'best'
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model

        model_performance = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Macro Precision': report['macro avg']['precision'],
            'Macro Recall': report['macro avg']['recall'],
            'Macro F1-score': report['macro avg']['f1-score'],
            'Weighted Precision': report['weighted avg']['precision'],
            'Weighted Recall': report['weighted avg']['recall'],
            'Weighted F1-score': report['weighted avg']['f1-score'],
            'True Negatives': cm[0][0],
            'False Positives': cm[0][1],
            'False Negatives': cm[1][0],
            'True Positives': cm[1][1]
        }
        performance_data.append(model_performance)

        preds_test = model.predict(X_test)
        model_test_predictions.append(preds_test)

    performance_df = pd.DataFrame(performance_data)
    print("\nPerformance Metrics for Individual Models (Test Set):")
    print(performance_df[['Model', 'Accuracy']])

    # Convert model predictions to a numpy array for final decision rule
    model_test_predictions = np.array(model_test_predictions).T  

    # Apply decision rule on test set
    test_decisions = []
    for i in range(len(y_test)):
        preds = model_test_predictions[i]
        num_ones = np.sum(preds == 1)
        final_decision = 1 if num_ones >= model_threshold else 0
        test_decisions.append(final_decision)

    # Compute final decision accuracy on the test set
    final_decision_accuracy_test = accuracy_score(y_test, test_decisions)
    print(f"\nFinal Decision Rule Accuracy on Test Set: {final_decision_accuracy_test * 100:.2f}%")

    # Print a sample of predictions to check
    sample_size = min(10, len(y_test))
    print("\nSample Final Decisions vs. Actual Labels (Test Set Only):")
    for i in range(sample_size):
        print(f"Test Sample {i+1}: Model Predictions={model_test_predictions[i]}, "
              f"Final Decision={test_decisions[i]}, Actual={y_test[i]}")

    return {
        'all_models': models,
        'best_model': best_model,
        'vectorizer': vectorizer,
        'final_decision_accuracy_test': final_decision_accuracy_test,
        'performance_df': performance_df,
        'X_test_texts': X_test_texts,
        'y_test': y_test
    }
