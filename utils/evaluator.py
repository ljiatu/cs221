"""
Result evaluation the result with ROC AUC.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.label import LABELS


def evaluate(prediction_file_path: str, true_label_file_path: str):
    prediction = pd.read_csv(prediction_file_path)
    true_labels = pd.read_csv(true_label_file_path)

    loss = []
    for i, j in enumerate(LABELS):
        filtered_labels = np.array([i for i in true_labels[j] if i != -1])
        preds = np.array([i for i in prediction[j]])
        roc_auc = roc_auc_score(filtered_labels, preds)
        print(f'ROC AUC for {LABELS[i]}: {roc_auc}')
        loss.append(roc_auc)

    print(f'Mean ROC AUC: {np.mean(loss)}')
