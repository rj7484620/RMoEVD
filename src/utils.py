import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def calculate_max_f1_threshold(df, col='pred_prob'):
    thresholds = np.linspace(0, 1, 100)
    max_f1 = 0
    best_threshold = 0
    max_precision = 0
    max_recall = 0
    max_accuracy = 0
    
    for threshold in thresholds:
        predictions = df[col].apply(lambda x: 1 if x >= threshold else 0)
        f1 = f1_score(df['target'], predictions)
        
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold
    
    predictions = df[col].apply(lambda x: 1 if x >= best_threshold else 0)       
    f1 = f1_score(df['target'], predictions)        
    precision = precision_score(df['target'], predictions)
    recall = recall_score(df['target'], predictions)
    accuracy = accuracy_score(df['target'], predictions)
    
    return best_threshold, f1, precision, recall, accuracy
