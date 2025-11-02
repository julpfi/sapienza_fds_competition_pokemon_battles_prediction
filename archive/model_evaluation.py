from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_classification(y_true, y_pred, out_print=False):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    
    if out_print:
        print("Model evaluation:")
        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall,4))

    return results
