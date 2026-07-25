""" model evaluation functions """
import matplotlib.pyplot as mp
from model_training import train_models
from sklearn import tree
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

LABELS = ["ENG", "FIL", "CS", "OTH"]

def evaulate_validation(models, x_val, y_val, labels):

def best_model(validation_results, metric="accuracy"):

def evaluate_best(best_name, models, x_test, y_test, labels):

def evaluate_trained_models(path="labeled_tokens.csv"):
    output = train_models(path)

    #unpack models, vectorizer split, test split
    models = output["models"]
    vectorizer = output["vectorizer"]

    x_val, y_val, metadata_val = output["splits"]["validation"]
    x_test, y_test, metadata_test = output["splits"]["test"]

    # evaluate all on validation set
    validation_results = evaluate_validation(models, x_val, y_val, LABELS)

    # best model = macro F1-score
    best_name, best_metrics = best_model(validation_results, "f1_macro")

    #evaluate best model on test set
    test_predictions, test_metrics, test_confusion_matrix = evaluate_best(best_name, models, x_test, y_test, LABELS)

    # summarize incorrect predicted tokens
    misclassified_tokens, misclassified_summary = analyze_misclassified(y_test, test_predictions, metadata_test)

    #return for report
    return{
        "models": models,
        "vectorizer": vectorizer,
        "validation_results": validation_results,
        "best_model": {
            "name": best_name,
            "metrics": best_metrics,
        },
        "test_results": {
            "predictions": test_predictions,
            "metrics": test_metrics,
            "confusion_matrix": test_confusion_matrix,
        },
        "misclassified": {
            "tokens": misclassified_tokens,
            "summary": misclassified_summary,
        },
        "validation_metadata": metadata_val,
        "test_metadata": metadata_test,
    }

def plot_decision_tree(model, vectorizer, output_path="decision_tree.png"):
    mp.figure(figsize=(30, 30))
    tree.plot_tree(
        model,
        feature_names=vectorizer.get_feature_names_out(),
        class_names=model.classes_,
        filled=True,
        max_depth=3,
    )
    mp.title("Decision Tree - First Three Levels")
    mp.savefig(output_path, dpi=150, bbox_inches="tight")
    mp.close()

    
    decision_tree_model = models["Decision Tree"]
    plot_decision_tree(decision_tree_model, fitted_vectorizer)

