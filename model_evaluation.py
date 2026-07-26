""" model evaluation functions """

import matplotlib.pyplot as mp
from model_training import train_models
from sklearn import tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from collections import Counter

LABELS = ["ENG", "FIL", "CS", "OTH"]

# helper func to evaluate a model
# x = label encoded feature matrix, y = label encoded labels
def evaluate_model(name, model, x, y, labels):
    predict = model.predict(x)

    #compute metrics
    accuracy = accuracy_score(y, predict)
    precision = precision_score(y, predict, average="macro", zero_division=0)
    recall = recall_score(y, predict, average="macro", zero_division=0)
    f1 = f1_score(y, predict, average="macro", zero_division=0)

    # classification report
    report = classification_report(y, predict, labels=labels, zero_division=0)

    # confusion matrix
    matrix = confusion_matrix(y, predict, labels=labels)

    metrics = {
        "model_name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "classification_report": report,
    }

    #display
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-macro: {f1:.4f}")
    print("Classification Report:")
    print(report)

    return predict, metrics, matrix

# func to evaluate each models on validation set
def evaluate_validation(models, x_val, y_val, labels):
    validation_results = {}

    # evaluate each model on validation set
    for name, model in models.items():
        print(f"Evaluating {name} on validation set")
        predict, metrics, matrix = evaluate_model(name, model, x_val, y_val, labels)

        #results each model
        validation_results[name] = {
            "predictions": predict,
            "metrics": metrics,
            "confusion_matrix": matrix,
        }

    return validation_results

# func to determine best model based on validation metrics
def best_model(validation_results, metric="f1_macro"):
    if not validation_results:
        raise ValueError("No validation results provided. cannot determine the best model.")
    
    best_name = None
    best_metrics = None
    best_score = -1

    #find best model based on validated metrics
    for name, result in validation_results.items():
        metrics = result["metrics"]

        if metric not in metrics:
            raise ValueError(f"Metric '{metric}' not found in model metrics in '{name}'.")

        score = metrics[metric]

        if score > best_score: #when tie it chooses first model
            best_score = score
            best_name = name
            best_metrics = metrics

    print(f"Best model based on validation {metric}:")
    print(f"{best_name}")
    print(f"{metric}: {best_score:.4f}")

    return best_name, best_metrics

#note: if testing every model, it use the test set to make decision, which can bias the evaluation -> ONLY eval best model on test set
def evaluate_best(best_name, models, x_test, y_test, labels):
    # check if best model exists
    if best_name not in models:
        raise ValueError(f"Best model '{best_name}' not found")

    model = models[best_name]

    #eval best model on test set
    test_predictions, test_metrics, test_confusion_matrix = evaluate_model(name=f"{best_name} (Test)", model=model, x=x_test, y=y_test, labels=labels)

    return test_predictions, test_metrics, test_confusion_matrix

# func to analyze misclassified tokens and give reports
def analyze_misclassified(y_true, y_pred, metadata_test):
    misclassified_tokens = []
    misclassified_summary = {}

    #check if all inputs refer to the same number of samples
    if len(y_true) != len(y_pred) or len(y_true) != len(metadata_test):
        raise ValueError("True labels, predictions, and metadata must be the same")

    misclassified_tokens = []

    #comppare true and predicted labels for each test
    for true_label, pred_label, metadata in zip(y_true, y_pred, metadata_test):
        if true_label != pred_label:
            misclassified_tokens.append({
                "word": metadata["word"],
                "sentence_id": metadata["sentence_id"],
                "word_id": metadata["word_id"],
                "true_label": true_label,
                "predicted_label": pred_label,
            })

    #summarize for error analysis
    true_label_counts = Counter([token["true_label"] for token in misclassified_tokens])
    predicted_label_counts = Counter([token["predicted_label"] for token in misclassified_tokens])
    word_counts = Counter([token["word"] for token in misclassified_tokens])

    misclassified_summary = {
        "total_misclassified": len(misclassified_tokens),
        "true_label_counts": dict(true_label_counts),
        "predicted_label_counts": dict(predicted_label_counts),
        "most_frequently_misclassified_words": word_counts.most_common(10),
    }

    #display indivisual errors
    print("\nMisclassified Tokens:")

    if not misclassified_tokens:
        print("No misclassified tokens found")
    else:
        for token in misclassified_tokens:
            print(f"Word: {token['word']} | Sentence ID: {token['sentence_id']} | Word ID: {token['word_id']} | True Label: {token['true_label']} | Predicted Label: {token['predicted_label']}")

    #display short summary
    print("\nMisclassification Summary:")
    print(f"Total Misclassified Tokens: {misclassified_summary['total_misclassified']}")
    print(f"True Label Counts: {misclassified_summary['true_label_counts']}")
    print(f"Predicted Label Counts: {misclassified_summary['predicted_label_counts']}")

    return misclassified_tokens, misclassified_summary


# whole evaluation pipeline
def evaluate_trained_models(path="labeled_tokens.csv"):
    output = train_models(path)

    #1 unpack models, vectorizer split, test split
    models = output["models"]
    vectorizer = output["vectorizer"]

    # x = feature matrix, y = labels, metadata = orig words + ids
    x_val, y_val, metadata_val = output["splits"]["validation"] #validation data
    x_test, y_test, metadata_test = output["splits"]["test"] #test data

    #2 evaluate all on validation set
    validation_results = evaluate_validation(models, x_val, y_val, LABELS)

    # best model = macro F1-score
    best_name, best_metrics = best_model(validation_results, "f1_macro")

    #3 evaluate best model on test set
    test_predictions, test_metrics, test_confusion_matrix = evaluate_best(best_name, models, x_test, y_test, LABELS)

    #4  summarize incorrect predicted tokens
    misclassified_tokens, misclassified_summary = analyze_misclassified(y_test, test_predictions, metadata_test)

    #5 opt plot trained Decision Tree model
    if "Decision Tree" in models:
        plot_decision_tree(model=models["Decision Tree"],vectorizer=vectorizer)

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
        rounded=True,
        max_depth=3,
        fontsize=8,
    )

    mp.title("Decision Tree - First Three Levels")
    mp.savefig(output_path, dpi=150, bbox_inches="tight")
    mp.close()

# main to run
if __name__ == "__main__":
    results = evaluate_trained_models("labeled_tokens.csv")

    print("\nEvaluation completed.")
    print(  f"Selected model: "
            f"{results['best_model']['name']}")


