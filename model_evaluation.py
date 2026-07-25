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


# =========================================================
# STEP 2: EVALUATE EACH MODEL
# STEP 6: PRODUCE METRICS
# STEP 7: PRODUCE CONFUSION MATRIX
# Purpose:
# - Reusable evaluator for one model and one split
# - Produces predictions, metrics, and confusion matrix
# =========================================================



def get_trained_models(file_path="labeled_tokens.csv"):
    # =========================================================
    # STEP 1: RECEIVE TRAINED MODELS
    # Purpose:
    # - Load trained models from model_training
    # - Unpack vectorizer and split payload
    # =========================================================
    trained_output = train_models(file_path)
    models = trained_output["models"]
    fitted_vectorizer = trained_output["vectorizer"]

    _, _, _ = trained_output["splits"]["train"]
    x_val_local, y_val_local, _ = trained_output["splits"]["validation"]
    x_test_local, y_test_local, _ = trained_output["splits"]["test"]

    # =========================================================
    # STEP 3: COMPARE VALIDATION RESULTS
    # Purpose:
    # - Store validation results for all models
    # =========================================================
    validation_results = {}

    # =========================================================
    # STEP 2: EVALUATE EACH MODEL
    # Purpose:
    # - Run every trained model on validation split
    # =========================================================
    for name, model in models.items():
        predictions, accuracy = evaluate_model(
            name=f"{name} - validation",
            model=model,
            x_data=x_val,
            y_data=y_val,
        )
        validation_results[name] = {
            "accuracy": accuracy,
            "predictions": predictions,
        }
    
    # =========================================================
    # STEP 4: CHOOSE BEST MODEL
    # Purpose:
    # - Select model using validation performance
    # =========================================================
    best_model_name = max(validation_results, key=lambda name: validation_results[name]["accuracy"])
    best_model = models[best_model_name]

    # =========================================================
    # STEP 5: EVALUATE ON TEST SET
    # Purpose:
    # - Run selected best model once on test split
    # =========================================================
    test_predictions, test_accuracy = evaluate_model(
        name=f"{best_model_name} - final test",
        model=best_model,
        x_data=x_test,
        y_data=y_test,
    )

    
    predictions = model.predict(x_data)
    accuracy = accuracy_score(y_data, predictions)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_data, predictions, labels=["ENG", "FIL", "CS", "OTH"], zero_division=0))
    # add confusion matrix
    cm = confusion_matrix(y_data, predictions, labels=["ENG", "FIL", "CS", "OTH"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ENG", "FIL", "CS", "OTH"])
    disp.plot(cmap="Blues")


# =========================================================
# STEP 8: ANALYZE MISCLASSIFIED WORDS
# Purpose:
# - Error analysis: inspect wrong predictions by word/sentence
# =========================================================
def display_error_analysis():
    #misclassified words
    #word
    #true label
    #predicted label
    #sentenceID
    pass


def display_misclassified_table():
    #display misclassified words
    #word
    #true label
    #predicted label
    #sentenceID
    pass


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

