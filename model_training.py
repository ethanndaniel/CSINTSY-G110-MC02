"""Train the model for the given dataset"""

from collections import Counter

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from tokens import load_dataset
import pickle

RANDOM_STATE = 67  # consistent shuffling

"""Training utilities."""

# Show class balance in a split.
def print_split_distribution(name, labels):
    print(f"{name} distribution: {dict(Counter(labels))}")

# Main training pipeline used by both wrappers.
def train_models(file_path="labeled_tokens.csv"):
    # Load, split, train, and return trained models with split data.

    x_data, y_data, fitted_vectorizer, metadata_data = load_dataset(file_path)

    # Split 1: training (70%) and temporary holdout (30%).
    x_train, x_temp, y_train_local, y_temp_local, metadata_train_local, metadata_temp_local = train_test_split(
        x_data,
        y_data,
        metadata_data,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    # Split 2: validation (15%) and test (15%) from holdout.
    x_val_local, x_test_local, y_val_local, y_test_local, metadata_val_local, metadata_test_local = train_test_split(
        x_temp,
        y_temp_local,
        metadata_temp_local,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    print_split_distribution("Training", y_train_local)
    print_split_distribution("Validation", y_val_local)
    print_split_distribution("Test", y_test_local)
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Decision Tree": tree.DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

    for model in models.values():
        model.fit(x_train, y_train_local)

    with open('training_model.pkl', 'wb') as model_file:
        pickle.dump(model[0], model_file)

    return {
        "models": models,
        "vectorizer": fitted_vectorizer,
        "splits": {
            "train": (x_train, y_train_local, metadata_train_local),
            "validation": (x_val_local, y_val_local, metadata_val_local),
            "test": (x_test_local, y_test_local, metadata_test_local),
        },
    }

if __name__ == "__main__":
    train_models()



