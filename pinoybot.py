"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
import features_tokens
from typing import List

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)
    read_model = open('best_model.pkl', 'rb')
    training_model = pickle.load(read_model)

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)
    extract = features_tokens
    features = []
    for word in tokens:
        feature = extract.get_features(word)
        features.append(feature)
    

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)
    # Loading the vectorizer
    read_vectorizer = open('vectorizer.pkl', 'rb')
    vectorizer = pickle.load(read_vectorizer)

    # Vectorize the features
    vectorized_features = vectorizer.transform(features)

    # Prediction
    predicted = training_model.predict(vectorized_features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]
    tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags
    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.
    # return ['FIL' for i in tokens]

if __name__ == "__main__":
    # Example usage
    example_tokens_1 = ["I", "love", "programming", "."]
    example_tokens_2 = ["Masaya", "akong", "kumain", "."]
    example_tokens_3 = ["Nag-download", "ako", "ng", "file", "."]
    example_tokens_4 = ["Philippines", "is", "mahal", "."]
    print("Tokens:", example_tokens_1)
    tags1 = tag_language(example_tokens_1)
    print("Tags:", tags1)
    print("Tokens:", example_tokens_2)
    tags2 = tag_language(example_tokens_2)
    print("Tags:", tags2)
    print("Tokens:", example_tokens_3)
    tags3 = tag_language(example_tokens_3)
    print("Tags:", tags3)
    print("Tokens:", example_tokens_4)
    tags4 = tag_language(example_tokens_4)
    print("Tags:", tags4)
    
