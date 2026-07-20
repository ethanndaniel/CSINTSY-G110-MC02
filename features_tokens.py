"Convert words to numerical features to be used in Decision Trees / Naive Bayes algorithms"



"add more possible features here"
def get_features(word):
    return {
        "length": len(word),
        "has_digit": any(c.isdigit() for c in word),
        "has_alpha": any(c.isalpha() for c in word),
        "is_upper": word.isupper(),
        "is_lower": word.islower()
    }
