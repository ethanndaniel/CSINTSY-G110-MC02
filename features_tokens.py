import string


def get_features(word):
    return {
    "==counts=="
        "length": len(word),
        "alphabet_count": sum(c.isalpha() for c in word),
        "digit_count": sum(c.isdigit() for c in word),

    "==booleans=="
        "has_digit": any(c.isdigit() for c in word),
        "has_alphabet": any(c.isalpha() for c in word),
        "has_hyphen": "-" in word,

        "is_alphabet": word.isalpha(),
        "is_upper": word.isupper(),
        "is_lower": word.islower(),
        "is_capital": word[0].isupper(),

        "is_punctuation": all(c in string.punctuation for c in word),

    }
