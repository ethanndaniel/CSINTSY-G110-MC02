import string

# global constants
VOWELS = "aeiouAEIOU"
CONSONANTS = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"

# 1. General word features are being accounted
def get_features(word):
    safe_word = word if isinstance(word, str) else ""  # if word is None or empty -> empty string (for edge cases)

    alphabet_count = sum(c.isalpha() for c in safe_word)
    first_alphabet = next((c for c in safe_word if c.isalpha()), "")

    lower_word = safe_word.lower()
    has_repeated_char = False
    for i in range(len(lower_word) - 1):
        if lower_word[i] == lower_word[i + 1]:
            has_repeated_char = True
            break

    return {

        "length": len(safe_word),
        "alphabet_count": alphabet_count,
        "has_multiple_alphabets": alphabet_count > 1,
        "digit_count": sum(c.isdigit() for c in safe_word),
        "vowel_count": sum(c in VOWELS for c in safe_word),
        "consonant_count": sum(c in CONSONANTS for c in safe_word),

        "has_hyphen": "-" in safe_word,
        "has_apostrophe": "'" in safe_word,
        "has_alpha_digit_mix": any(c.isalpha() for c in safe_word) and any(c.isdigit() for c in safe_word),
        "has_repeated_char": has_repeated_char,

        "is_upper": safe_word.isupper(),
        "is_lower": safe_word.islower(),
        "is_punctuation": all(c in string.punctuation for c in safe_word) if safe_word else False, # empty should not get treated as punctuation = return false
        "is_single_alphabet": alphabet_count == 1,
        "is_capital": bool(first_alphabet) and first_alphabet.isupper(), #improved logic for single letter words


        "starts_with_vowel": safe_word[:1] in VOWELS,
        "ends_with_vowel": safe_word[-1:] in VOWELS,

        "first1char": lower_word[:1],
        "first2char": lower_word[:2],
        "first3char": lower_word[:3],

        "last1char": lower_word[-1:],
        "last2char": lower_word[-2:],
        "last3char": lower_word[-3:],

    }

# 2. Word features are being compared to previous word for pattern tracing
def p_recognition(word2): #word2 = previous word
    safe_word2 = word2 if isinstance(word2, str) else ""

    return{
        "prev_has_token": len(safe_word2) > 0,
        "prev_length": len(safe_word2),
        "prev_alphabet_count": sum(c.isalpha() for c in safe_word2),
        "prev_digit_count": sum(c.isdigit() for c in safe_word2),

        "prev_has_hyphen": "-" in safe_word2,
        "prev_is_upper": safe_word2.isupper(),
        "prev_is_lower": safe_word2.islower(),
        "prev_is_capital": safe_word2[:1].isupper(),
        "prev_is_punctuation": all(c in string.punctuation for c in safe_word2) if safe_word2 else False,

        "prev_ends_with_vowel": safe_word2[-1:] in VOWELS,
    }
