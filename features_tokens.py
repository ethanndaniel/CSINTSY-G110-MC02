import string


# 1. General word features are being accounted
def get_features(word):
    return {
        "length": len(word),
        "alphabet_count": sum(c.isalpha() for c in word),
        "digit_count": sum(c.isdigit() for c in word),

        "has_digit": any(c.isdigit() for c in word),
        "has_alphabet": any(c.isalpha() for c in word),
        "has_hyphen": "-" in word,

        "is_alphabet": word.isalpha( ),
        "is_upper": word.isupper(),
        "is_lower": word.islower(),
        "is_capital": len(word) > 0 and word[0].isupper(),

        "is_punctuation": all(c in string.punctuation for c in word),

    }

# 2. Word features are being compared to previous word for pattern tracing
def p_recognition(word3,word2, word1, label3, label2, label1):
    return{
        # Count number of code-switches per sentence id (General count)
        "switch_count": sum(c.isalpha() for c in word2),

        # Count occurences of code-switches per sentence id
        "ENG_FIL": int(label2 == "ENG" and label1 == "FIL"),
        "ENG_OTH": int(label2 == "ENG" and label1 == "OTH"),
        "FIL_ENG": int(label2 == "FIL" and label1 == "ENG"),
        "FIL_OTH": int(label2 == "FIL" and label1 == "OTH"),
        "OTH_ENG": int(label2 == "OTH" and label1 == "ENG"),
        "OTH_FIL": int(label2 == "OTH" and label1 == "FIL"),

        # Count occurences of codeswitching after two consecutive languages
        "ENG_ENG_FIL": int(label3 == "ENG" and label2 == "ENG" and label1 == "FIL"),
        "ENG_ENG_OTH": int(label3 == "ENG" and label2 == "ENG" and label1 == "OTH"),
        "FIL_FIL_ENG": int(label3 == "FIL" and label2 == "FIL" and label1 == "ENG"),
        "FIL_FIL_OTH": int(label3 == "FIL" and label2 == "FIL" and label1 == "OTH"),
        "OTH_OTH_FIL": int(label3 == "OTH" and label2 == "OTH" and label1 == "FIL"),
        "OTH_OTH_ENG": int(label3 == "OTH" and label2 == "OTH" and label1 == "ENG"),
    }
