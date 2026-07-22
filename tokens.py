"4 Labels"
ENG = "ENG"
FIL = "FIL"
CS = "CS"
OTH = "OTH"


import pandas as pd
from features_tokens import get_features
from sklearn.feature_extraction import DictVectorizer
"== NOTE copy paste this in vscode terminal to install sklearn: 'pip install scikit-learn' =="


"read Sentence IDs 2180 to 2200"
"FORMAT: word_id, sentence_id, sentence, word, label"
df = pd.read_excel("labeled_tokens.xlsx")


"read through each row and get its features and label"
x = []
y = []
for _, row in df.iterrows():
    x.append(get_features(row["word"]))
    y.append(row["label"])


"convert to dicts using sklearn's DictVectorizer"
vectorizer = DictVectorizer()
X = vectorizer.fit_transform(x)


"print feature matrix"
for _, row in df.iterrows():
    features = get_features(row["word"])
    print(f'Word: {row["word"]}')
    print(features)
    print()
"==NOTE terminal might not show all words but it still works; use head()/tail() to verify it=="