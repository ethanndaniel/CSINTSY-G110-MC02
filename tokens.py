"4 Labels"
ENG = "ENG"
FIL = "FIL"
CS = "CS"
OTH = "OTH"


import pandas as pd
from features_tokens import get_features, p_recognition
from sklearn.feature_extraction import DictVectorizer
"== NOTE copy paste this in vscode terminal to install sklearn: 'pip install scikit-learn' =="


"read Sentence IDs 2180 to 2200"
"FORMAT: word_id, sentence_id, sentence, word, label"
df = pd.read_excel("labeled_tokens.xlsx")


"read through each row and get its features and label"

def load_dataset():
 x = []
 y = []
 for _, row in df.iterrows():
    x.append(get_features(row["word"]))
    y.append(row["label"])
 "convert to dicts using sklearn's DictVectorizer"
 vectorizer = DictVectorizer()
 X = vectorizer.fit_transform(x)
 return X, y

# Implement the algo specifically for index lookback (Can combine algo w/original implementation)
def load_dataset_model():
   df_sorted = df.sort_values(["sentence_id", "word_id"]).reset_index(drop=True)
   x = []
   y = []
   for i in range(len(df_sorted)):
    row = df_sorted.iloc[i]
    word1, label1 = row["word"], row["label"]
    sentence_id = row["sentence_id"]

    if i - 1 >= 0 and df_sorted.iloc[i - 1]["sentence_id"] == sentence_id:
       word2, label2 = df_sorted.iloc[i - 1]["word"], df_sorted.iloc[i - 1]["label"]
    else:
        word2, label2 = "", None

    if i - 2 >= 0 and df_sorted.iloc[i - 2]["sentence_id"] == sentence_id:
     word3, label3 = df_sorted.iloc[i - 2]["word"], df_sorted.iloc[i - 2]["label"]
    else:
     word3, label3 = "", None

    feats = get_features(word1)
    feats.update(p_recognition(word3, word2, word1, label3, label2, label1))
    x.append(feats)
    y.append(label1)

    vectorizer = DictVectorizer()
   X = vectorizer.fit_transform(x)
   return X, y


"print feature matrix"
feature_list = []
for _, row in df.iterrows():
    features = get_features(row["word"])
    #print(f'Word: {row["word"]}')
    #print(features)
    #print()
    features["word"] = row ["word"]
    feature_list.append(features)
"==NOTE terminal might not show all words but it still works; use head()/tail() to verify it=="

