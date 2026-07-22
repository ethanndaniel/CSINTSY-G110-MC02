"4 Labels"
ENG = "ENG"
FIL = "FIL"
CS = "CS"
OTH = "OTH"


import pandas as pd
from features_tokens import get_features

"read Sentence IDs 2180 to 2200"
"FORMAT: word_id, sentence_id, sentence, word, label"
df = pd.read_excel("labeled_tokens.xlsx")


"feature matrix; read through each row and get its features and label"
x = []
y = []

for _, row in df.iterrows():
    x.append(get_features(row["word"]))
    y.append(row["label"])


