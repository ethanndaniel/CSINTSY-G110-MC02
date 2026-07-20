"4 Labels"
ENG = "ENG"
FIL = "FIL"
CS = "CS"
OTH = "OTH"


import pandas as pd

"read Sentence IDs 2180 to 2200"
"FORMAT: word_id, sentence_id, sentence, word, label"
df = pd.read_excel("labeled_tokens.xlsx")
print(df.head())


