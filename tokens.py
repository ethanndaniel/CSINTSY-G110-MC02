"4 Labels"
ENG = "ENG"
FIL = "FIL"
CS = "CS"
OTH = "OTH"


import pandas as pd

"read and filter dataset with Sentence IDs 2180 to 2200 only"
df = pd.read_excel("raw_tokens.xlsx")
df = df[(df["sentence_id"] >= 2180) & (df["sentence_id"] <= 2200)]
