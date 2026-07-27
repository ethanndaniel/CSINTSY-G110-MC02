"4 Labels"
ENG = "ENG"
FIL = "FIL"
CS = "CS"
OTH = "OTH"


import pandas as pd
from features_tokens import get_features, p_recognition
from sklearn.feature_extraction import DictVectorizer


def load_dataset(file_path="labeled_tokens.csv"):
   #read labeled_tokens
   df = pd.read_csv(file_path, keep_default_na=False, na_values=[])

   # VALIDATION
   #check required columns exist
   required_col = ["word_id", "sentence_id", "word", "label"]

   for col in required_col:
      if col not in df.columns:
         raise ValueError(f"Missing required column: {col}")

   #clean text fields
   df["label"] = df["label"].fillna("").astype(str).str.strip()
   df["word"] = df["word"].fillna("").astype(str).str.strip()

   #validate data quality -> no NaN and no blanks
   if df.isnull().any().any():
      raise ValueError("Dataset contains NaN values")
   if (df["word"] == "").any():
      raise ValueError("Dataset contains blank words")
   if (df["label"] == "").any():
      raise ValueError("Dataset contains blank labels")

   #validate labels -> ENG, FIL, CS, OTH only
   invalid = set(df["label"].unique()) - {ENG, FIL, CS, OTH}
   if invalid:
      raise ValueError(f"Dataset contains invalid labels: {invalid}")

   # FEATURE EXTRACTION
   # sort rows sentence_id then word_id
   df = df.sort_values(by=["sentence_id", "word_id"]).reset_index(drop=True)

   features_list = []
   labels_list = []
   metadata_list = []

   # get current features + previous features of same sentence
   for row in df.itertuples(index=True):
      index = row.Index
      current_word = row.word
      current_feat = get_features(current_word)

      # previous word
      if index > 0 and df.iloc[index - 1]["sentence_id"] == row.sentence_id:
         previous_word = df.iloc[index - 1]["word"]
      else:
         previous_word = None

      previous_feat = p_recognition(previous_word)

      # merge feature dicts
      combined_feat = {**current_feat, **previous_feat}

      features_list.append(combined_feat)
      labels_list.append(row.label)
      metadata_list.append({
         "row_index": index,
         "sentence_id": row.sentence_id,
         "word_id": row.word_id,
         "word": row.word,
      })

   #vectorize + transform into matrix
   vectorizer = DictVectorizer(sparse=True)
   X = vectorizer.fit_transform(features_list)
   y = labels_list
   metadata = metadata_list

   #final checks
   if len(features_list) != len(labels_list) or len(labels_list) != len(metadata_list):
      raise ValueError("Feature, label, and metadata lengths do not match")
   if len(vectorizer.feature_names_) == 0:
      raise ValueError("Vectorizer did not learn any features")

   return X, y, vectorizer, metadata
