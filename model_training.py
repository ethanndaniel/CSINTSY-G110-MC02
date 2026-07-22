"""Train the model for the given dataset."""

import tokens
import pandas as pd
import numpy as np 
import matplotlib.pyplot as mp
#import feature_tokens

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from tokens import load_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.metrics import classification_report
#from feature_tokens import get_features

# Initialize Word Extraction 

list_words_df = pd.DataFrame(tokens.feature_list)

# Variable Testing (Do check if variables are passed)
#print(list_words_df.head(10))

# print(list_words_df["word"][0:2])

"""Set the training set split"""
X,y = load_dataset()

# Split 1: Get the 70% training set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Split 2: Get the remaining 30% of the dataset (15-15 split)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

""" Raw Data Analysis: Running Naive-Bayes & Decision trees"""

def raw_no_analysis():

 # Do naive-bayes analysis of selected data (2180 - 2200 range)

 """Test 1: Using Mutinomial Naive Bayes"""
 mnb = MultinomialNB()  

 y_pred = mnb.fit(X_train, y_train).predict(X_test)

 print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

 print(classification_report(y_val, y_pred))


 """Test 2: Using Bernoulli Naive Bayes"""

 bnb = BernoulliNB()

 y_pred2 = bnb.fit(X_train, y_train).predict(X_test)

 print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

 print(classification_report(y_val, y_pred2))

 # Do a decision tree analysis

 clf = tree.DecisionTreeClassifier(random_state=0)
 clf = clf.fit(X_train, y_train)

 y_pred3 = clf.predict(X_test)

 # Print figure using pyplot
 mp.figure(figsize=(30, 30))
 tree.plot_tree(clf, filled=True, max_depth=3)  # max_depth limits readability
 mp.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
 mp.close()

# Tester to check base token analysis
# if  __name__ == "__main__":
#  raw_no_analysis()



""" Run features created by group (Implement from features_tokens.py)"""
def raw_analysis():

 # Logic Steps 
 # 0. Provide observation of features that the group wants to observe
 # 1. Code the logic of said features 
 # 2. Append the given logic into a table (dataframe)
 # 3. Check (by-hand) and look for observations 
 # 4. Validate logic and improve syntax

 # To Complete:
 # 1. Find patterns in the dataset that we assume has a relationship
 # 2. Code implementation of the relevant features (I'm assuming we need at least 3 unique ones)
 # 3. Run the naive-bayes & tree implementation per analysis

 '''1. Basic Word Analysis (from get_features)'''


 '''2. TBA'''

 '''3. TBA'''
 pass


# Tester to check analysis w/ feature engineering
#if  __name__ == "__main__":
#  raw_no_analysis()














