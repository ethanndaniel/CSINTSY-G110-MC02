"""Train the model for the given dataset."""

import tokens
import pandas as pd
import numpy as np 
import matplotlib.pyplot as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from tokens import load_dataset, load_dataset_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.metrics import classification_report


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

 print(classification_report(y_test, y_pred))


 """Test 2: Using Bernoulli Naive Bayes"""

 bnb = BernoulliNB()

 y_pred2 = bnb.fit(X_train, y_train).predict(X_test)

 print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred2).sum()))

 print(classification_report(y_test, y_pred2))

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
#if  __name__ == "__main__":
# raw_no_analysis()



""" Run features created by group (Implement from features_tokens.py)"""
 # Logic Steps 
 # 0. Provide observation of features that the group wants to observe
 # 1. Code the logic of said features 
 # 2. Append the given logic into a table (dataframe)
 # 3. Check (by-hand) and look for observations 
 # 4. Validate logic and improve syntax

 # To Complete:
 # 1. Find patterns in the dataset that we assume has a relationship
 # 2. Code implementation of the relevant features (Do add further features to improve analysis)
 # 3. Run the naive-bayes & tree implementation per analysis

def raw_analysis():
    
    '''1 & 2. Basic word features (get_features) + pattern recognition (p_recognition)'''
    X_full, y_full = load_dataset_model()

    # Cross-check: confirm y_full actually holds what we expect before splitting
    #print(f"Total rows in y_full: {len(y_full)}")
    #print(f"ENG count: {y_full.count('ENG')}")
    #print(f"FIL count: {y_full.count('FIL')}")
    #print(f"OTH count: {y_full.count('OTH')}")
    #print(f"CS count: {y_full.count('CS')}")

    # Split 1: Get the 70% training set
    X_train_f, X_temp_f, y_train_f, y_temp_f = train_test_split(X_full, y_full, test_size=0.3, random_state=42
)
    # Split 2: Get the remaining 30% of the dataset (15-15 split)
    X_val_f, X_test_f, y_val_f, y_test_f = train_test_split(X_temp_f, y_temp_f, test_size=0.50, random_state=42)

    '''3. Run naive-bayes & tree implementation on the engineered feature set'''
    mnb = MultinomialNB()
    y_pred = mnb.fit(X_train_f, y_train_f).predict(X_test_f)
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test_f.shape[0], (y_test_f != y_pred).sum()))
    print("Multinomial Naive-Bayes\n")
    print(classification_report(y_test_f, y_pred))

    bnb = BernoulliNB()
    y_pred2 = bnb.fit(X_train_f, y_train_f).predict(X_test_f)
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test_f.shape[0], (y_test_f != y_pred2).sum()))
    print("Bernoulli's Naive-Bayes\n")
    print(classification_report(y_test_f, y_pred2))

    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X_train_f, y_train_f)
    y_pred3 = clf.predict(X_test_f)
    print(classification_report(y_test_f, y_pred3))

    mp.figure(figsize=(30, 30))
    tree.plot_tree(clf, filled=True, max_depth=3)
    mp.savefig("decision_tree_model.png", dpi=150, bbox_inches="tight")
    mp.close()



# Tester to check analysis w/ feature engineering
#if  __name__ == "__main__":
# raw_analysis()














