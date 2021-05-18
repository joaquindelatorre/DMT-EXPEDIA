"""
This script creates an ensemble containing three models, which are logistic regression, 
decision tree and gradient boosting. 

Author: Ximeng Wang
Date: 16/05/2020

"""

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def ensemble_run(df_x, df_y, test_set):
    # Logistic regression
    logreg = LogisticRegression(max_iter= 10000)
    logreg.fit(df_x, df_y)
    pred_logreg = logreg.predict(test_set)

    # Decision tree classifier
    dtree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=14,
                                   max_features='sqrt', max_leaf_nodes=10,
                                   min_impurity_split=1e-07, min_samples_leaf=1,
                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    random_state=1, splitter='best')
    dtree.fit(df_x, df_y)
    pred_dtree = dtree.predict(test_set)

    # Gradient boosting classifier
    gbc = GradientBoostingClassifier()
    gbc.fit(df_x, df_y)
    pred_gbc = gbc.predict(test_set)

    # K-nearest neighbour classifier
    knn = KNeighborsClassifier(n_neighbors= 5)
    knn.fit(df_x, df_y)
    pred_knn = knn.predict(test_set)

    # Assign weightings on predictions of different models
    test_label = 0.25 * pred_logreg + 0.1 * pred_dtree + 0.4 * pred_gbc + 0.25 * pred_knn
    test_set= test_set.assign(label=test_label)

    # # Create ensemble table and take the mode for every array as class
    # ensemble_table = pd.DataFrame()
    # ensemble_table = ensemble_table.append(pd.Series(pred_logreg, name="logreg"))
    # ensemble_table = ensemble_table.append(pd.Series(pred_dtree, name="dtree"))
    # ensemble_table = ensemble_table.append(pd.Series(pred_gbc, name="gbc"))
    # ensemble_table = ensemble_table.append(pd.Series(pred_knn, name="knn"))
    #
    # final_pred = ensemble_table.mode(axis = 0, numeric_only= False).iloc[0] #.tolist()
    # test_set = test_set.assign(label=pd.Series(final_pred))

    # Sort samples with the same 'srch_id' in descending order of the 'label' value
    test_set = test_set.groupby(['srch_id']).apply(lambda x:
            x.sort_values(['label'], ascending =False)).reset_index(drop=True)

    submission = test_set[['srch_id','prop_id']]

    return submission


if __name__ == "__main__":
    # Read two data sets
    train_set = pd.read_csv('train_processed.csv')
    test_set = pd.read_csv('test_processed.csv')

    # Get features and class from the training set
    df_x = train_set.drop(columns = 'label')
    df_y = train_set.label

    # Run ensemble_run and get final submission
    submission = ensemble_run(df_x, df_y, test_set)
    submission.to_csv("submission_2.csv", index=False)


