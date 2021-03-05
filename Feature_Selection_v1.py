import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

def chiSquared():

    # drop target columns
    dataframe = pd.read_csv('Zasd.csv')

    # drop target columns
    drop_cols = ['Cath']
    X = dataframe.drop(drop_cols, axis=1)  # X = independent columns (potential predictors)
    y = dataframe['Cath']  # y = target column (what we want to predict)
    # initialize SelectKBest to dind 16 best features
    best_features = SelectKBest(chi2, k=20)
    fit = best_features.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Feature_Name', 'Score']  
    return (feature_scores.nlargest(16,'Score'))
    # export the selected features to a .csv file
    df_univ_feat = feature_scores.nlargest(16, 'Score')
    df_univ_feat.to_csv('feature_selection_UNIVARIATE.csv', index=False)




    

def treeClassifier():
    dataframe = pd.read_csv("Zasd.csv")
    X = dataframe.iloc[:,0:54]  #independent columns
    y = dataframe.iloc[:,-1]    #target column i.e price range
    model = ExtraTreesClassifier()
    model.fit(X,y)
  
    #plot graph of features 
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(16).plot(kind='barh')
    # plt.show()
    # print(feat_importances.nlargest(20))
    return (feat_importances.nlargest(16))

# chiSquared()
# treeClassifier()
