import sys
import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


def getFiles(input, output):
    all_files = []
    if os.path.isdir(input):
        for r, d, f in os.walk(input):
            for file in f:
                if '.csv' in file:
                    all_files.append(os.path.join(r, file))
    
    elif os.path.isfile(input):
        all_files.append(input)

    else :
        print("input is not a file or a directory")
    
    print(all_files)
    for file in all_files:
        fileName = os.path.splitext(os.path.basename(input))[0]
        print(fileName)
        baseOutputFile = output + '/' + fileName

        df = pd.read_csv(file)
        tronc = len(df.columns) - 1
        dfs = np.split(df, [tronc], axis=1)
        data = dfs[0]
        target = dfs[1]
        print(data)
        print(target)
        X = data.values
        X = X.astype(float)
        X = np.nan_to_num(X)
        print(X)
        print("does X contains nan values ? : ", np.isnan(X).any())
        y = target.values
        y = y.astype(bool)
        print(y)

        univariateSelection(X, y, data, baseOutputFile)
        recursiveFeatureElimination(X, y, baseOutputFile)
        extraTreesClassifier(X, y, data, baseOutputFile)

def univariateSelection(X, y, data, baseOutputFile):
    test = SelectKBest(score_func=chi2, k=20)
    fit = test.fit(X, y)
    np.set_printoptions(precision=3)
    print(fit.scores_)
    print(type(fit.scores_))
    # Les valeurs de fit.scores_ représentent le score de l'attribut en fonction de la ligne (Ligne 1 --> HPCP_0; Ligne 29 --> HPCP_28)
    np.savetxt(baseOutputFile + "-TestChi2.csv", fit.scores_, delimiter=',')
    print("univariate Selection scores save in " + baseOutputFile)
    
    # IMAGE DEBUT
    feat_importances = pd.Series(fit.scores_, index=data.columns)
    feat_importances.nlargest(len(data.columns)).plot(kind='barh') # len(data.columns) à remplacer (si nécessaire) pour modifier le nombre de features que tu veux avoir sur l'image
    plt.savefig(baseOutputFile + "-TestChi2.png")
    # IMAGE FIN

def recursiveFeatureElimination(X, y, baseOutputFile):
    model = LogisticRegression()
    rfe = RFE(model, 1)

    fit = rfe.fit(X, y)

    print("Num Features: %d"% fit.n_features_)
    print("Selected Features: %s"% fit.support_) 
    print("Feature Ranking: %s"% fit.ranking_)
    
    # Les valeurs de fit.ranking_ représentent la position au classement de l'attribut en fonction de la ligne (Ligne 1 --> HPCP_0; Ligne 29 --> HPCP_28)
    np.savetxt(baseOutputFile + "-RecursiveFeatureElimination.csv", fit.ranking_, delimiter=',', fmt="%s")
    print("Recursive feature elemination ranking save in " + baseOutputFile)

def extraTreesClassifier(X, y, data, baseOutputFile):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    
    # Les valeurs de model.feature_importances_ représentent le score de l'attribut en fonction de la ligne (Ligne 1 --> HPCP_0; Ligne 29 --> HPCP_28)
    np.savetxt(baseOutputFile + "-ExtraTreesClassifier.csv", model.feature_importances_, delimiter=',')
    print("Extra Trees Classifier scores save in " + baseOutputFile)

    #IMAGE DEBUT
    feat_importances = pd.Series(model.feature_importances_, index=data.columns)
    feat_importances.nlargest(len(data.columns)).plot(kind='barh')
    plt.savefig(baseOutputFile + '-ExtraTreeClassifier.png')
    #IMAGE FIN

if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    getFiles(input, output)