import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import precision_recall_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', 200)


#evaluation the models
def evaluating(X_test, y_test, X_train, y_train, model):
    y_predict = model.predict(X_test)
    model.score(X_test, y_test)
    print(' training score: {}'.format(model.score(X_train, y_train)))
    print(' testing score: {}'.format(model.score(X_test, y_test)))
    
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict, labels=[0, 1]).ravel()
    print(tn, fp, fn, tp)
    FalseNegativeRate= fp/(fp+tp)
    print('False negative rate: %.3f' %(FalseNegativeRate))

    predict = model.predict(X_train)
    accuracyTrain = accuracy_score(y_train, predict)
    accuracy = accuracy_score(y_test, y_predict)
    print('Accuracy test set: %.3f' %(accuracy*100))
    print('Accuracy train set: %.3f' %(accuracyTrain*100))

    precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

def cross_validation(X_train, y_train, model):
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print('Cross validation scores: {}'.format(scores))
    print('Mean cross validation score: {}'.format(scores.mean()))
    print('Standard deviation cross validation score: {}'.format(scores.std()))
    return scores