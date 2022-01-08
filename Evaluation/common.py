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

# This file contains some common functions to make working in our notebooks easier and lessen the use of copy and paste.
# To use these functions in a subfolder, add the folowing to the top of your notebook (without quotes): "! ln -s ../common.py ./common.py"

random_state = 42

def readFile(filePath, sep=';'):
    data = pd.read_csv(filePath, sep=sep)
    return data

def loadTraining():
    return pd.read_csv("../Data Files/Training set/X_train.csv", sep=","), pd.read_csv("../Data Files/Training set/y_train.csv", sep=",")

def loadValidation():
    return pd.read_csv("..Data Files/Validation set/X_test.csv", sep=","), pd.read_csv("../Data Files/Validation set/y_test.csv", sep=",")

# Divide columns into a list containing numeric columns and a list containing object columns and return those
def getSubsetColumnsList(df):
    numeric_cols = []
    non_numeric_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() > (0.5*len(df)):
                numeric_cols.append(col)
            elif pd.to_numeric(df[col], errors='coerce').notnull().sum() < (0.5*len(df)):
                non_numeric_cols.append(col)

    return numeric_cols, non_numeric_cols 

# Remove strings from numeric columns and convert column datatype to numeric and return converted dataframe
def convertColumnTypes(cols_to_convert, df):
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df



# Remove columns where more than the half entries are missing and return new dataframe
def removeColumnWithMissingEntries(df, percentage):
    df_without_missing_entries = df.copy()
    for col in df_without_missing_entries.columns:
        if df_without_missing_entries[col].isna().sum() >= (percentage * len(df)):
            df_without_missing_entries.drop(col, axis=1, inplace=True)
            
    return df_without_missing_entries                



# Remove outliers which are bigger than 2* standard deviation + mean of a column
def removeOutliers(original_df):
    df = original_df.copy()
    df_std = df.std()
    df_mean = df.mean()
    
    upper_limit = {}
    lower_limit = {}
    for col in df.columns:
        if not df[col].dtype == 'object' and not 'Respondent' in col and not 'Zip' in col and not 'code' in col:
            upper_limit[col] = 2*df_mean[col]+ df_std[col]
            lower_limit[col] = -2*df_mean[col] - df_std[col]

    for col in upper_limit.keys():
        for ind, row in df.iterrows():
            if df.loc[ind, col] > upper_limit[col] or df.loc[ind, col] < lower_limit[col]:
                df.loc[ind,col] = np.nan
    return df

def removeOutliersIQR(original_df):
    df = original_df.copy()
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    for col in q1.index:
        if q1[col] == q3[col]:
            q1.drop(labels=col, inplace=True)
            q3.drop(labels=col, inplace=True) 
    iqr = q3-q1
    lower_limit = q1-1.5*iqr
    upper_limit = q3+1.5*iqr
    for col in iqr.index:
        for ind, row in df.iterrows():
            if not df.loc[ind, col] > upper_limit[col] and not df.loc[ind, col] > lower_limit[col]:
                df.at[ind, col] = np.nan
    return df 

# Method for saving names of columns with dtype == object in a list
def saveObjectColumnNameToList(df):
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_cols.append(col)

    return categorical_cols   


# drop columns named opmerkingen and unnamed
def dropUnusedColumns(df):
    for col in df.columns:
        if 'Opmerk' in col:
            df.drop(labels=col, axis=1, inplace=True)
        elif 'Unnamed' in col:
            df.drop(labels=col, axis=1, inplace=True)
        elif 'Remark' in col:
            df.drop(labels=col, axis=1, inplace=True)   
    return df

class ModelType(Enum):
    RandomForest = 0
    DecisionTree = 1
    KNN = 2
    Bagging = 3

def CreateModel(ModelType, train_X, train_y):
    if ModelType == ModelType.RandomForest:
        model = RandomForestClassifier(bootstrap = False,
                                       max_depth = 90,
                                       max_features = 3,
                                       min_samples_leaf = 1,
                                       min_samples_split = 5,
                                       n_estimators = 200,
                                       random_state = random_state)
        model.fit(train_X, train_y)
        return model
    elif ModelType == ModelType.DecisionTree:
        model = DecisionTreeClassifier(criterion='gini',max_depth=18, min_samples_leaf=1, splitter='random', random_state=random_state)
        model.fit(train_X, train_y)
        return model
    elif ModelType == ModelType.KNN:
        model= KNeighborsClassifier(n_neighbors=5)
        model.fit(train_X, train_y)
        return model
    elif ModelType == ModelType.Bagging:
        pipeline = make_pipeline(StandardScaler(),
                                    LogisticRegression(random_state=1))
        model = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
                                    max_features=10,
                                    max_samples=50,
                                    random_state=1, n_jobs=5)
        model.fit(train_X, train_y)
        return model
    else:
        return None

def CreateModelWithGridSearch(ModelType, train_X, train_y):
    if ModelType == ModelType.RandomForest:
        model = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 50, 100, 200, 500, 1000], 'max_depth': range(1,50), 'min_samples_leaf': range(1,100), 'random_state': [random_state]}, cv=5, scoring='accuracy')
        model.fit(train_X, train_y)
        return model
    elif ModelType == ModelType.DecisionTree:
        model = GridSearchCV(DecisionTreeClassifier(), param_grid = {'criterion': ["gini", "entropy"], 'splitter': ['best', 'random'], 'max_depth': range(1,50), 'min_samples_leaf': range(1,10)}, random_state=random_state)
        model.fit(train_X, train_y)
        return model
    elif ModelType == ModelType.KNN:
        model = GridSearchCV(KNeighborsClassifier(), param_grid = {'n_neighbors': range(1,50)}, random_state=random_state)
        model.fit(train_X, train_y)
        return model
    else:
        return None


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




# makes heatmaps and removes columns with correlation higher than 80%
def correlation(Data):
    import seaborn as sns
    corr = Data.corr()
    top_corr_features = corr.index
    plt.figure(figsize=(40,40))
    g=sns.heatmap(Data[top_corr_features].corr(), linewidths= .9 ,annot=True,cmap="RdYlGn", fmt='.1f')
    
    kot = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.8:
                if kot[j]:
                    kot[j] = False
    selected_columns = Data.columns[kot]
    print(selected_columns)
    df = Data[selected_columns]
    print(len(df.columns))
    return df



# Correlation between features and MQ category 
def correlation_MQ(Data, column):
    correlation_MQ = Data.corr()[str(column)].abs().sort_values(ascending = False)
    
    return correlation_MQ


# FeaturesImportances with RandomForestClassifier
def FeaturesImportances(X , y):
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt

    model= RandomForestClassifier(n_estimators = 340)
    model.fit(X,y)
    importances= model.feature_importances_
    #print(X.columns)
    findal_df = pd.DataFrame({"Features" : pd.DataFrame(X).columns , "importances" :importances })
    findal_df.set_index('importances')
    findal_df.plot.bar(color='blue')
    return findal_df

# Split data into train/test and vald files.
def Split(Data):
    
    Data = Data.drop(Data.columns[[0,1,4]], axis=1)
    y = Data['MQ category'].values 
    X = Data.drop('MQ category', axis = 1)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2 ,random_state=11111 , stratify = y) 
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=11111 )
    
    X_test.to_csv(r'\X_test.csv', index = False)
    y_test.to_csv(r'\y_test.csv', index = False)
    X_train.to_csv(r'\X_train.csv', index = False)
    y_train.to_csv(r'\y_train.csv', index = False)
    X_val.to_csv(r'\X_val.csv', index = False)
    y_val.to_csv(r'\y_val.csv', index = False)
    
    return X_test, y_test, X_train, y_train, X_val, y_val


# Scaling X 
def Scaling(X):
    from sklearn.preprocessing import StandardScaler 
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X 

# Balancing X and y train  
def Balancing(X_train,y_train):
    from imblearn.over_sampling import SMOTE
    import imblearn
    from collections import Counter
    from matplotlib import pyplot

    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    counter = Counter(y_train)
    for k,v in counter.items():
        per = v / len(k) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()
    
    return X_train , y_train
