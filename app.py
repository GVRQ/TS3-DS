print("NUWE: Talent Squad 3 2022 - Data Science")
print("Working...")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB

le = preprocessing.LabelEncoder()

RNDM = 1234
warnings.simplefilter(action='ignore', category=FutureWarning)

# train dataset
try:
    df_train = pd.read_csv('https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Talent+Squad+League/3rd_batch/data/train.csv', sep=',')
except:
    df_train = pd.read_csv('/train.csv', sep=',')
df_train.head()


# Test dataset
try:
    df_test = pd.read_csv('https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Talent+Squad+League/3rd_batch/data/test.csv', sep=',')
except:
    df_test = pd.read_csv('test.csv', sep=',')
df_test.head()


train = df_train.copy()
test = df_test.copy()


train = df_train.copy()
train.drop(["Unnamed: 0"], axis=1, inplace=True)
test = df_test.copy()
test.drop(["Unnamed: 0"], axis=1, inplace=True)


# Let's rename columns

train = train.rename(columns={'parental level of education':'parental_ed', 
                    'test preparation course':'test_prep', 'math score':'math_score', 
                    'reading score':'reading_score','writing score':'writing_score'})
train.columns = train.columns.str.lower()
train.columns

test = test.rename(columns={'parental level of education':'parental_ed', 
                    'test preparation course':'test_prep', 'math score':'math_score', 
                    'reading score':'reading_score','writing score':'writing_score'})
test.columns = test.columns.str.lower()
test.columns


# Adding `total_score` to datasets

train['total_score']=(train['math_score']+train['reading_score']+train['writing_score'])/3
test['total_score']=(test['math_score']+test['reading_score']+test['writing_score'])/3



# Top Scorers 
top_scorers = train[(train['math_score'] > 90) & (train['reading_score'] > 90) & (train['writing_score']>90)]\
.sort_values(by=['total_score'],ascending=False)

# Other Scorers


other_scorers = train[(train['math_score'] <= 90) & (train['reading_score'] <= 90) & (train['writing_score']<=90)]\
.sort_values(by=['total_score'],ascending=False)

overall_passmark = 60*3

# Overall Pass students

train['overall_pass'] = np.where(train['total_score'] < overall_passmark, '0', '1')

test['overall_pass'] = np.where(test['total_score'] < overall_passmark, '0', '1')



# Low-income
train.loc[(train['test_prep'] != 'comleted') & (train['lunch'] != 'standard'), 'low_income'] = 1
train['low_income'] = train['low_income'].fillna(0).astype('int')

# High-income
train.loc[(train['test_prep'] == 'comleted') & (train['lunch'] != 'standard'), 'high_income'] = 1
train['high_income']= train['high_income'].fillna(0).astype('int')

# Low-income
test.loc[(test['test_prep'] != 'comleted') & (test['lunch'] != 'standard'), 'low_income'] = 1
test['low_income'] = test['low_income'].fillna(0).astype('int')

# High-income
test.loc[(test['test_prep'] == 'comleted') & (test['lunch'] != 'standard'), 'high_income'] = 1
test['high_income']=test['high_income'].fillna(0).astype('int')


# function for creation of column: Other family income 
train['other_income'] = np.nan 

def other_income_cat(row):
    if row['low_income']==1:
        return '0'
    elif row['high_income']==1:
        return '0'
    else:
        return '1'

train['other_income'] = train.apply(other_income_cat, axis=1)
train['other_income'] = train['other_income'].astype('int')


# function for creation of column: Other family income 
test['other_income'] = np.nan 

def other_income_cat(row):
    if row['low_income']==1:
        return '0'
    elif row['high_income']==1:
        return '0'
    else:
        return '1'

test['other_income'] = test.apply(other_income_cat, axis=1)
test['other_income'] = test['other_income'].astype('int')

# Replacing NaN values & Changing Type to int
train['low_income'] = train['low_income'].fillna(0).astype('int')
train['high_income'] = train['high_income'].fillna(0).astype('int')
test['low_income'] = test['low_income'].fillna(0).astype('int')
test['high_income'] = test['high_income'].fillna(0).astype('int')

failers = train.loc[train['overall_pass'] != '1']

passers = train.loc[train['overall_pass'] == '1']

# Preprocessing data
train["lunch"] = le.fit_transform(train["lunch"])
train["test_prep"] = le.fit_transform(train["test_prep"])
train["gender"] = le.fit_transform(train["gender"])
train["math_score"] /= 100
train["reading_score"] /= 100
train["writing_score"] /= 100
train["total_score"] /= 100

test["lunch"] = le.fit_transform(test["lunch"])
test["test_prep"] = le.fit_transform(test["test_prep"])
test["gender"] = le.fit_transform(test["gender"])
test["math_score"] /= 100
test["reading_score"] /= 100
test["writing_score"] /= 100
test["total_score"] /= 100

# copying train datasets

train_pre = train.copy()

X = train_pre.drop(["parental_ed"], axis=1)
y = train_pre["parental_ed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RNDM, stratify=y)
class_names = ["0", "1", "2", "3", "4", "5"]


clf = LazyClassifier(random_state=RNDM, verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# The best model is GaussianNB.

# Building Model: GaussianNB

model = GaussianNB()


model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
f1_None = metrics.f1_score(y_test, y_pred, average=None)

# Prediction

# JSON Export
prediction = model.predict(test)
df_prediction = pd.DataFrame(prediction, columns=['target'])
json_prediction = df_prediction.to_json()
with open('predictions.json', 'w') as outfile:
    outfile.write(json_prediction)

print("Done!")
