import pandas as pd
 
df_train = pd.read_csv('CREDIT CARD FRAUD DETECTION\\Data\\fraudTrain.csv')
df_test = pd.read_csv('CREDIT CARD FRAUD DETECTION\\Data\\fraudTest.csv')

#EDA to show that this is imbalanced dataset, and make it balanced again

x_train = df_train[['category', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']]
y_train = df_train['is_fraud']

x_test = df_test[['category', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']]
y_test = df_test['is_fraud']

# Labelencoding x['fraud]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x_train['category'], x_test['category'] = encoder.fit_transform(x_train['category']), encoder.fit_transform(x_test['category'])

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
models = [LogisticRegression(), DecisionTreeClassifier(), XGBClassifier()]

for i in models:
    i.fit(x_train, y_train)
    y_pred = i.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for using model {i} is {accuracy*100: .2f}%')

