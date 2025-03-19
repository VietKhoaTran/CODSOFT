import pandas as pd
 
df_train = pd.read_csv('CREDIT CARD FRAUD DETECTION\\Data\\fraudTrain.csv')
df_test = pd.read_csv('CREDIT CARD FRAUD DETECTION\\Data\\fraudTest.csv')

#understand the dataset
print(df_train.info())
print(df_train.head())

import matplotlib.pyplot as plt
# Distribution of fraud transaction across categories
df_train['category'][df_train['is_fraud'] == 1].value_counts().plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.title('Distribution of Fraud among category')
plt.show()

# Distribution of fraud transaction across numerical variables
numerical_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.to_list()
remove = ['is_fraud', 'cc_num', 'Unnamed: 0']
numerical_cols = list(filter(lambda x: x not in remove, numerical_cols))

for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i+1)
    plt.hist(df_train[col][df_train['is_fraud'] == 1], bins = 30)
    plt.xlabel(f'{col}')
    plt.ylabel('Frequency')
plt.show()