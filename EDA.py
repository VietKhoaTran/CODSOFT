import pandas as pd
import numpy as np

df_train = pd.read_csv('MOVIE GENRE CLASSIFICATION\\Data\\train_data.txt', sep = ":::", names = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine = 'python')
df_test = pd.read_csv('MOVIE GENRE CLASSIFICATION\\Data\\test_data_solution.txt', sep = ":::", names = ['ID', 'TITLE','GENRE', 'DESCRIPTION'], engine = 'python')

#Understanding the dataset
print(df_train.head())

print(df_train.info())
print(df_test.info())

#Visualizing distribution of the dataset
import matplotlib.pyplot as plt

df_train['GENRE'].value_counts().plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.title('Distribution of Movie Genres for train dataset')
plt.show()

df_test['GENRE'].value_counts().plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.title('Distribution of Movie Genres for test dataset')
plt.show()