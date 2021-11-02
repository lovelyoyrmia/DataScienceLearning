import pandas as pd
from sklearn import linear_model

dataTitanic = pd.read_csv('train.csv',sep=',')

dataTitanic = dataTitanic[['Name','Survived','Sex','Pclass','Age','Ticket','Fare']]

'''To print the entire data i have'''
print(dataTitanic.head())

'''To know the amount of data that i have'''
print(dataTitanic.shape)

'''To give information about the entire data'''
print(dataTitanic.info())

'''To check if there's data is Null'''
print(dataTitanic.isnull().sum())

'''To fill data's null'''
dataTitanic = dataTitanic.fillna(dataTitanic.mean())
print(dataTitanic.mean())
print(dataTitanic.isnull().sum())

'''To describe data'''
print(dataTitanic.describe())

'''To make an histogram from data'''
dataTitanic[['Age','Fare']].hist(figsize=(16,10),xlabelsize=8,ylabelsize=8);
print(dataTitanic)

'''To get the amount of People who Survived'''
print(dataTitanic['Survived'].value_counts())
print(dataTitanic.pivot_table(['Survived'],['Sex','Pclass']).sort_values(by=['Survived'],ascending=False))

'''To get the amount of People who went alone'''
alone = [0 for k in range(len(dataTitanic))]
for p in range(len(dataTitanic)):
    if dataTitanic['SibSp'] == 0 and dataTitanic['Parch'][p] == 0:
       alone[p] = 1
    dataTitanic = dataTitanic.assign(isAlone = alone)
    print(dataTitanic['isAlone'].value_counts())

'''To get the correlation of data'''
print(dataTitanic.corr())


'''To get the correlation of data Survived'''
df_corr = dataTitanic[['Survived', 'Age', 'Fare']]
df_corr.loc[:, 'Age'] = df_corr.loc[:, 'Age'].round()
corr = df_corr.corr()['Survived'][1:]
print('Korelasi dengan variable Survived: ', '\n', corr)

