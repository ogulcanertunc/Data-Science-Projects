import numpy as np
import pandas as pd
from helpers.data_prep import *
from helpers.eda import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc

import string
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Lectures/Week 7/Dosyalar/titanic.csv")

df.head()
### Overview of the Data ###
# PassengerId is the unique id of the row and it doesn't have any effect on target
# Survived is the target variable we are trying to predict (0 or 1):
# 1 = Survived
# 0 = Not Survived
# Pclass (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has 3 unique values (1, 2 or 3):
# 1 = Upper Class
# 2 = Middle Class
# 3 = Lower Class
# Name, Sex and Age are self-explanatory
# SibSp is the total number of the passengers' siblings and spouse
# Parch is the total number of the passengers' parents and children
# Ticket is the ticket number of the passenger
# Fare is the passenger fare
# Cabin is the cabin number of the passenger
# Embarked is port of embarkation and it is a categorical feature which has 3 unique values (C, Q or S):
# C = Cherbourg
# Q = Queenstown
# S = Southampton

# def titanic_data_prep(dataframe):

    # FEATURE ENGINEERING
    dataframe["NEW_CABIN_BOOL"] = dataframe["Cabin"].isnull().astype('int')
    dataframe["NEW_NAME_COUNT"] = dataframe["Name"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["Name"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    dataframe['NEW_TITLE'] = dataframe.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    dataframe["NEW_AGE_PCLASS"] = dataframe["Age"] * dataframe["Pclass"]

    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

    dataframe.loc[(dataframe['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['Age'] >= 18) & (dataframe['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['Sex'] == 'male') & ((dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & ((dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # AYKIRI GOZLEM
    num_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) > 20
                and dataframe[col].dtypes != 'O'
                and col not in "PASSENGERID"]

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # for col in num_cols:
    #    print(col, check_outlier(df, col))
    # print(check_df(df))


    dataframe.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))

    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    # LABEL ENCODING
    binary_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) == 2 and dataframe[col].dtypes == 'O']

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    dataframe = rare_encoder(dataframe, 0.01)

    ohe_cols = [col for col in dataframe.columns if 10 >= len(dataframe[col].unique()) > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    return dataframe

# df.columns

# df_prep = titanic_data_prep(df)



print(df.info())

######################
### Missing Values ###
######################

def display_missing(df):
    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')
# display_missing(df)

df.isnull().sum()

### Age ###
###########
# Missing values in Age are filled with median age, but using median age of the whole data set is
# not a good choice. Median age of Pclass groups is the best choice because of its high correlation with Age
# (0.408106) and Survived (0.338481). It is also more logical to group ages by passenger classes instead of
# other features.

df_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_corr[df_corr['Feature 1'] == 'Age']


age_by_pclass_sex = df.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

### Embarked ###
################
# Embarked is a categorical feature and there are only 2 missing values in whole data set.
# Both of those passengers are female, upper class and they have the same ticket number. This means
# that they know each other and embarked from the same port together. The mode Embarked value for an upper
# class female passenger is C (Cherbourg), but this doesn't necessarily mean that they embarked from that port.

df[df['Embarked'].isnull()]

# When I googled Stone, Mrs. George Nelson (Martha Evelyn), I found that she embarked from S (Southampton) with her
# maid Amelie Icard, in this page Martha Evelyn Stone: Titanic Survivor.

# Mrs Stone boarded the Titanic in Southampton on 10 April 1912 and was travelling in first class with her maid
# Amelie Icard. She occupied cabin B-28.

# Missing values in Embarked are filled with S with this information.
# Filling the missing values in Embarked with S
df['Embarked'] = df['Embarked'].fillna('S')

### Cabin ###
# Cabin feature is little bit tricky and it needs further exploration. The large portion of the Cabin feature is
# missing and the feature itself can't be ignored completely because some the cabins might have higher survival rates.
# It turns out to be the first letter of the Cabin values are the decks in which the cabins are located. Those decks
# were mainly separated for one passenger class, but some of them were used by multiple passenger classes.

# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_decks = df.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch',
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId',
                                                                        'Ticket']).rename(
    columns={'Name': 'Count'}).transpose()


def get_pclass_dist(df):
    # Creating a dictionary for every passenger class count in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    decks = df.columns.levels[0]

    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass][0]
                deck_counts[deck][pclass] = count
            except KeyError:
                deck_counts[deck][pclass] = 0

    df_decks = pd.DataFrame(deck_counts)
    deck_percentages = {}
    # Creating a dictionary for every passenger class percentage in every deck
    for col in df_decks.columns:
        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]

    return deck_counts, deck_percentages


all_deck_count, all_deck_per = get_pclass_dist(df_decks)

# Passenger in the T deck is changed to A
idx = df[df['Deck'] == 'T'].index
df.loc[idx, 'Deck'] = 'A'

df_all_decks_survived = df.groupby(['Deck', 'Survived']).count().drop(
    columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare',
             'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()
df
df = df.drop("Cabin",axis = 1)
def get_survived_dist(df):
    # Creating a dictionary for every survival count in every deck
    surv_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}}
    decks = df.columns.levels[0]

    for deck in decks:
        for survive in range(0, 2):
            surv_counts[deck][survive] = df[deck][survive][0]

    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}

    for col in df_surv.columns:
        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]

    return surv_counts, surv_percentages

all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)

df.loc[((df['SibSp'] + df['Parch']) > 0), "Alone"] = "No"
df.loc[((df['SibSp'] + df['Parch']) == 0), "Alone"] = "Yes"

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'Child'
df.loc[(df['Age'] >= 18) & (df['Age'] < 25), 'NEW_AGE_CAT'] = 'Young'
df.loc[(df['Age'] >= 25) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'Mature'
df.loc[(df['Age'] >= 56), "NEW_AGE_CAT"] = 'Senior'


df.loc[(df['Sex'] == 'male') & (df['NEW_AGE_CAT'] == "Child"), 'NEW_SEX_CAT'] = 'Child-male'
df.loc[(df['Sex'] == 'male') & (df['NEW_AGE_CAT'] == "Young"), 'NEW_SEX_CAT'] = 'Young-male'
df.loc[(df['Sex'] == 'male') & (df['NEW_AGE_CAT'] == "Mature"), 'NEW_SEX_CAT'] = 'Mature-male'
df.loc[(df['Sex'] == 'male') & (df['NEW_AGE_CAT'] == "Child"), 'NEW_SEX_CAT'] = 'Senior-male'
df.loc[(df['Sex'] == 'female') & (df['NEW_AGE_CAT'] == "Child"), 'NEW_SEX_CAT'] = 'Child-female'
df.loc[(df['Sex'] == 'female') & (df['NEW_AGE_CAT'] == "Young"), 'NEW_SEX_CAT'] = 'Young-female'
df.loc[(df['Sex'] == 'female') & (df['NEW_AGE_CAT'] == "Mature"), 'NEW_SEX_CAT'] = 'Mature-female'
df.loc[(df['Sex'] == 'female') & (df['NEW_AGE_CAT'] == "Senior"), 'NEW_SEX_CAT'] = 'Senior-female'

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
sum(df["NEW_NAME_WORD_COUNT"])

df.columns = [x.lower() for x in df.columns]

###############
### Outlier ###
###############
num_cols = [col for col in df.columns if len(df[col].unique()) > 20
            and df[col].dtypes != 'O'
            and col not in "passengerid"]

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

check_df(df)


######################
### Label Encoding ###
######################
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

for col in binary_cols:
    df = label_encoder(df, col)

df


########################
### One Hot Encoding ###
########################

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols)

df.head()

#############
### MODEL ###
#############
df.drop(["ticket","name"], inplace=True, axis=1)
y = df["survived"]
X = df.drop(["passengerid", "survived"], axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier().fit(X, y)
y_pred = rf_model.predict(X)
accuracy_score(y_pred, y)

from matplotlib import pyplot as plt

def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    #plt.savefig('importance-01.png')
    plt.show()

plot_importance(rf_model, X)




















