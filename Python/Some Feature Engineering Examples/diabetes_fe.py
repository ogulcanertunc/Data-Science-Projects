import numpy as np
import pandas as pd
from helpers.data_prep import *
from helpers.eda import *
from helpers.helpers import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("Lectures/Week 7/Dosyalar/diabetes.csv")
# Pregnancies: Number of times pregnant
# Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml)
# BMI: Body mass index (weight in kg/(height in m)2)
# DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
# Age: Age (years)
# Outcome: Class variable (0 if non-diabetic, 1 if diabetic)

check_df(df)
df.describe().T

#####################
### Missing Value ###
#####################
def display_missing(df):
    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

display_missing(df)
# Some variables in our dataset have 0, but in theoratically it cant be possible, so
# in that step I will change these values with variable's median.

def change_0s_with_median(dataframe, na_list):
    for mem in na_list:
        if (dataframe[mem] == 0).any() == True:
            print(mem , "has 0 value")
            print("##########################")
            median_val = dataframe[mem].median()
            dataframe[mem] = dataframe[mem].replace(to_replace=0, value=median_val)
            print(mem , "values have been changed with variable's median")
            print("_____________________________________________________")
list_0 = ["Insulin","SkinThickness","Glucose","BloodPressure","BMI"]
change_0s_with_median(df, list_0)

##########################
### FEATURE EXTRACTION ###
##########################

## AGE ###
##########
# https://www.who.int/southeastasia/health-topics/adolescent-health#:~:text=WHO%20defines%20'Adolescents'%20as%20individuals,age%20range%2010%2D24%20years.
df.loc[(df["Age"] < 25), "AGE_CAT"] = "young"
df.loc[(df["Age"] >= 25) & (df["Age"] <= 55), "AGE_CAT"] = "mature"
df.loc[(df["Age"] > 55), "AGE_CAT"] = "senior"

### BMI ###
###########
# https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html
# Below 18.5	Underweight
# 18.5 – 24.9	Normal or Healthy Weight
# 25.0 – 29.9	Overweight
# 30.0 and Above	Obese

df.loc[(df['BMI'] < 18.5), "Weight"] = "Underweight"
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] <= 24.9), 'Weight'] = 'Normal'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'Weight'] = 'Overweight'
df.loc[(df['BMI'] >= 30), 'Weight'] = 'Obese'
df.head()

### Glucose ###
###############
# https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451#:~:text=A%20blood%20sugar%20level%20less,mmol%2FL)%20indicates%20prediabetes.
#A blood sugar level less than 140 mg/dL (7.8 mmol/L) is normal. A reading of more than 200 mg/dL (11.1 mmol/L) after
# two hours indicates diabetes. A reading between 140 and 199 mg/dL (7.8 mmol/L and 11.0 mmol/L) indicates prediabetes.

df.loc[(df['Glucose'] < 140), "Sugar Risk"] = "Normal"
df.loc[(df['Glucose'] >= 140) & (df['Glucose'] <= 190), 'Sugar Risk'] = 'Risky'
df.loc[(df['Glucose'] >= 199), 'Sugar Risk'] = 'Dangerous'


########################
### ONE HOT ENCODING ###
########################
ohe = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
from helpers.data_prep import one_hot_encoder

df = one_hot_encoder(df, ohe)

df.columns = [col.lower() for col in df.columns]

################
### Outliers ###
################

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col != 'Outcome']

for col in num_cols:
        replace_with_thresholds(df, col)

#############
### MODEL ###
#############
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

y = df["outcome"]
X = df.drop(["outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,stratify=y, random_state=123)
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    #plt.savefig('importances-01.png')
    plt.show()

plot_importance(rf_model, X)
