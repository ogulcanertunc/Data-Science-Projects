import numpy as np
import pandas as pd
from helpers.data_prep import *
from helpers.eda import *
from helpers.helpers import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv('Lectures/Week 7/Dosyalar/hitters.csv')
df.info()
df.describe().T
check_df(df)
### Description ###
# Description
# Context
#
# This dataset is part of the R-package ISLR and is used in the related book by G. James et al. (2013) "An Introduction
# to Statistical Learning with applications in R" to demonstrate how Ridge regression and the LASSO are performed using R.
#
# Content
#
# This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University. This is
# part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from
# Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia
# Update published by Collier Books, Macmillan Publishing Company, New York.

# Format
# A data frame with 322 observations of major league players on the following 20 variables.
# AtBat: Number of times at bat in 1986
# Hits: Number of hits in 1986
# HmRun: Number of home runs in 1986
# Runs: Number of runs in 1986
# RBI: Number of runs batted in in 1986
# Walks: Number of walks in 1986
# Years: Number of years in the major leagues
# CAtBat: Number of times at bat during his career
# CHits: Number of hits during his career
# CHmRun: Number of home runs during his career
# CRuns: Number of runs during his career
# CRBI: Number of runs batted in during his career
# CWalks: Number of walks during his career
# League: A factor with levels A and N indicating player’s league at the end of 1986
# Division: A factor with levels E and W indicating player’s division at the end of 1986
# PutOuts: Number of put outs in 1986
# Assists: Number of assists in 1986
# Errors: Number of errors in 1986
# Salary: 1987 annual salary on opening day in thousands of dollars
# NewLeague: A factor with levels A and N indicating player’s league at the beginning of 1987




######################
### Missing Values ###
######################
df.isna().sum()

df["Salary"] = df["Salary"].fillna(df['Salary'].median())
df.dropna(inplace=True)
df.info()
df.shape

###########################
### Feature Engineering ###
###########################
df["Hit_class"] = pd.qcut(df['Hits'], 4, labels=['D','C','B','A'])

df['HitRatio'] = df['Hits'] / df['AtBat']
df['RunRatio'] = df['HmRun'] / df['Runs']
df['CHitRatio'] = df['CHits'] / df['CAtBat']
df['CRunRatio'] = df['CHmRun'] / df['CRuns']

df['Avg_AtBat'] = df['CAtBat'] / df['Years']
df['Avg_Hits'] = df['CHits'] / df['Years']
df['Avg_HmRun'] = df['CHmRun'] / df['Years']
df['Avg_Runs'] = df['CRuns'] / df['Years']
df['Avg_RBI'] = df['CRBI'] / df['Years']
df['Avg_Walks'] = df['CWalks'] / df['Years']
df['Avg_PutOuts'] = df['PutOuts'] / df['Years']
df['Avg_Assists'] = df['Assists'] / df['Years']
df['Avg_Errors'] = df['Errors'] / df['Years']

df.loc[(df['Years'] <= 5), 'Exp'] = 'Fresh'
df.loc[(df['Years'] > 5) & (df['Years'] <= 10), 'Exp'] = 'Starter'
df.loc[(df['Years'] > 10) & (df['Years'] <= 15), 'Exp'] = 'Average'
df.loc[(df['Years'] > 15) & (df['Years'] <= 20), 'Exp'] = 'Experienced'
df.loc[(df['Years'] > 20), 'Exp'] = 'Veteran'

dff = df.copy()
dff.isna().sum()
dff.RunRatio.head()
dff['RunRatio'].fillna((dff['RunRatio'].mean()), inplace=True)
dff = df.drop(['AtBat','Hits','HmRun','Salary','Runs','RBI','League','Division','NewLeague'], axis = 1)

######################
### Outlier Values ###
######################
num_cols = [col for col in dff.columns if len(dff[col].unique()) > 20 and dff[col].dtypes != 'O']

for col in num_cols:
    replace_with_thresholds(dff, col)
dff.shape

######################
### LABEL ENCODING ###
######################

binary_cols = [col for col in dff.columns if len(dff[col].unique()) == 2 and dff[col].dtypes == 'O']

for col in binary_cols:
    dff = label_encoder(dff, col)

########################
### ONE HOT ENCODING ###
########################

ohe = [col for col in dff.columns if 10 >= len(dff[col].unique()) > 2]
dff = one_hot_encoder(dff, ohe)
dff.columns = [col.lower() for col in dff.columns]



