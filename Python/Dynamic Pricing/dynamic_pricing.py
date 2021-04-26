########################
### 1.Data Preparing ###
########################

# loading necessary libraries
#___________________________#
import pandas as pd
import itertools
import statsmodels.stats.api as sms
from scipy.stats import shapiro
import scipy.stats as stats
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# useful functions #
#__________________#
def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.01)
    quartile3 = dataframe[col_name].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# reading the data
df = pd.read_csv("pricing.csv", sep=";")
df.head()
df.isna().sum()


check_df(df) # There is a huge difference between 95% and 99% values
df.groupby("category_id")["price"].agg(["mean","sum","std"])

# Threshold values are determined for the price variable.
low, up = outlier_thresholds(df,"price")
print(f'Low Limit: {low}  Up Limit: {up}')

# Outlier values need to remove.
def has_outliers(dataframe, numeric_columns):
    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers, "outliers")

has_outliers(df, ["price"])

def remove_outliers(dataframe, numeric_columns):
    for variable in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]
    return dataframe_without_outliers

df = remove_outliers(df, ["price"])

check_df(df)
df.groupby("category_id").agg({"price": "mean"}).reset_index()
# It appears like there is difference between but we will use AB test to indicate the difference between and is statistically significant.


####################
### 2.AB Testing ###
####################

# 1.Checking Assumptions
# 1.1 Normal Distribution
# 1.2 Homogeneity of Variance

# 1.1 Normal Distribution
# H0: There is no statistically significant difference between sample distribution and theoretical normal distribution
# H1: There is statistically significant difference between sample distribution and theoretical normal distribution

print("Shapiro Wilks Test Result \n")
for x in df["category_id"].unique():
    test_statistic, pvalue = shapiro(df.loc[df["category_id"] == x, "price"])
    if (pvalue<0.05):
        print(f'{x}:')
        print('Test statistic = %.4f, p-value = %.4f' % (test_statistic, pvalue), "H0 is rejected")
    else:
        print(f'{x}:')
        print('Test statistic = %.4f, p-value = %.4f' % (test_statistic, pvalue), "H0 is not rejected")

# Normal distribution is not provided,so we can apply a non-parametric method.

# 2.Implementing Hypothesis
groups = []
for x in itertools.combinations(df["category_id"].unique(),2):
    groups.append(x)

result = []
print("Mann-Whitney U Test Result ")
for x in groups:
    test_statistic, pvalue = stats.stats.mannwhitneyu(df.loc[df["category_id"] == x[0], "price"],
                                                      df.loc[df["category_id"] == x[1], "price"])
    if (pvalue<0.05):
        result.append((x[0], x[1], "H0 is rejected"))
        print('\n', "{0} - {1} ".format(x[0], x[1]))
        print('Test statistic= %.4f, p-value= %.4f' % (test_statistic, pvalue), "H0 is rejected")
    else:
        result.append((x[0], x[1], "H0 is not rejected"))
        print('\n', "{0} - {1} ".format(x[0], x[1]))
        print('Test statistic= %.4f, p-value= %.4f' % (test_statistic, pvalue), "H0 is not rejected")


result_df = pd.DataFrame()
result_df["Category 1"] = [x[0] for x in result]
result_df["Category 2"] = [x[1] for x in result]
result_df["H0"] = [x[2] for x in result]
result_df



##################
### 3.Problems ###
##################


# Does the price of the item differ by category?
result_df[result_df["H0"] == "H0 is not rejected"]
# There is no statistically significant difference average price between 5 categorical groups
result_df[result_df["H0"] == "H0 is rejected"]
# There is a statistically significant difference average price between 9 categorical groups


# What should the item cost?
# The average of 4 statistically identical categories will be the price we will determine.
signif_cat = [361254, 874521, 675201, 201436]
sum = 0
for i in signif_cat:
    sum += df.loc[df["category_id"] == i,  "price"].mean()
PRICE = sum / 4

print("PRICE : %.4f" % PRICE)


# Flexible Price Range
# We list the prices of the 4 categories that selected for pricing
prices = []
for category in signif_cat:
    for i in df.loc[df["category_id"]== category,"price"]:
        prices.append(i)

print(f'Flexible Price Range: {sms.DescrStatsW(prices).tconfint_mean()}')


# Simulation For Item Purchases
# We will calculate the incomes that can be obtained from the minimum, maximum values of the confidence interval and the
# prices we set.

# for minimum price in confidence interval
# Price:36.7109597897918
freq = len(df[df["price"] >= 36.7109597897918])
# number of sales equal to or greater than this price
income = freq * 36.7109597897918
print(f'Income: {income}')

# for decided price
# Price:37.0924
freq = len(df[df["price"] >= 37.0924])
# number of sales equal to or greater than this price
income = freq * 37.0924
print(f'Income: {income}')

# for maximum price in confidence interval
# Price:38.17576299427283
freq = len(df[df["price"] >= 38.17576299427283])
# number of sales equal to or greater than this price
income = freq * 38.17576299427283
print(f'Income: {income}')


















