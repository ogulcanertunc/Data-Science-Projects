### Data Preparation ###
########################

import numpy as np
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.set_option('display.width', 1000)

# Import Data
load_data = pd.read_csv(r"/Data/credit_risk_dataset.csv")

df = load_data.copy()

# Explore Data
df.head(20)
df.nunique()

df.isnull().sum()

df['person_emp_length'].fillna((df['person_emp_length'].median()), inplace=True)
df['loan_int_rate'].fillna((df['loan_int_rate'].median()), inplace = True)

df.info()


loan_data_dummies = [pd.get_dummies(df["person_home_ownership"], prefix = "person_home_ownership", prefix_sep=":"),
                     pd.get_dummies(df["loan_intent"], prefix = "loan_intent", prefix_sep=":"),
                     pd.get_dummies(df["loan_grade"], prefix = "loan_grade", prefix_sep=":"),
                     pd.get_dummies(df["cb_person_default_on_file"], prefix = "cb_person_default_on_file", prefix_sep=":")]

loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)
df_dummy = pd.concat([df, loan_data_dummies], axis = 1)
df_dummy.columns

################
### PD MODEL ###
################

df_dummy["loan_status"].unique()

df_dummy["loan_status"].value_counts()
df_dummy["loan_status"].value_counts()/df_dummy["loan_status"].count()

df_dummy["good_bad"] = np.where(df_dummy["loan_status"].isin([1]), 0,1)
df_dummy["loan_grade"].unique()


df_dummy.groupby(df_dummy.columns.values[5], as_index = False)[df_dummy.columns.values[-1]].count()
df_dummy.groupby(df_dummy.columns.values[5], as_index = False)[df_dummy.columns.values[-1]].mean()

df1 = pd.concat([df_dummy.groupby(df_dummy.columns.values[5], as_index = False)[df_dummy.columns.values[-1]].count(),
                 df_dummy.groupby(df_dummy.columns.values[5], as_index = False)[df_dummy.columns.values[-1]].mean()], axis= 1)

df1 = df1.iloc[:, [0,1,3]]
df1.columns = [df1.columns.values[0], "n_obs", "prop_good"]
df1

df1["prop_n_obs"] = df1["n_obs"]/df1["n_obs"].sum()

df1["n_good"] = df1["prop_good"] * df1["n_obs"]
df1["n_bad"] = (1-df1["prop_good"])*df1["n_obs"]

df1["prop_n_good"] = df1["n_good"]/df1["n_good"].sum()
df1["prop_n_bad"] = df1["n_bad"]/df1["n_bad"].sum()

df1["WoE"] = np.log(df1["prop_n_good"]/df1["prop_n_bad"])
df1



df1 = df1.sort_values(["WoE"])
df1 = df1.reset_index(drop=True)
df1


df1["diff_prop_good"] = df1["prop_good"].diff().abs()
df1["diff_WoE"] = df1["WoE"].diff().abs()

df1


df1["IV"] = (df1["prop_n_good"]-df1["prop_n_bad"])*df1["WoE"]
df1["IV"] = df1["IV"].sum()
df1




### Processing Discrete Variables: Automating Calculations ###
##############################################################
def woe_discrete(df, discrite_variable_name):
    df = pd.concat([df[discrite_variable_name], df["loan_status"]], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0,1,3]]
    df.columns = [df.columns.values[0], "n_obs", "prop_good"]
    df["prop_n_obs"] = df["n_obs"]/df["n_obs"].sum()
    df["n_good"] = df["prop_good"] * df["n_obs"]
    df["n_bad"] = (1-df["prop_good"])*df["n_obs"]
    df["prop_n_good"] = df["n_good"]/df["n_good"].sum()
    df["prop_n_bad"] = df["n_bad"]/df["n_bad"].sum()
    df["WoE"] = np.log(df["prop_n_good"]/df["prop_n_bad"])
    df = df.sort_values(["WoE"])
    df = df.reset_index(drop=True)
    df["diff_prop_good"] = df["prop_good"].diff().abs()
    df["diff_WoE"] = df["WoE"].diff().abs()
    df["IV"] = (df["prop_n_good"]-df["prop_n_bad"])*df["WoE"]
    df["IV"] = df["IV"].sum()

    return df



df_temp = woe_discrete(df_dummy, "loan_grade")

### Processing Discrete Varaibles : Visualizing Results ###
###########################################################
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    x = np.array(df_WoE.iloc[:,0].apply(str))
    y = df_WoE["WoE"]
    plt.figure(figsize = (18, 6))
    plt.plot(x,y, marker="o", linestyle="--", color="k")
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel("Weight of Evidence")
    plt.title(str("Weight of Evidence by " + df_WoE.columns[0]))
    plt.xticks(rotation=rotation_of_x_axis_labels)
    plt.show()

plot_by_woe(df_temp)



df_temp = woe_discrete(df_dummy, "person_home_ownership")
df_temp
plot_by_woe(df_temp) #others and non gonna default

df_dummy.person_home_ownership.value_counts()
df_dummy["person_home_ownership:OTHER_RENT"] =sum([df_dummy["person_home_ownership:RENT"],
                                                     df_dummy["person_home_ownership:OTHER"]])

df_temp = woe_discrete(df_dummy, "loan_intent")
df_temp
plot_by_woe(df_temp)
df_dummy['loan_intent:MEDICAL_HOMEIMPROVEMENT'] = sum([df_dummy['loan_intent:HOMEIMPROVEMENT'], df_dummy['loan_intent:MEDICAL']])

df.info()
df_temp = woe_discrete(df_dummy, "loan_grade")
df_temp
plot_by_woe(df_temp)
df_dummy['loan_grade:A_B_C'] = sum([df_dummy['loan_grade:A'], df_dummy['loan_grade:B'], df_dummy['loan_grade:C']])
df_dummy['loan_grade:D_E_F'] = sum([df_dummy['loan_grade:D'], df_dummy['loan_grade:E'], df_dummy['loan_grade:F']])



### Preprocessing continuous variables ###
##########################################


def woe_ordered_continuous(df, discrite_variable_name):
    df = pd.concat([df[discrite_variable_name], df["loan_status"]], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], "n_obs", "prop_good"]
    df["prop_n_obs"] = df["n_obs"]/df["n_obs"].sum()
    df["n_good"] = df["prop_good"] * df["n_obs"]
    df["n_bad"] = (1-df["prop_good"])*df["n_obs"]
    df["prop_n_good"] = df["n_good"]/df["n_good"].sum()
    df["prop_n_bad"] = df["n_bad"]/df["n_bad"].sum()
    df["WoE"] = np.log(df["prop_n_good"]/df["prop_n_bad"])
    #df = df.sort_values(["WoE"])
    #df = df.reset_index(drop=True)
    df["diff_prop_good"] = df["prop_good"].diff().abs()
    df["diff_WoE"] = df["WoE"].diff().abs()
    df["IV"] = (df["prop_n_good"]-df["prop_n_bad"])*df["WoE"]
    df["IV"] = df["IV"].sum()

    return df

df_dummy["person_emp_length"].unique()
df_dummy["person_emp_length"] =  df_dummy["person_emp_length"].astype('int64')

df_temp = woe_ordered_continuous(df_dummy, "person_emp_length")
plot_by_woe(df_temp)


df_dummy["person_emp_length:0"] = np.where(df_dummy["person_emp_length"].isin([0]),1,0)
df_dummy["person_emp_length:1"] = np.where(df_dummy["person_emp_length"].isin([1]),1,0)
df_dummy["person_emp_length:2-4"] = np.where(df_dummy["person_emp_length"].isin(range(2,5)),1,0)
df_dummy["person_emp_length:5-11"] = np.where(df_dummy["person_emp_length"].isin(range(5,12)),1,0)
df_dummy["person_emp_length:12-15"] = np.where(df_dummy["person_emp_length"].isin(range(12,16)),1,0)
df_dummy["person_emp_length:16-18"] = np.where(df_dummy["person_emp_length"].isin(range(16,19)),1,0)
df_dummy["person_emp_length:19"] = np.where(df_dummy["person_emp_length"].isin([19]),1,0)
df_dummy["person_emp_length:20-21"] = np.where(df_dummy["person_emp_length"].isin(range(20,22)),1,0)
df_dummy["person_emp_length:22"] = np.where(df_dummy["person_emp_length"].isin([22]),1,0)
df_dummy["person_emp_length:23-24"] = np.where(df_dummy["person_emp_length"].isin(range(23,25)),1,0)
df_dummy["person_emp_length:=>24"] = np.where(df_dummy["person_emp_length"].isin(range(23,int(df_dummy["person_emp_length"].max()+1))),1,0)


df_dummy["loan_percent_income"].unique()
df_dummy["loan_percent_income"].max()
df_temp = woe_ordered_continuous(df_dummy, "loan_percent_income")
plot_by_woe(df_temp,45)

df_dummy["loan_percent_income:0-0.15"] = np.where(df_dummy["loan_percent_income"].isin(np.arange(0.0, 0.16, 0.01)),1,0)
df_dummy["loan_percent_income:0.16-0.25"] = np.where(df_dummy["loan_percent_income"].isin(np.arange(0.16, 0.26, 0.01)),1,0)
df_dummy["loan_percent_income:0.25-0.30"] = np.where(df_dummy["loan_percent_income"].isin(np.arange(0.26, 0.31, 0.01)),1,0)
df_dummy["loan_percent_income:0.31-0.50"] = np.where(df_dummy["loan_percent_income"].isin(np.arange(0.31, 0.52, 0.01)),1,0)
df_dummy["loan_percent_income:0.52"] = np.where(df_dummy["loan_percent_income"].isin([0.52]),1,0)
df_dummy["loan_percent_income:0.53-0.54"] = np.where(df_dummy["loan_percent_income"].isin(np.arange(0.53, 0.55, 0.01)),1,0)
df_dummy["loan_percent_income:0.55"] = np.where(df_dummy["loan_percent_income"].isin([0.55]),1,0)
df_dummy["loan_percent_income:0.56-0.57"] = np.where(df_dummy["loan_percent_income"].isin(np.arange(0.56, 0.58, 0.01)),1,0)
df_dummy["loan_percent_income:0.58"] = np.where(df_dummy["loan_percent_income"].isin([0.58]),1,0)
df_dummy["loan_percent_income:>59"] = np.where(df_dummy["loan_percent_income"].isin(np.arange(0.59, 0.85, 0.01)),1,0)



df_dummy["cb_person_cred_hist_length"].unique()
df_dummy["cb_person_cred_hist_length"].value_counts()
df_temp = woe_ordered_continuous(df_dummy, "cb_person_cred_hist_length")
plot_by_woe(df_temp,45)


df_dummy["cb_person_cred_hist_length:2-18"] = np.where(df_dummy["cb_person_cred_hist_length"].isin(range(2,19)),1,0)
df_dummy["cb_person_cred_hist_length:19-22"] = np.where(df_dummy["cb_person_cred_hist_length"].isin(range(19,23)),1,0)
df_dummy["cb_person_cred_hist_length:23"] = np.where(df_dummy["cb_person_cred_hist_length"].isin([23]),1,0)
df_dummy["cb_person_cred_hist_length:24"] = np.where(df_dummy["cb_person_cred_hist_length"].isin([24]),1,0)
df_dummy["cb_person_cred_hist_length:25"] = np.where(df_dummy["cb_person_cred_hist_length"].isin([25]),1,0)
df_dummy["cb_person_cred_hist_length:26"] = np.where(df_dummy["cb_person_cred_hist_length"].isin([26]),1,0)
df_dummy["cb_person_cred_hist_length:27-28"] = np.where(df_dummy["cb_person_cred_hist_length"].isin(range(27,29)),1,0)
df_dummy["cb_person_cred_hist_length:30"] = np.where(df_dummy["cb_person_cred_hist_length"].isin([30]),1,0)

df_temp = woe_ordered_continuous(df_dummy, "person_income")
plot_by_woe(df_temp,45)
df_dummy['person_income:<38.5K'] = np.where((df_dummy['person_income'] <= 38500), 1, 0)
df_dummy['person_income:38.5K-55K'] = np.where((df_dummy['person_income'] > 38500) & (df_dummy['person_income'] <= 55000), 1, 0)
df_dummy['person_income:55K-79.2K'] = np.where((df_dummy['person_income'] > 55000) & (df_dummy['person_income'] <= 79200), 1, 0)
df_dummy['person_income:>79.2K'] = np.where((df_dummy['person_income'] > 79200), 1, 0)



df.info()

df_dummy.columns
ref_categories = ['cb_person_default_on_file', 'person_home_ownership', 'person_home_ownership:RENT','person_home_ownership:OTHER',
                  'loan_intent', 'loan_intent:MEDICAL', 'loan_intent:HOMEIMPROVEMENT','loan_grade',
                  'loan_grade:A','loan_grade:B','loan_grade:C',
                  'loan_grade:D','loan_grade:E','loan_grade:F',
                  'person_emp_length', 'loan_percent_income', 'cb_person_cred_hist_length','person_income', 'good_bad']


df_dummy = df_dummy.drop(ref_categories, axis = 1)
df_dummy.head()




### PD Model Estimation ###
###########################
## Based model ###

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
df_dummy.head()
x_train, x_test, y_train, y_test = train_test_split(df_dummy.drop("loan_status", axis=1), df_dummy["loan_status"], random_state=42,
                                                    test_size=.30)
reg = LogisticRegression()
reg.fit(x_train, y_train)

feature_name = x_train.columns.values
summary_table = pd.DataFrame(columns = ["Feature Name"], data=feature_name)
summary_table["Coefficients"] = np.transpose(reg.coef_)
summary_table.index = summary_table.index+1
summary_table.loc[0] = ["Intercept", reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table.head(50)

### Build a Logistic Regression Model with P-Values ###

# P values for sklearn logistic regression.
# Class to display p-values for logistic regression in sklearn.

from sklearn import linear_model
import scipy.stats as stat


class LogisticRegression_with_p_values:

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)  # ,**args)

    def fit(self, X, y):
        self.model.fit(X, y)

        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)  ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates  # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]  ### two tailed test for p-values

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values


reg = LogisticRegression_with_p_values()
reg.fit(x_train, y_train)

feature_name = x_train.columns.values
summary_table = pd.DataFrame(columns=["Feature Name"], data=feature_name)
summary_table["Coefficients"] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ["Intercept", reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table.head(50)


p_values = reg.p_values

p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table


reg2 = LogisticRegression_with_p_values()
reg2.fit(x_train, y_train)

feature_name = x_train.columns.values
summary_table = pd.DataFrame(columns = ["Feature Name"], data=feature_name)
summary_table["Coefficients"] = np.transpose(reg2.coef_)
summary_table.index = summary_table.index+1
summary_table.loc[0] = ["Intercept", reg2.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table.head(50)

p_values = reg2.p_values

p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table


### Acc ###

y_hat_test = reg2.model.predict(x_test)
y_hat_test
y_hat_test_proba = reg2.model.predict_proba(x_test)
y_hat_test_proba


y_hat_test_proba[:][:,1]

y_hat_test_proba = y_hat_test_proba[:][:,1]
loan_data_targets_test_temp = y_test
loan_data_targets_test_temp.reset_index(drop=True, inplace=True)
df_actual_predicted_probs = pd.concat([loan_data_targets_test_temp, pd.DataFrame(y_hat_test_proba)], axis = 1)
df_actual_predicted_probs.shape
df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']

df_actual_predicted_probs.index = y_test.index
df_actual_predicted_probs.head()




### Accuracy and Area under the Curve ###

tr = 0.5
df_actual_predicted_probs.columns
df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)

pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted'])
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]


pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]
(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]


from sklearn.metrics import roc_curve, roc_auc_score
roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


plt.plot(fpr, tpr)
# We plot the false positive rate along the x-axis and the true positive rate along the y-axis,
# thus plotting the ROC curve.
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
# We plot a seconary diagonal line, with dashed line style and black color.
plt.xlabel('False positive rate')
# We name the x-axis "False positive rate".
plt.ylabel('True positive rate')
# We name the x-axis "True positive rate".
plt.title('ROC curve')
# We name the graph "ROC curve".
plt.show()

AUROC = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC




########### XGB ###########

#Start of gradient boosted tree
from sklearn import model_selection,linear_model, metrics
import re
import xgboost as xgb
regex = re.compile(r"\[|\]|<", re.IGNORECASE)

df_dummy.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df_dummy.columns.values]
x_train, x_test, y_train, y_test = train_test_split(df_dummy.drop("loan_status", axis=1), df_dummy["loan_status"], random_state=42,
                                                    test_size=.30)
xgb_model = xgb.XGBClassifier() # initialize tree

xgb_model.fit(x_train, y_train) # train tree

predict_xgb = xgb_model.predict_proba(x_test) # 1st col = pred val, 2nd col = pred prob

predict_xgb_prob = pd.DataFrame(predict_xgb[:,1],columns = ['Default Probability'])

pd.concat([predict_xgb_prob, y_test.reset_index(drop=True)],axis=1)


round(xgb_model.score(x_test,y_test),3)

# display feature and their importance
feat_imp = xgb_model.get_booster().get_score(importance_type='weight')

feat_imp

set(x_train.columns) - set(feat_imp)

# display top 5 most import features
sorted(feat_imp.items(), key=lambda kv: kv[1],reverse=True)[0:5]
xgb.plot_importance(xgb_model,importance_type='weight')
plt.show()


x_train, x_test, y_train, y_test = model_selection.train_test_split(df_dummy.drop("loan_status", axis=1), df_dummy["loan_status"], random_state=2020, test_size=.30)
#Start of gradient boosted tree
xgb_model = xgb.XGBClassifier() # initialize tree
xgb_model.fit(x_train, np.ravel(y_train)) # train tree
predict_xgb = xgb_model.predict_proba(x_test) # 1st col = pred val, 2nd col = pred prob
predict_xgb_prob = pd.DataFrame(predict_xgb[:,1],columns = ['Default Probability'])
pd.concat([predict_xgb_prob, y_test.reset_index(drop=True)],axis=1)
round(xgb_model.score(x_test,y_test),3)
thresh = np.linspace(0,1,21)
thresh


def find_opt_thresh(predict, thr=thresh, y_true=y_test):
    data = predict
    def_recalls = []
    nondef_recalls = []
    accs = []
    for threshold in thr:
        # predicted values for each threshold
        data['loan_status'] = data['Default Probability'].apply(lambda x: 1 if x > threshold else 0)
        accs.append(metrics.accuracy_score(y_true, data['loan_status']))
        stats = metrics.precision_recall_fscore_support(y_true, data['loan_status'])
        def_recalls.append(stats[1][1])
        nondef_recalls.append(stats[1][0])
    return accs, def_recalls, nondef_recalls

accs, def_recalls, nondef_recalls = find_opt_thresh(predict_xgb_prob)


plt.plot(thresh,def_recalls)
plt.plot(thresh,nondef_recalls)
plt.plot(thresh,accs)
plt.xlabel("Probability Threshold")
plt.xticks(thresh, rotation = 'vertical')
plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])
#plt.axvline(x=0.45, color='pink')
plt.show()

optim_threshold = accs.index(max(accs))

print(round(accs[optim_threshold],3))

thresh[optim_threshold]

def_recalls[optim_threshold]

accs[optim_threshold]
