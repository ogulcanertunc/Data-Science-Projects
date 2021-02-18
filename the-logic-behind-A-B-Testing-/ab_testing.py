import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.help_ogi_intro import basic_analysis, outlier_thresholds,ogi_AB, is_any_outlier
import warnings
warnings.filterwarnings("ignore")
import statistics
from scipy import stats
from scipy.stats import shapiro
import statsmodels.stats.api as sms


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

control_data = pd.read_excel("Lectures/Week 5/Dosyalar/ab_testing_data.xlsx", sheet_name="Control Group")
test_data = pd.read_excel("Lectures/Week 5/Dosyalar/ab_testing_data.xlsx", sheet_name="Test Group")

control_data["Group"] = "A"
test_data["Group"] = "B"

is_any_outlier(control_data, "Purchase")
is_any_outlier(test_data, "Purchase")
control_data.head()
test_data.head()


plt.title("Purchase distribution of Test and Control group")
plt.xlabel('Purchase')
plt.hist(control_data['Purchase'],bins=15, alpha=0.7, label='Control Group')
plt.hist(test_data['Purchase'],bins=15, alpha=0.7, label='Test Group')
plt.legend(loc='upper right')
plt.show()


##############################
### 1. Confidence Interval ###
##############################

sms.DescrStatsW(control_data["Purchase"]).tconfint_mean()
sms.DescrStatsW(test_data["Purchase"]).tconfint_mean()

#####################
### 2. AB Testing ###
#####################
#-----------------------------#
###############################
### 2.1. Assumption Control ###
###############################
#--------------------------------#
##################################
#>>>  2.1.1 Normality assumption #
##################################

# H0: Normal distribution assumption is provided.
# H1: Normal distribution assumption cannot be achieved.
AB_test = control_data.append(test_data)
test_statistics, pvalue = shapiro(AB_test.loc[AB_test["Group"] == "A", "Purchase"])
print('Test Statistics is  %.4f, p-value = %.4f' % (test_statistics, pvalue))
#Test Statistics = 0.9773, p-value = 0.5891

# If the p-value < 0.05, H0 is rejected.
# If the p-value > 0.05, H0 can not be rejected.


test_statistics, pvalue = shapiro(AB_test.loc[AB_test["Group"] == "B", "Purchase"])
print('Test Statistics is  %.4f, p-value = %.4f' % (test_statistics, pvalue))
# Test Statistics = 0.9589, p-value = 0.1541


# In our A and B groups, H0 was not rejected because our p-value was not less than 0.05.
# Therefore, the assumption of normal distribution is provided.

#################################
#>>> 2.1.2 Variance Homogeneity #
#################################

# H0: Variances Are Homogeneous
# H1: Variances Are Not Homogeneous


stats.levene(AB_test.loc[AB_test["Group"] == "A", "Purchase"],
             AB_test.loc[AB_test["Group"] == "B", "Purchase"])

#LeveneResult(statistic=2.6392694728747363, pvalue=0.10828588271874791)
# H0 was not rejected because the p-value was not less than 0.05.
# Variances are homogeneous.

########################################
### 2. Application of the Hypothesis ###
########################################

# Independent two-sample t test if assumptions are provided (parametric test)
# H0: M1 = M2 (There is no statistically significant difference between the two group averages.)

# H1: M1! = M2 (There is a statistically significant difference between the two group averages)

# H0 Rejected if p-value <0.05.
# If p-value> 0.05, H0 Cannot be denied.

test_statistics, pvalue = stats.ttest_ind(AB_test.loc[AB_test["Group"] == "A", "Purchase"],
                                           AB_test.loc[AB_test["Group"] == "B", "Purchase"],
                                           equal_var=True)
print('Test Statistics is  %.4f, p-value is %.4f' % (test_statistics, pvalue))

# Test Statistics = -0.9416, p-value = 0.3493
# H0 was not rejected because the p value was not less than 0.05.

############SORU2############
# No statistically significant difference was observed between the means of the two groups.

############SORU3############
# I did the shapiro test for the assumption of normality and saw that there was a normal distribution.
# I used the levene test for variance homogeneity, and observed variance homogeneity.
# I used the t-test after checking the assumptions.

############SORU4############
# There is no difference between the two, so Test Group can be recommended by looking at the confidence interval. Or,
# the number of observations can be increased for this hypothesis test.

##################
### Bonus Part ###
##################
# Click Through Rate (CTR)
# It is the ratio of users who visit the website, see the ad and click the ad.
# Clicks / Impression

control_data['purchase_per_click'] = control_data["Click"]/control_data["Impression"]
test_data['purchase_per_click'] = test_data["Click"]/test_data["Impression"]

# Group A has a higher value compared to the overall rates seen for now.
# At first glance, we see that the click-through rate is in favor of the control group. So while the ad is showing,
# the rate of site visitors clicking seems to be better in the current system.

# But this will be a simple view, will there be a statistically significant difference here,
# so we need to do our AB test.

ogi_AB(control_data,test_data, "purchase_per_click")
# When the ad bid methods are examined, the click effect of these methods on the site visitors is different.
# And this difference is in favor of the current advertising bid method.