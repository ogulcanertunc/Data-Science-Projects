import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import statistics
from scipy.stats import shapiro, mannwhitneyu
from scipy import stats
import statsmodels.stats.api as sms



def basic_analysis(dataframe):
    print("Lets start with info")
    print(dataframe.info())
    print("##########################Num of total NAs################################", "\n\n")
    print(dataframe.isnull().sum())
    print("######################Dataframe's Describe################################", "\n\n")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)
    print("######################Variables Distribution################################", "\n\n")
    for col_name in dataframe.columns:
        plt.title("Histogram for " + col_name)
        sns.distplot(dataframe[col_name], color="skyblue")
        plt.show()

    for col_name in dataframe.columns:
        plt.title("Box Plot for " + col_name)
        dataframe.boxplot(column=[col_name], return_type=None)
        plt.show()

def outlier_thresholds(dataframe, variable, low_quantile=0.01, up_quantile=0.99):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit



def ogi_AB(a_dataframe, b_dataframe, variable):

    plt.title("Distribution of Test and Control group")
    plt.xlabel(variable)
    plt.hist(a_dataframe[variable], bins=15, alpha=0.7, label='Control Group')
    plt.hist(b_dataframe[variable], bins=15, alpha=0.7, label='Test Group')
    plt.legend(loc='upper right')
    plt.show()

    data_a_w, data_a_p = shapiro(a_dataframe[variable])
    data_b_w, data_b_p = shapiro(b_dataframe[variable])
    print("P value for A data = %.3f, P value for B data = %.3f" % (data_a_p, data_b_p), "\n")
    print("##################################################")
    print("With that situations we have to check our methods","\n")

    def ab_test():
        if data_a_p < 0.05 and data_b_p < 0.05:
            print("Our p values obtained from the Shapiro Wilk test were less than 5%. This shows that we should reject H0. "
                  "Accordingly distribution of " , variable , " values in" , " A and B are not likely to normal distribution.","\n\n")
            return str("both_less")

        elif data_a_p > 0.05 and data_b_p > 0.05:
            print("Our p values obtained from the Shapiro Wilk test were greater than 5% (0.05). This shows that we should not reject H0. "
                      "Accordingly distribution of " , variable + " values in" , "  A and B are likely to normal distribution.", "\n\n")
            return str("both_great")

        elif data_a_p <0.05 and data_b_p > 0.05:
            print("Shapiro Wilk Test, p is less than 5% for A, but p value for B is greater than 5%. This shows that the H0 hypothesis for A is rejected."
                  "Accordingly you can check if the " , variable, " values in" , " a contain outlier or not", "\n\n")
            return str("a_less_b_grt")

        else:
            print("Shapiro Wilk Test, p is less than 5% for B, but p value for A is greater than 5%. This shows that the H0 hypothesis for B is rejected. "
                  "Accordingly you can check if the " , variable, " values in" ," GroupB contain outlier or not", "\n\n")
            return str("a_grt_b_less")

    p_result = ab_test()
    if p_result == "both_great":
        print("Lets test our Homogeneity")
        print("Homogeneity Assumption for " + variable,  "\n\n")
        levene_F, levene_p = stats.levene(a_dataframe[variable], b_dataframe[variable])
        print("levene F value is %.3f, levene p value is %.3f" % (levene_F, levene_p), "\n\n")
        if levene_p > 0.05:
            print("When we did the Levene's Test of Homogeneity, we saw our p value greater than 5%, which shows that we cannot reject H0. Accordingly, the variances of A and B are equal.", "\n\n\n")

            print("Independent Samples T Test for " , variable , "\n\n")
            t_value, t_test_p = stats.ttest_ind(a_dataframe[variable], b_dataframe[variable], equal_var=True)
            print("t_value is %.3f, t_test_p is %.3f" % (t_value, t_test_p), "\n\n")

            if t_test_p > 0.05:
                print("We see that H0 cannot be rejected if the p value we got from the T Test is greater than 5% (0.05)."
                      " Thus, with this result, we can say that there is no significant difference between A and B in " , variable , " variable.", "\n\n")
            else:
                print(" With T Test we see that the p value is less than 5%, which shows us that H0 is rejected."
                      " Thus, with this result, we can say that there is significant difference between A and B in " ,variable, " variable.", "\n\n")
                if statistics.mean(a_dataframe["Earning"]) > statistics.mean(b_dataframe["Earning"]):
                    print("Mean of A in " ,variable,  " is greater than B", "\n\n")
                else:
                    print("Mean of B in " ,variable, " is greater than A", "\n\n")
        else:
            print("When we did the Levene's Test of Homogeneity, we saw our p value less than 5%, which shows that we can reject H0. Accordingly, the variances of A and B are different",
                  "\n\n")
            print("Independent Samples t Test for " , variable , "\n\n")
            t_value, t_test_p = stats.ttest_ind(a_dataframe[variable], b_dataframe[variable], equal_var=False)
            print('t_value is %.3f, t_test_p is %.3f' % (t_value, t_test_p), "\n\n")

            if t_test_p > 0.05:
                print("We see that H0 cannot be rejected if the p value we got from the T Test is greater than 5% (0.05)."
                      " Thus, with this result, we can say that there is no significant difference between A and B in " ,variable," variable.", "\n\n")

            else:
                print("With T Test we see that the p value is less than 5%, which shows us that H0 is rejected."
                      " Thus, with this result, we can say that there is significant difference between A and B in" ,variable, " variable.", "\n\n")
                if statistics.mean(a_dataframe[variable]) > statistics.mean(b_dataframe[variable]):
                    print("Mean of A in " ,variable , " is greater than GroupB", "\n\n")
                else:
                    print("Mean of B in " , variable , " is greater than A", "\n\n")


    elif p_result == "both_less" or p_result == "a_less_b_grt" or p_result == "a_grt_b_less":
        print("Since our Shapiro Wilk test results we need to apply MannWhitney U Test for ", variable, "\n\n")

        u_value, mannwhit_test_p = mannwhitneyu(a_dataframe[variable], b_dataframe[variable])
        print('U value is %.3f, MannWhitney U Test p value is %.3f' % (u_value, mannwhit_test_p), "\n\n")
        if mannwhit_test_p > 0.05:
            print("Our p value from the Mann Whitney U Test is higher than 5%. This shows that we cannot reject H0."
                  " Thus, with this result, we can say that there is no significant difference between A and B in  " ,variable, " variable.", "\n\n")

        else:
            print("Our p value from the Mann Whitney U Test is less than 5%(0.05). This shows that we can reject H0."
                  "  Thus, with this result, we can say that there is a significant difference between A and B in " + variable+ " variable.", "\n\n")

            if statistics.median(a_dataframe["Earning"]) > statistics.median(b_dataframe["Earning"]):
                print("Median of A Earning in ", variable, " is greater than B Earning", "\n\n")
            else:
                print("Median of B Earning in ", variable,  " is greater than A Earning", "\n\n")
    else:
        print("Something might have gone wrong, do you want to review the data?")

def is_any_outlier(dataframe, variable):
    low, upper= outlier_thresholds(dataframe, variable)
    dataframe_out = dataframe[~((dataframe[variable] < low) |(dataframe[variable] > upper))]
    dataframe.shape[0] == dataframe_out.shape[0]
    if dataframe.shape[0] == dataframe_out.shape[0]:
        print("Dataset has no outliers.")
    else:
        print("Dataset has outliers.")