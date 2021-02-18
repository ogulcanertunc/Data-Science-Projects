##
# Level Based Müşteri Tanımı, Basit Segmentasyon ve Kural Tabanlı Sınıflandırma
##

import pandas as pd
users = pd.read_csv('Lectures/Week 2/Dosyalar/users.csv')
purchases = pd.read_csv('Lectures/Week 2/Dosyalar/purchases.csv')

df = purchases.merge(users, how = "inner", on = "uid")

tot_price = df.groupby(["country", "device", "gender", "age"]).agg({"price" : sum})
tot_price.head()

# sorted tot_price df

agg_df = tot_price.sort_values(by = "price", ascending = False)
agg_df.head()

# change indexes' names
agg_df = agg_df.reset_index()
agg_df.head()

# adding age_cat
agg_df["age_cat"] = pd.cut(agg_df["age"], bins=[0, 18, 23, 30, 40, 75], labels=['0_18', '19_23', '24_30', '31_40', '41_75'])
agg_df.head()

# customer_level_based
agg_df["customer_level_based"] = [row[0] + "_" + row[1].upper() + "_" + row[2] + "_" + row[5] for row in agg_df.values]
agg_df.head()

agg_df = agg_df[["customer_level_based","price"]]
agg_df.head()

agg_df = agg_df.groupby("customer_level_based").agg({"price":"mean"})
#agg_df = agg_df.groupby("customer_level_based").agg({"price":["mean","max","min"]})
agg_df.reset_index(inplace = True)

agg_df["segment"] = pd.qcut(agg_df["price"], 4 ,labels = ["D","C", "B", "A"])
agg_df.groupby("segment").agg({"price" : "mean"}).reset_index(inplace = True)
agg_df.head()


# find turkish ios female 41-75
new_user = "TUR_IOS_F_41_75"

new_customer = agg_df[agg_df["customer_level_based"] == new_user]
new_customer










