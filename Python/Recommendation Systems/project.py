### Hybrid Recommender System ###
#################################

##################
### USER BASED ###
##################
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie = pd.read_csv('Lectures/Week 10/Dosyalar/movie.csv')
rating = pd.read_csv('Lectures/Week 10/Dosyalar/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
df['title'] = df['title'].apply(lambda x: x.strip())
a = pd.DataFrame(df["title"].value_counts())
rare_movies = a[a["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
df_ = user_movie_df.copy()
df.head()
df.shape
df.columns
### 1st Step Select Randomly a user ###
#######################################
#random_user = int(pd.Series(df.index).sample(1, random_state=45).values)
# random user 108170
user_id = 108170

### 2nd Step Detection of movies which are watched ###
######################################################
user_id_df = user_movie_df[user_movie_df.index == user_id]
user_id_df.head()

movies_watched = user_id_df.columns[user_id_df.notna().any()].tolist()
len(movies_watched)
movies_watched[:10]

### 3rd Step Accesing ids which are correspond who watched same movies ###
##########################################################################

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


### 4th Step Detecting users which are similar to our random person ###
#######################################################################

# We have 3 steps in that stage
## 1. We merge our random user and similar users data
## 2. We will create correlation dataframe
## 3. We will select top similarity users from that dataframe

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)], user_id_df[movies_watched]])
final_df.head()
final_df.shape
final_df.T.corr()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()
# Let's choose the closest users with high correlation
top_users = corr_df[(corr_df["user_id_1"] == user_id) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]]\
    .reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.head()

rating = pd.read_csv('Lectures/Week 10/Dosyalar/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')



### 5th Step Calculation of Weighted Average Recommendation Score ###
#################################################
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
temp.columns = ['sum_corr', 'sum_weighted_rating']
temp.head()
recommendation_df = pd.DataFrame()
recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
recommendation_df['movieId'] = temp.index
recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
recommendation_df.head(30)

movie = pd.read_csv('Lectures/Week 10/Dosyalar/movie.csv')
movies_from_user_based = movie.loc[movie['movieId'].isin(recommendation_df['movieId'].head(10))]['title']
movies_from_user_based.head(30)
movies_from_user_based[:5].values


### 6th Show 5 movies by user based recommendation  ###
#######################################################

# show first 5 recommended movies
movie = pd.read_csv('Lectures/Week 10/Dosyalar/movie.csv')
movies_user_based = movie.loc[movie['movieId'].isin(recommendation_df['movieId'].head(10))]['title']
movies_user_based[:5].values

##################
### Item Based ###
##################

### 1st Step Making a Recommendation  by the title of the most recent highest rated movie ###
#############################################################################################
movie = pd.read_csv('Lectures/Week 10/Dosyalar/movie.csv')
rating = pd.read_csv('Lectures/Week 10/Dosyalar/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

### 2nd Step Making changes to the title variable ###
#####################################################

df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False)
df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False)
df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
df['title'] = df['title'].apply(lambda x: x.strip())

df.shape

### 3rd Step Remove the parting line in the genre variable ###
##############################################################

df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])
df.drop("genres", inplace=True, axis=1)
df.head()

### 4th Step Timestamp ###
##########################
df.dtypes

# in timestamp variable we have "1999-12-11 13:36:47" type data,
# which mean Year-Month-Day  Hour:Minutes:Seconds
# In our case we need just year-month-day so we convert it to this format
# and change the type of data
df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')
df.info()

# Seperate the timestamp variable to year, month and day
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df.head()

### 5th Step Creating User Movie Df ###
#######################################

df["title"].nunique()
# unique title num is 26213
a = pd.DataFrame(df["title"].value_counts())
a.head()

rare_movies = a[a["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()

item_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
item_movie_df.shape
user_movie_df.head(10)
item_movie_df.columns

len(item_movie_df.columns)
common_movies["title"].nunique()


### 6th Step Item-Based Recommendation based on Correlation ###
###############################################################

movieId = rating[(rating["rating"] == 5.0) & (rating["userId"] ==user_id)].sort_values(by="timestamp",ascending=False)["movieId"][0:1].values[0]
movie_title = movie[movie["movieId"] == movieId]["title"].str.replace('(\(\d\d\d\d\))', '').str.strip().values[0]

movie = item_movie_df[movie_title]
movie_item_based = item_movie_df.corrwith(movie).sort_values(ascending=False)
movie_item_based[1:6].index


#############################################################################
#
##############################################################################

data_user_item = pd.DataFrame()
data_user_item["userbased_recommendations"] = movies_from_user_based[:5].values
data_user_item["itembased_recommendations"] = movie_item_based[:5].index
data_user_item
