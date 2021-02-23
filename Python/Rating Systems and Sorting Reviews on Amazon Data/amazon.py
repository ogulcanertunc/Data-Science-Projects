#http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics.json.gz

import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
##############
### Task 1 ###
##############

### First Step of Task 1: reading data ###
df_ = pd.read_csv('Lectures/Week 6/Dosyalar/df_sub.csv')
df = df_.copy()
df.shape
df.head()

df["reviewTime"] = pd.to_datetime(df["reviewTime"], dayfirst=True)
current_date = pd.to_datetime("2021-02-12")
df["days"] = (current_date - df["reviewTime"]).dt.days
df.head()

### Second Step of Task 1: Average Point of Product ###
df["overall"].mean()

df.head()
df["asin"].value_counts()

### Third Step of Task 1: Calculate weighted average point of product by date ###
a = df["days"].quantile(0.25)
b = df["days"].quantile(0.50)
c = df["days"].quantile(0.75)

### Forth Step of Task 1: Calculating weighted average with previous steps a,b,c ###
df.loc[df["days"] <= a, "overall"].mean() * 28 / 100 + \
    df.loc[(df["days"] > a) & (df["days"] <= b), "overall"].mean() * 26 / 100 + \
    df.loc[(df["days"] > b) & (df["days"] <= c), "overall"].mean() * 24 / 100 + \
    df.loc[(df["days"] > c), "overall"].mean() * 22 / 100

# Review: at the previous step rating waas 4.58 , but with date weighted rating is  4.5955 ~ 4.60

##############
### Task 2 ###
##############

df["helpful"].value_counts()
new_features = df["helpful"].str.split(",",expand=True)

### First step of task 2: Create new "meaningful" variables from helpful variable ###
new_features = new_features.astype("string")
helpful_yes = new_features[0].str.lstrip("[")
helpful_yes = helpful_yes.astype("int64")

total_vote = new_features[1].str.rstrip("]")
total_vote = total_vote.astype("int64")

helpful_no = total_vote - helpful_yes
helpful_no

df["helpful_yes"] = helpful_yes
df["helpful_no"] = helpful_no
df["total_vote"] = total_vote

### Second step of Task 2: create score_pos_neg_diff scores ###

def score_pos_neg_diff(pos, neg):
    return pos - neg

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                 x["helpful_no"]),
                                                                   axis=1)

### Third Step of Task 2: Create a core_average_rating variable, using with average_rating function ###

def score_average_rating(pos, neg):
    if pos - neg == 0:
        return 0
    return pos / (pos + neg)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]),
                                                                     axis=1)

df.head()

### Fourth Step of Task 2: create new scores with wilson lower bound rule, using with wilson_lower_bound ###

def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    import scipy.stats as st
    import math
    n = pos + neg
    if (pos-neg) == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]),axis=1)


df["total_score"] = (df["score_average_rating"] * 40 / 100 +
                     df["wilson_lower_bound"] * 60 / 100)

df["score_average_rating"].sort_values(ascending=False).head()
df["wilson_lower_bound"].sort_values(ascending=False).head()
df["score_pos_neg_diff"].sort_values(ascending=False)
#################################################################################
### Final step of Task 2: Set 20 comments to be displayed on the product page ###
#################################################################################
df["helpful"][df["total_score"].sort_values(ascending=False).head(10).index]

df.columns


# Comments: So if we calculate scores with just wilson lower bound, we miss some parts especially social proof comments,
# from valuable users. So to avoid it, I  created new total score variable, I combined average rating, wilson lower bound,
# pos ne differences and brp score and multiply with special weights. So in my opinion, statistically best comments are these.


#############
### Bonus ###
#############

### Step 1: Calculate the reviews' sentiment scores ###

# if also calculate sentiment scores of the comments and multiply it with our total_score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import nltk
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

sns.set(style="whitegrid", palette = "muted", font_scale = 1.2)
happy_colors_palette = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(happy_colors_palette))

#df = df_.copy()

sns.countplot(df.overall)
plt.xlabel('review score')
plt.show()

def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  elif rating == 3:
    return 1
  else:
    return 2

df['sentiment'] = df.overall.apply(to_sentiment)
class_names = ['negative', 'neutral', 'positive']

ax = sns.countplot(df.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names)
plt.show()

df['reviewText_'] = df['reviewText'].astype(str)
# To lower case everything
df['reviewText_'] = df['reviewText_'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# remove punctuation
df['reviewText_'] = df['reviewText_'].str.replace('[^\w\s]','')

# remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['reviewText_'] = df['reviewText_'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

most = pd.Series(' '.join(df['reviewText_']).split()).value_counts()[:10]
most

most = list(most.index)
df['reviewText_'] = df['reviewText_'].apply(lambda x: " ".join(x for x in x.split() if x not in most))
df['reviewText_'].head()

TextBlob(df['reviewText_'][1]).words

from nltk.stem import PorterStemmer
st = PorterStemmer()
df['reviewText_'] = df['reviewText_'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#make wordcoud

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(col, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=500,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(str(col))

    fig = plt.figure(1, figsize=(14, 14))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

if __name__ == '__main__':

    show_wordcloud(df['reviewText_'])


# sentiment scores
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity


df['polarity'] = df['reviewText_'].apply(pol)
df['subjectivity'] = df['reviewText_'].apply(sub)
df.head()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[['polarity']])
df['polarity_01'] = scaler.transform(df[['polarity']])


df["total_score_"] = (df["score_average_rating"] * 35 / 100 +
                     df["wilson_lower_bound"] * 55 / 100 +
                     df["polarity_01"] * 10/100)

df["reviewText"][df["total_score_"].sort_values(ascending=False).head(20).index]
df[['reviewText','overall', 'helpful_yes','helpful_no', 'score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound','total_score']].sort_values('total_score', ascending=False).head(20)






