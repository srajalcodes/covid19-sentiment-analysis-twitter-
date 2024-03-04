#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
from utils import clean_text, get_lower, get_subjectivity, get_polarity, get_analysis
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from wordcloud import WordCloud

# Twitter API credentials
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define search query parameters
query = "covid19"
tweetCount = 5000
language = "en"

# Fetch tweets using the Twitter API
results = api.search(q=query, language=language, count=tweetCount)

# Create a DataFrame to store tweets
df = pd.DataFrame([tweet.text for tweet in results], columns=['Tweets'])

# Remove duplicate tweets
df.drop_duplicates(subset='Tweets', inplace=True)

# Preprocess tweets
df['Tweets'] = df['Tweets'].apply(clean_text)
df['Tweets'] = df['Tweets'].apply(get_lower)

# Perform sentiment analysis
df['Subjectivity'] = df['Tweets'].apply(get_subjectivity)
df['Polarity'] = df['Tweets'].apply(get_polarity)
df['Analysis'] = df['Polarity'].apply(get_analysis)

# Calculate percentage of positive and negative tweets
ptweets = df[df.Analysis == "Positive"]
percentage_positive = ptweets.shape[0] / df.shape[0]

ntweets = df[df.Analysis == "Negative"]
percentage_negative = ntweets.shape[0] / df.shape[0]

# Visualize sentiment analysis
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='bar')
plt.show()

# Generate word cloud
allwords = ''.join([twts for twts in df['Tweets']])
wordcloud = WordCloud(width=500, height=300, random_state=21, max_font_size=119).generate(allwords)

plt.imshow(wordcloud, 'gray')
plt.axis("off")
plt.show()

# Convert sentiment labels to numerical values
df['Analysis'].replace(['Negative', 'Positive'], [0, 1], inplace=True)

# Prepare data for machine learning
xTrain, xTest, yTrain, yTest = train_test_split(df['Tweets'], df['Analysis'], test_size=0.3, random_state=0)

# Vectorize text data using TF-IDF
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['Tweets'])
Train_X_Tfidf = Tfidf_vect.transform(xTrain)
Test_X_Tfidf = Tfidf_vect.transform(xTest)

# Print the TF-IDF transformed data
print(Train_X_Tfidf)


# In[ ]:




