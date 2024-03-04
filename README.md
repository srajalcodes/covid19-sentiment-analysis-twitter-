Twitter Sentiment Analysis on COVID-19 Tweets


This project aims to analyze the sentiment of tweets related to COVID-19 using Python. It fetches tweets through the Twitter API, preprocesses the text data, performs sentiment analysis, and visualizes the results. The analysis helps in understanding public sentiment towards COVID-19, which can be useful for researchers, policymakers, and the general public.


Features

Data Collection: Fetches tweets using Twitter API based on a specific query (covid19).

Data Preprocessing: Cleans the tweets by removing special characters, links, and non-alphabetic characters.

Sentiment Analysis: Determines the sentiment (positive, negative, neutral) of each tweet.

Data Visualization: Visualizes the distribution of sentiments and generates a word cloud to highlight the most frequent words in the tweets.

Machine Learning Preparation: Prepares the dataset for machine learning models by converting text to numerical values using TF-IDF vectorization.


Installation
To run this project, you need Python 3.x and the following Python libraries installed:

tweepy
pandas
matplotlib
scikit-learn
nltk
wordcloud
textblob

You can install these packages using pip:

bash
Copy code
pip install tweepy pandas matplotlib scikit-learn nltk wordcloud textblob
Usage
Before running the script, you need to obtain Twitter API credentials by creating a Twitter developer account and creating an application. Once you have your credentials, replace the placeholders in the script with your own consumer_key, consumer_secret, access_token, and access_token_secret.

To run the script, simply execute the Python file in your terminal:

bash
Copy code
python twitter_sentiment_analysis.py
Contributing
Contributions to this project are welcome. Please follow these steps to contribute:

Fork the repository.
Create a new branch (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.
License
Distributed under the MIT License. See LICENSE for more information.

Acknowledgements
This project uses the Tweepy library to interact with the Twitter API.
Sentiment analysis is performed using the TextBlob library.
