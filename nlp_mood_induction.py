from bs4 import BeautifulSoup
import requests
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import openpyxl
import time
import signal

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def getdata(url):
    """
    Gets text from given url
    """
    r = requests.get(url)
    return r.text

def text_preprocessing(text):
    text = text.lower()
    text = re.sub("@\\w+", "", text)
    text = re.sub("https?://.+", "", text)
    text = re.sub("\\d+\\w*\\d*", "", text)
    text = re.sub("#\\w+", "", text)
    return(text)

def handler(signum, frame):
    """
    Used to skip loop if webpage has crawlers blocked
    """
    raise Exception("end of time")

df = pd.read_csv('Web-Browsing_Mood_Induction_January+12,+2022_06.47.csv', skipinitialspace=True, usecols=['Q3'])

signal.signal(signal.SIGALRM, handler)
signal.alarm(5)

score_averages = []
for i in range(len(df)-2):
    url_string = df.loc[i+2, 'Q3']
    url_list = url_string.split()
    sentiment_list = []
    for url in url_list:
        try:
            html = getdata(url)
            soup = BeautifulSoup(html, 'html.parser')
            paragraph = []
            for data in soup.find_all("p"):
                sentence = data.get_text()
                sentence = text_preprocessing(sentence)
                text_tokens = word_tokenize(sentence)
                tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
                filtered_sentence = (" ").join(tokens_without_sw)
                paragraph.append(filtered_sentence)
            # print(paragraph)
            paragraph = ' '.join(paragraph)
            score = sia.polarity_scores(paragraph)['compound']
            # sentiment = sia.polarity_scores(paragraph)['pos'] - sia.polarity_scores(paragraph)['neg']
            sentiment_list.append(score)
        except Exception as exc:
            sentiment_list.append(np.nan)
    scores_df = pd.DataFrame({'URL': url_list, 'Score': sentiment_list})
    score_averages.append(scores_df['Score'].mean())
    print(scores_df)

summary_df = pd.DataFrame({'Participant': [i+1 for i in range(len(score_averages))], 'Score Average': score_averages})
summary_df.to_excel('mood_induction_nltk.xlsx')

