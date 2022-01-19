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
import os

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

df = pd.read_csv('Web-Browsing_Mood__Induction_Main_Master.csv', skipinitialspace=True, usecols=['Q3'])
df_prolific = pd.read_csv('Web-Browsing_Mood__Induction_Main_Master.csv', skipinitialspace=True, usecols=['Q158'])

signal.signal(signal.SIGALRM, handler)

participants = []
prolific_id = []
url_full_list = []

score_averages = []
for i in range(len(df)-1):
    os.mkdir('/Users/olivia/Desktop/Rotation 1 - ABL/Chris/Mood induction/' + str(i+1))
    url_string = df.loc[i+1, 'Q3']
    url_list = url_string.split()
    url_full_list.append(url_list)
    participants.append(i+1)
    prolific_id.append(df_prolific.loc[i+1, 'Q158'])
    sentiment_list = []
    for url in url_list:
        signal.alarm(5)
        try:
            html = getdata(url)
            soup = BeautifulSoup(html, 'html.parser')
            paragraph = []
            text_file = open('/Users/olivia/Desktop/Rotation 1 - ABL/Chris/Mood induction/' + str(i + 1) + '/' + str(i + 1) + "URL" + str(url_list.index(url) + 1) + '.txt', 'w')
            for data in soup.find_all("p"):
                sentence = data.get_text()
                text_file.write(sentence)
                sentence = text_preprocessing(sentence)
                text_tokens = word_tokenize(sentence)
                tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
                filtered_sentence = (" ").join(tokens_without_sw)
                paragraph.append(filtered_sentence)
            text_file.close()
            paragraph = ' '.join(paragraph)
            score = sia.polarity_scores(paragraph)['compound']
            sentiment = sia.polarity_scores(paragraph)['pos'] - sia.polarity_scores(paragraph)['neg']
            sentiment_list.append(score)
        except Exception as exc:
            sentiment_list.append(np.nan)
    scores_df = pd.DataFrame({'URL': url_list, 'Score': sentiment_list})
    score_averages.append(scores_df['Score'].mean())
    print(scores_df)
    scores_df.to_excel('/Users/olivia/Desktop/Rotation 1 - ABL/Chris/Mood induction/Session Values/mood_induction_participant_' + str(i+1) + '.xlsx', index=False)

url_df = pd.DataFrame(url_full_list, columns = ['URL' + str(i+1) for i in range(max([len(list) for list in url_full_list]))])
url_df.insert(0, 'Prolific ID', prolific_id)
url_df.insert(0, 'Participant', participants)
url_df.to_excel('mood_induction_urls.xlsx', index=False)

summary_df = pd.DataFrame({'Participant': [i+1 for i in range(len(score_averages))], 'Score Average': score_averages})
summary_df.to_excel('mood_induction_nltk.xlsx', index=False)

