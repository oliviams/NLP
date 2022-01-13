from bs4 import BeautifulSoup
import requests
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def getdata(url):
    """
    Automatically gets text from given url
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

df = pd.read_csv('Web-Browsing_Mood_Induction_January+12,+2022_06.47.csv', skipinitialspace=True, usecols=['Q3'])

score_averages = []
for i in range(len(df)-2) :
  url_string = df.loc[i+2, 'Q3']
  url_list = url_string.split()
  print(url_list)
  sentiment_list = []
  for url in url_list:
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

      scores = sia.polarity_scores(paragraph)['compound']
      # sentiment = sia.polarity_scores(paragraph)['pos'] - sia.polarity_scores(paragraph)['neg']
      sentiment_list.append(scores)
  scores_df = pd.DataFrame({'URL': url_list, 'Score': sentiment_list})
  print(scores_df)
  score_averages.append(scores_df[['Score']].mean(axis=1))

summary_df = pd.DataFrame({'Participant': [i+1 for i in len(score_averages)], 'Score Average': score_averages})
summary_df.to_excel('mood_induction_nltk.xlsx')




