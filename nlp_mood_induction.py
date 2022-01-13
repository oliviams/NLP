# import pip
# pip.main(['install', 'requests'])
# pip.main(['install', 'bs4'])
# pip.main(['install', 'nltk'])
# pip.main(['install', 're'])
# pip.main(['install', 'pandas'])
from bs4 import BeautifulSoup
import requests
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

nltk.download('vader_lexicon')


def getdata(url):
    """
    Automatically gets text from given url
    """
    r = requests.get(url)
    return r.text

# READING CSV
df = pd.read_csv('Web-Browsing_Mood_Induction_January+12,+2022_06.47.csv', skipinitialspace=True, usecols=['Q3'])

for i in range(len(df)-2) :
  url_string = df.loc[i+2, 'Q3']
  url_list = url_string.split()
  print(url_list)
  for url in url_list:
      html = getdata(url)
      soup = BeautifulSoup(html, 'html.parser')
      paragraph = []
      for data in soup.find_all("p"):
          paragraph.append(data.get_text())

      print(paragraph)


#
# html = getdata('https://www.tomsguide.com/reviews/iphone-13-pro')
# soup = BeautifulSoup(html, 'html.parser')
#
# paragraph = []
# for data in soup.find_all("p"):
#     paragraph.append(data.get_text())
#
# print(paragraph)
#
# def text_preprocessing(text):  # What other preprocessing is needed? remove punctuation, \n...?
#     text = text.lower() # Are all these necessary?
#     text = re.sub("@\\w+", "", text)
#     text = re.sub("https?://.+", "", text)
#     text = re.sub("\\d+\\w*\\d*", "", text)  # removes numbers, maybe other things?
#     text = re.sub("#\\w+", "", text)
#     return(text)
#
# paragraph = ' '.join(paragraph) # when in a list this had lots of \n which don't get printed, are these going into the sentiment analysis? Do they affect the score?
# paragraph = text_preprocessing(paragraph)
# print(paragraph)
#
# sid = SentimentIntensityAnalyzer()
#
# print(sid.polarity_scores(paragraph))
# print(sid.polarity_scores(paragraph)['compound'])
#
# urls = ['https://www.tomsguide.com/reviews/iphone-13-pro']
# # scores = [sid.polarity_scores(paragraph)['compound']]
# sentiment = [sid.polarity_scores(paragraph)['pos'] - sid.polarity_scores(paragraph)['neg']] # check if this doesn't account for intensity, if not might be better to use compound
#
# scores_df = pd.DataFrame({'URLs': urls, 'Scores': sentiment})
# print(scores_df)



