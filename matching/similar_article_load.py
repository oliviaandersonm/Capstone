#similar_article_load.py
#from requested url, gets similar recent news articles by topic
import sys, os, re

###MOVE
<<<<<<< HEAD
NEWS_API_KEY = 'ec91d31a64444c83a1b3c9e928c341fd'
=======
NEWS_API_KEY = ''
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18

sources = ['abc-news','ars-technica','associated-press','bbc-news','bbc-sport',
           'bloomberg','business-insider','buzzfeed','cbs-news','cnbc','cnn',
           'financial-times','fortune','fox-news','medical-news-today','msnbc',
           'national-geographic','nbc-news','newsweek','new-york-magazine',
           'politico', 'recode','reuters','techcrunch','the-economist',
           'the-new-york-times', 'the-telegraph','the-wall-street-journal',
           'the-washington-post','time','wired']

<<<<<<< HEAD
#get requested article
url = sys.argv[1]

## note: if site is protected (subscription) full article may not be available
from newspaper import Article
article = Article(url)
article.download()
article.parse()
text = article.text
title = article.title

#get keywords from title
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
stop_words = stopwords.words('english')
keywords = title.split(' ')
keywords = [x for x in keywords if x not in stop_words]


#identify and remove source from list
source_str = url.lstrip('http://www.').lstrip('https://www.')
source_str = re.split('\..', source_str)[0]
print(source_str)

from difflib import get_close_matches
try:
    source = get_close_matches(source_str, sources)[0]
    sources.remove(source)
except:
    source = source_str

#query for similar topics from different sources (newsapi)
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
ars = newsapi.get_everything(q=' OR '.join(keywords), sources=','.join(sources),
                            language='en', page_size=25)

#store urls
article_dic = {}
for i in range(25):
    article_dic[ars['articles'][i]['url']] = [ars['articles'][i]['title']]

text_dic = {}
for key in article_dic:
    a = Article(key)
    a.download()
    a.parse()
    text_dic[key] = a

#store in temp file, url, source, date, one sentence per line, end with *****
f = open('temp_sim.txt', 'w')
for key in text_dic:
    f.write(key + '\n' + text_dic[key].title + '\n' + text_dic[key].text + '\n*****\n')
=======
def get_similar_articles(url):
    #get requested article
    ## note: if site is protected (subscription) full article may not be available
    from newspaper import Article
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    title = article.title

    #get keywords from title
    from nltk.corpus import stopwords
    from nltk import download
    download('stopwords')
    stop_words = stopwords.words('english')
    keywords = title.split(' ')
    keywords = [x for x in keywords if x not in stop_words]


    #identify and remove source from list
    source_str = url.lstrip('http://www.').lstrip('https://www.')
    source_str = re.split('\..', source_str)[0]

    from difflib import get_close_matches
    try:
        source = get_close_matches(source_str, sources)[0]
        sources.remove(source)
    except:
        source = source_str

    #query for similar topics from different sources (newsapi)
    from newsapi import NewsApiClient
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    ars = newsapi.get_everything(q=' OR '.join(keywords), sources=','.join(sources),
                                language='en', page_size=25)

    #store urls
    article_dic = {}
    for i in range(25):
        article_dic[ars['articles'][i]['url']] = [ars['articles'][i]['title']]

    text_dic = {}
    for key in article_dic:
        a = Article(key)
        a.download()
        a.parse()
        text_dic[key] = a

    #store in temp file, url, source, date, one sentence per line, end with *****
    f = open('temp_sim.txt', 'w')
    for key in text_dic:
        f.write(key + '\n' + text_dic[key].title + '\n' + text_dic[key].text + '\n*****\n')
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
