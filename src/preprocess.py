import re, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)
from gensim import corpora, models, similarities, matutils


lmtzr = WordNetLemmatizer()
stops = stopwords.words('english')
nonan = re.compile(r'[^a-zA-Z ]')
shortword = re.compile(r'\W*\b\w{1,2}\b')
 
def clean(text):
    clean_text = nonan.sub('',text)
    words = nltk.word_tokenize(shortword.sub('',clean_text.lower()))
    filtered_words = [w for w in words if not w in stops]
    tags = nltk.pos_tag(filtered_words)
    #print tags

    cleaned = " "
    for word, tag in zip(filtered_words, tags):
      if tag[1] == 'NN' or tag[1] == 'NNS':
        cleaned = cleaned + lmtzr.lemmatize(word) + " "
    return cleaned

def processdata():
  with open('data/train8.txt','r') as f:
    with open('data/corpus.txt','w') as f2:
      text = []
      for line in f:
        text.append(line.split('\t')[1])
      f2.truncate()
      for line in text:
        text = clean(line)
        f2.write(text +'\n')
 
