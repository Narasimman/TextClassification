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
 
tag_to_type = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}
def get_wordnet_pos(treebank_tag):
    return tag_to_type.get(treebank_tag[:1], wordnet.NOUN)

def clean(text):
    clean_text = nonan.sub('',text)
    words = nltk.word_tokenize(shortword.sub('',clean_text.lower()))
    filtered_words = [w for w in words if not w in stops]
    tags = nltk.pos_tag(filtered_words)
    #print tags
    return ' '.join(
        lmtzr.lemmatize(word, get_wordnet_pos(tag[1]))
        for word, tag in zip(filtered_words, tags)
    )

def processdata():
  with open('data/train.txt','r') as f:
    with open('data/corpus.txt','w') as f2:
      text = []
      for line in f:
        text.append(line.split('\t')[1])
      f2.truncate()
      for line in text:
        text = clean(line)
        f2.write(text +'\n')
 
