import re, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import logging
logging.basicConfig(filename='data/lda.log', format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)
from gensim import corpora, models, similarities, matutils
import numpy as np
import scipy.stats as stats
from Corpus import Corpus


nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


import preprocess
'''
An LDA model implementation that considers the POS tag of the sentence and 
build lda model for the cluster of words that are tagged as NN/NNS
'''
class TagLDA():

  '''
  Call the preprocessor to get the data and clean it up
  '''
  def process(self):
    preprocess.processdata()

  def filter(self):
    once_ids = [tokenid for tokenid, docfreq in 
      self.dictionary.dfs.iteritems() if docfreq == 1]
    self.dictionary.filter_tokens(once_ids)
    self.dictionary.filter_extremes()
    self.dictionary.compactify()

  '''
  Creates the dictionary of all the cleaned and processed words in the documents
  '''
  def create_dictionary(self):
    self.dictionary = corpora.Dictionary(line.lower().split() for 
      line in open('data/corpus.txt','rb'))
     
  '''
  Creates a corpus with the given dictionary and the bag of words of the training set
  '''
  def create_corpus(self):
    self.corpus = Corpus(self.dictionary)
    #print list(self.corpus)

  '''
  The main function that runs the lda model on the corpus
  '''
  def run(self,min_topics=5,max_topics=20,step=1):
      self.model = models.ldamodel.LdaModel(corpus=self.corpus, 
        id2word=self.dictionary,num_topics=max_topics, chunksize=700, alpha='auto', passes=10)
        
      #Document-topic matrix
      lda_topics = self.model[self.corpus]
      print "lda done!"
        
  def savetopics(self, i):
    topwords = self.model.show_topics(20)
    print topwords

    with open('data/final_topics_' + str(i) + '.txt', 'w') as f:
      for word in topwords:
        f.write(str(word[0]) + ":" + word[1] + "\n")

if __name__ == "__main__":
  lda = TagLDA()
  lda.process()
  lda.create_dictionary()
  lda.create_corpus()
  kl = lda.run(max_topics=10)
  model = lda.model
  model.save('data/lda.model')
  model =  models.LdaModel.load('data/lda.model')
  lda.model = model
  #print list(model[lda.corpus])
  lda.savetopics(00)
  print "\n\n\n\n\n"

  
  with open("test.txt", "w") as f:
    for i in model.show_topics(num_topics=10, num_words=len(lda.dictionary), formatted=False):
      for pair in i[1]:
        f.write(pair[0] + " " + str(pair[1]) + "\n")
      f.write("\n")


