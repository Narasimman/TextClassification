import re, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)
from gensim import corpora, models, similarities, matutils
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from Corpus import Corpus
import preprocess


nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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
    self.dictionary.filter_extremes(no_above=5,keep_n=100000)
    self.dictionary.compactify()
    print len(self.dictionary)

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

  '''
  The main function that runs the lda model on the corpus
  '''
  def run(self,min_topics=6,max_topics=8,step=1):
        l = np.array([sum(cnt for _, cnt in doc) for doc in self.corpus])
        kl = []
    #for i in range(min_topics,max_topics,step):
        self.model = models.ldamodel.LdaModel(corpus=self.corpus,
            id2word=self.dictionary,num_topics=8)
        #m1 = self.model.expElogbeta
        #U,cm1,V = np.linalg.svd(m1)
        
        #Document-topic matrix
        lda_topics = self.model[self.corpus]

        #m2 = matutils.corpus2dense(lda_topics, self.model.num_topics).transpose()
        #cm2 = l.dot(m2)
        #cm2 = cm2 + 0.0001
        #cm2norm = np.linalg.norm(l)
        #cm2 = cm2/cm2norm
        #kl.append(self.sym_kl(cm1,cm2))
        return kl
        
  #def sym_kl(self, p,q):
    #return np.sum([stats.entropy(p,q),stats.entropy(q,p)])

if __name__ == "__main__":
  lda = TagLDA()
  lda.process()
  lda.create_dictionary()
  lda.create_corpus()
  kl = lda.run(max_topics=8)
  model = lda.model
  model.print_topics(8)
  print "\n\n\n\n\n"


  testtxt = "bowater industries profit exceed expectations bowater industries plc bwtr pretax profits mln stg exceeded market expectations mln and pushed company shares sharply high last night dealers shares eased back bowater reported mln stg profit company statement accompanying results that underlying trend showed improvement and intended expand developing existing businesses and seeking opportunities added that had appointed david lyon managing director redland plc rdld chief executive analysts noted that bowater profits mln stg mln previously had boost pension benefits mln stg profit australia and east showed greatest percentage rise jumping pct mln mln profit operations rose pct mln and europe pct mln reuter"

  t = lda.dictionary.doc2bow(testtxt.lower().split())
  
  print lda.model.get_document_topics(t)
  print lda.model.get_topic_terms(1, topn=20)



  # Plot kl divergence against number of topics
  #plt.plot(kl)
  #plt.ylabel('Symmetric KL Divergence')
  #plt.xlabel('Number of Topics')
  #plt.savefig('kldiv.png', bbox_inches='tight')



