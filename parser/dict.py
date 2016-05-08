from gensim.corpora.dictionary import Dictionary
from reutersparser import *

class Dict():
  def __init__(self, dictionary=None):
    self.metadata = False
    self.doc_gen = get_reuters_documents("/home/sims/nlp/TextClassification/data/")

    if dictionary is None:
      dictionary = Dictionary()
      for doc in self.doc_gen:
        print doc['body']
        #dictionary.add_documents([doc['title']])
      self.dictionary = dictionary

if __name__ == "__main__":
  dic = Dict()
  print dic.dictionary
