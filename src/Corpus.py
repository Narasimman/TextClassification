class Corpus(object):

  def __init__(self, d):
    self.dictionary = d

  def __iter__(self):
    for line in open('data/corpus.txt','r'):
      yield self.dictionary.doc2bow(line.lower().split())


if __name__ == "__main__":
  Corpus()
