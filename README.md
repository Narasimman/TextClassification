To run the program, 

1. you need to download the processed Reuters R10 dataset from the following link
http://csmining.org/tl_files/Project_Datasets/r8%20r52/r8-train-all-terms.txt
http://csmining.org/tl_files/Project_Datasets/r8%20r52/r8-test-all-terms.txt

2. create the virtual environment for python and pip install all the required dependencies with the requirements.txt

3. run the trainlda.py file to build the lda model

4. use the attached ipython notebook to run the classifiers (involves further setting up of ipython notebook)


More Details:

Reuters dataset:

http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz

unzip in a data folder


In our model we only used nouns (treebank NN and NNS).  
Part of speech tagging is important to building most topic models.  

At a minimum we should stop common stop words (the, a, it, etc).  

We should also make sure that we don't allow very high frequency words to overpower the rest of the corpus.  
We likely don't want very infrequent words either.  
In our model we removed any words that were in 5 or less documents and any word that appeared in more than 60% of documents.

We used techniques like spell correction and lemmatization to further aggregate words and reduce the dimension of the data.

The idea of combining POS tagging, text chunking, and LDA is well established in various papers, including, for example, “TagLDA: Bringing document structure knowledge into topic models” (2006) 

References:

http://cmci.colorado.edu/~mpaul/files/poslda12.pdf
http://pages.cs.wisc.edu/~jerryzhu/pub/taglda.pdf
http://www.umiacs.umd.edu/~jbg/docs/nips2009-rtl.pdf
