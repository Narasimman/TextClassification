#collecting all possible words that will be the features for our model 
with open('data/corpus.txt', 'r') as f1:
    feature_list = []
    for line in f1:
        words = line.split()
        feature_list.extend(words)


#Extracting the values of LDA and storing in dictionary of format (key,value) as ((topic,word),probablity) 
topic = ["interest", "wheat", "earn", "corn", "grain", "acq", "crude", "money-supply", "trade", "ship"]
lda = {}
i=0
with open('ldaresults.txt','r') as f3:
    for line in f3:
        tokens = line.split()
        if len(tokens) > 0:
            word_value = (topic[i],tokens[0])
            lda[word_value] = tokens[1]
        else:
            i = i + 1    

# For each line(document) in train.txt, it populates the lda value of the word and corresponding topic of that document
# in feature.csv file which goes as input to the model 
with open ('data/train.txt', 'r') as f4:
	with open ('data/feature_train.csv', 'w') as f5:
		#to write heading
		for word in list(set(feature_list)):
			f5.write(word)
			f5.write(",")
		f5.write("Topic")
		f5.write('\n')
		j=0
		# populates the lda value if present in the document (each line)
		for line in f4:
			topic, text = line.split("\t")
			innerList = []
			for word in feature_list:
				ldaValue = lda.get((topic,word))
				if ldaValue != None:
					f5.write(ldaValue)
				else:
					f5.write('0')
				f5.write(',')
			f5.write(topic)
			f5.write('\n')
			print j
			j = j+1
