import os
import wordcloud

MODELS_DIR = "data"

output_dir = "output"
final_topics = open(os.path.join(MODELS_DIR, "final_topics.txt"), 'rb')
curr_topic = 0
for line in final_topics:
  line = line.strip()[line.rindex(":") + 2:]
  scores = [float(x.split("*")[0]) for x in line.split(" + ")]
  words = [x.split("*")[1] for x in line.split(" + ")]
  freqs = []
  for word, score in zip(words, scores):
    freqs.append((word, score))
  wc = wordcloud.WordCloud()
  elements = wc.fit_words(freqs)
  #wc.draw(elements, "gs_topic_%d.png" % (curr_topic),
                #width=120, height=120)
  wc.generate_from_frequencies(freqs)
  wc.to_file(os.path.join(output_dir, "abc_%d.png" % (curr_topic)))
  curr_topic += 1
final_topics.close()
