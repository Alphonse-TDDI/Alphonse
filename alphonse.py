import nltk #Importing NLTK
nltk.download('punkt') #Redownloading/updating the main-base requirement.
from nltk.stem.lancaster import LancasterStemmer #calling LancasterStemmer Function
stemmer = LancasterStemmer() #creating a stemmer
import numpy
import tflearn
import tensorflow
import random
import json #other libraries has been used
import pandas as pd




# Or export it in many ways, e.g. a list of tuples
tuples = [tuple(x) for x in df.values]

# or export it as a list of dicts
dicts = df.to_dict().values()

with open("datta.json") as file: #reading the json (intents) file
    data = df = pd.read_csv('datta.json', sep=',')
    words = [] #creating a list of required elements and data-tables
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]: #reading every individual tables one by one
      for pattern in intent["patterns"]:
          wrds = nltk.word_tokenize(pattern) #tokenizing every individual tables one by one
          words.extend(wrds)
          docs_x.append(wrds)
          docs_y.append(intent["id"])          
          if intent["id"] not in labels:
              labels.append(intent["id"]) 
words = [stemmer.stem(w.lower()) for w in words if w not in "?"] #labeling words for tables
words = sorted(list(set(words)))
labels = sorted(labels) #sorting every table for its exact label
training = []
output = []
out_empty = [0 for _ in range(len(labels))] #reading the doc
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else: #creating a bag of words can be chosen by AI
            bag.append(0)
            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])]
            training.append(bag)
            output.append(output_row)

training = numpy.array(training) #creating a training option
output = numpy.array(output) #creating output

training = numpy.array(training)
output = numpy.array(output)
tensorflow.compat.v1.reset_default_graph() #resetting TF numbers to prevent from getting errors.
#tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net) #main architecture of the AI

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #config of the training 
model.save("model.tflearn") #saving the AI model into a file.