# -*- coding: utf-8 -*-
#importing 

import nltk 
nltk.download()
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle
import tflearn 
import random 
import numpy
import json    
import tensorflow as tf
import pickle 
with open(r'F:\PROJECTS\chatbot\codes\intents.json') as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:  
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    
    #tokenize words
    #tokenize means chopping off the words into different chunks  
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wr = nltk.word_tokenize(pattern)
            words.extend(wr)
    #its just going to append all the patterns we have         
            docs_x.append(wr)
            docs_y.append(intent["tag"])
    
            
    #taking all the tags in a list        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    #we are going to to covert every word and turn them into lower case 
    words = [stemmer.stem(w.lower()) for w in words if w !="?"]
    #sorted them by creating a list, set remove duplicate
    words = sorted(list(set(words)))  
    
    labels = sorted(labels) 
    
    # up until now we created a string but we need to convert into numbers to work on DNN
    
    #bagging onehotencoder
    training = []
    output = []   
    
    out_empty = [0 for _ in range (len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc] 
        
        for w in words: 
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1            
                
                
        training.append(bag)
        output.append(output_row)
    #training
    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)         
                
    tf.reset_default_graph()
    
net = tflearn.input_data(shape=[None, len(training[0])])            
net = tflearn.fully_connected(net, 10) 
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
    
model = tflearn.DNN(net)   
#if trained don't train it again 
try: 
    model.load("model.tflearn")
except:

    model.fit(training, output, n_epoch=1000, batch_size=10)
    model.save("model.tflearn")        

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s]
     
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i].append(1)
    return numpy.array(bag)            
 
#chat method to take input and show output    
def chat():
    print("start talking to boi")  
    while True:
        inpu = input("Talker: ")
        if inpu.lower() == "quit":
            break
        
        result = model.predict([bag_of_words(inpu, words)])
        result_index = numpy.argmax(result) 
        tag = labels[result_index]  
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                response = tg['responses']
                    
        
        print(random.choice(response))
        
chat()        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
