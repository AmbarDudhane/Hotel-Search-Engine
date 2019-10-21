'''
Author : Ambar Dudhane
Date : 6-10-2019
Contact : ambarsd12345@hotmail.com
'''
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import math
import pickle
import datetime


def storeTotalDocuments(total):
    f = open(r'totalDoc.txt', "w")
    f.write(str(total))
    f.close()

def storeIndex(path, filename, index):
    f = open(path + '\\' + "mainIndex.pkl", "wb")
    pickle.dump(index, f)
    f.close()


def computeTFIDF(mainIndex, total):
    tfidf = {}
    idfData = {}
    tfData = {}
    df = 0
    for term, posting in mainIndex.items():
        df = len(posting)
        idf = math.log2((total / df))

        for item in posting:
            if item[0] in tfidf.keys():
                tfidf[item[0]][term] = item[1] * idf
                idfData[item[0]][term] = idf
                tfData[item[0]][term] = item[1]
            else:
                tfidf[item[0]] = {term: item[1] * idf}
                idfData[item[0]] = {term: idf}
                tfData[item[0]] = {term : item[1]}

    f = open(r'F:\UTA\1st sem\DM\hotel-reviews (1)\tfidfDataset.pkl', "wb")
    pickle.dump(tfidf, f)    # which can be used in searchIndex.py
    f.close()
    f1 = open(r'F:\UTA\1st sem\DM\hotel-reviews (1)\idfDataset.pkl', "wb")
    pickle.dump(idfData, f1)
    f1.close()
    f2 = open(r'F:\UTA\1st sem\DM\hotel-reviews (1)\tfDataset.pkl', "wb")
    pickle.dump(tfData, f2)
    f2.close()
    print("TFIDF of doc1",tfidf[1])
    print("TFIDF of doc2", tfidf[2])
    print("IDF of doc1", idfData[1])
    print("IDF of doc2", idfData[2])


os.chdir("F:\\UTA\\1st sem\\DM\\hotel-reviews (1)")
now = datetime.datetime.now()

print("Starting time: ", now.strftime("%Y-%m-%d %H:%M:%S"))
data = pd.read_csv("Hotel_Reviews_Jun19_reduced.csv", usecols=['reviews.id', 'reviews.text'])

print(data.shape, "Data length=", len(data))


reviewDict = data.set_index("reviews.id")['reviews.text'].to_dict()

bowDict = {}
tokenizer = RegexpTokenizer(r'\w+')  # getting rid of punctuation while tokenizing
for key, value in reviewDict.items():
    bowDict[key] = list(tokenizer.tokenize(str(value).lower()))



# removing stop words
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

for key, value in bowDict.items():
    temp = []
    for word in value:
        if word not in stop_words:
            temp.append(ps.stem(word))  # performing stemming
    bowDict[key] = temp


# creating inverted index
# first create document level index and then merge it to main index

docIndex = {}

for id, words in bowDict.items():
    docIndex[id] = []
    for word in words:
        docIndex[id].append({word: [id, words.count(word)]})




mainIndex = {}

for indexList in docIndex.values():
    for element in indexList:
        term = list(element.keys())  # term
        value = list(element.values())  # [docId, occurances]

        if term[0] in mainIndex:  # term is already present in mainIndex
            if value[0] not in mainIndex[term[0]]:
                mainIndex[term[0]].append(value[0])
        else:  # term is not present in mainIndex
            mainIndex[term[0]] = [value[0]]

print("mainIndex size=", len(mainIndex))

# saving index file to disk
storeIndex(r'F:\UTA\1st sem\DM\hotel-reviews (1)', 'MainIndex.txt', mainIndex)

# Calculating tf-idf
computeTFIDF(mainIndex, len(docIndex))
storeTotalDocuments(len(docIndex))
now2 = datetime.datetime.now()
print("Ending time: ", now2.strftime("%Y-%m-%d %H:%M:%S"))