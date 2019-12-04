from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import math
import pickle
import time
import operator
import json
import re

class ICSearchIndex:

    def __inti__(self):
        self.index = {}
        self.queryvector = []
        self.filteredQuery = []
        self.totalDocs = 0
        self.tfidfDataset = {}
        self.tfDataSet = {}
        self.idfDataset = {}
        self.output_5k = dict()

    def retrieveImgUrl(self,docList):

        docs = []
        urlMapping = dict()
        for item in docList:
            docs.append(item[0])
        print("retrieveImgUrl docs:", docs)
        with open("image_caption/panoptic_val2017.json", "r") as read_file:
            data = json.load(read_file)
            for item in data["images"]:
                if item["file_name"] in docs:
                    urlMapping[item["file_name"]] = item["coco_url"]

        return urlMapping


    def loadFiles(self):    # Loads files such as MainIndex, TFIDF, TF and IDF
        f1 = open("image_caption/imp_files/mainIndex.pkl", "rb")
        tempIndex = pickle.load(f1)
        f1.close()
        ts = time.gmtime()
        print("In ICSearchIndex loadFiles loading index complete", time.strftime("%Y-%m-%d %H:%M:%S", ts))
        self.index = tempIndex
        f2 = open("image_caption/imp_files/tfidfDataset.pkl", "rb")
        tfidf = pickle.load(f2)
        f2.close()
        self.tfidfDataset = tfidf
        f3 = open("image_caption/imp_files/idfDataset.pkl", "rb")
        idf = pickle.load(f3)
        f3.close()
        self.idfDataset = idf
        f4 = open("image_caption/imp_files/tfDataset.pkl", "rb")
        tf = pickle.load(f4)
        f4.close()
        self.tfDataSet = tf
        f5 = open('image_caption/imp_files/totalDoc.txt', "r")
        self.totalDocs = f5.readline()
        f5.close()
        # f6 = open("output_5k.pkl", "rb")
        # self.output_5k = pickle.load(f6)
        # f6.close()

    def filterQuery(self, terms):  # removes stop words and perform stemming
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        result = []
        for word in terms:
            if word not in stop_words:
                result.append(ps.stem(word))

        self.filteredQuery = result

    def computeTFIDF(self):
        print("in computeTFIDF filteredQuery", self.filteredQuery)
        qv = {}

        for term in self.filteredQuery:
            if term in self.index.keys():
                df = len(self.index[term])  # not working properly, returns wrong value
                qv[term] = self.filteredQuery.count(term) * math.log2(int(self.totalDocs) / df)
            else:
                continue
                print(term + " not present in Index file")
        self.queryvector = qv

    def returnResults(self):
        print("in returnResults()", self.filteredQuery)

        docs = []

        for term, posting in self.index.items():
            if term in self.filteredQuery:
                for item in posting:
                    docs.append(item[0])
        docs = list(set(docs))
        temp = {your_key: self.tfidfDataset[your_key] for your_key in docs} # filter tfidf
        sum = dict.fromkeys(docs,0)
        for docID, terms in temp.items():
            for term in list(terms.keys()):
                if term in self.filteredQuery:
                    sum[docID] = sum[docID] + terms[term]
                    continue
                else:
                    del terms[term]
        # print("Sum = ", sum)

        sorted_x = sorted(sum.items(), key=operator.itemgetter(1), reverse=True)
        final = self.retrieveImgUrl(sorted_x[0:20])   # getting URL's from output_5k
        output_idf = {your_key: self.idfDataset[your_key] for your_key in list(sum.keys())}  # filter idf
        output_tf = {your_key: self.tfDataSet[your_key] for your_key in list(sum.keys())}  # filter tf
        for docID, terms in output_idf.items():
            for term in list(terms.keys()):
                if term in self.filteredQuery:
                    continue
                else:
                    del terms[term]
        for docID, terms in output_tf.items():
            for term in list(terms.keys()):
                if term in self.filteredQuery:
                    continue
                else:
                    del terms[term]
        output_tfidf = {your_key: self.tfidfDataset[your_key] for your_key in list(sum.keys())}  # filter tfidf
        print(final)
        print(sorted_x[0:20])
        return [final, sorted_x[0:20]]


    def calculateMagnitude(self,vecValues):
        sum = 0
        for item in vecValues:
            sum = sum + (item*item)
        magnitude = math.sqrt(sum)
        return magnitude


    def calculateCosine(self):

        queryVecValues = list(self.queryvector.values())
        qvMag = self.calculateMagnitude(queryVecValues)
        print("QueryVector magnitude=", qvMag)
        f1 = open(r"F:\UTA\1st sem\DM\hotel-reviews (1)\tfidfDataset.pkl", "rb")
        tfidf = pickle.load(f1)
        f1.close()
        self.tfidfDataset = tfidf

        docMag = {}
        cosine = {}
        # print("doc 1 weights", self.tfidfDataset[1])
        for docID, weight in self.tfidfDataset.items():
            intersection = list( set(self.queryvector.keys()) & set(weight.keys()) )

            docMag[docID] = self.calculateMagnitude(list(weight.values()))
            cosine[docID] = (self.dotproduct(intersection, weight)) / (docMag[docID] * qvMag)


        sorted_x = sorted(cosine.items(), key=operator.itemgetter(1), reverse=True)
        final = self.retrieveText(sorted_x[0:21])
        return final


    def dotproduct(self, intersection, docVector):
        n = 0
        sum = 0
        while n < len(intersection):
            sum = sum + self.queryvector[intersection[n]] * docVector[intersection[n]]
            n = n + 1
        return sum

