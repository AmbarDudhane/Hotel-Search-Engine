from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os
import pickle
import pandas as pd
import operator
import math
from nltk.stem import WordNetLemmatizer


class Classifier:

    def __inti__(self):
        print("in constructor")

    def load(self):
        print("in load function")
        f1 = open(r"C:\Users\Ambar\PycharmProjects\FlaskDemo\Classifier\vocab.pkl", "rb")
        f2 = open(r"C:\Users\Ambar\PycharmProjects\FlaskDemo\Classifier\prior.pkl", "rb")
        f3 = open(r"C:\Users\Ambar\PycharmProjects\FlaskDemo\Classifier\cond_prob.pkl", "rb")
        vocab = pickle.load(f1)
        prior = pickle.load(f2)
        cond_prob = pickle.load(f3)
        f1.close()
        f2.close()
        f3.close()

        return [vocab, prior, cond_prob]

    def apply_multinomial_nb(self, vocabulary, prior, cond_prob, df_test, flag):
        df_test["Prediction"] = ""
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')  # getting rid of punctuation while tokenizing
        for index, row in df_test.iterrows():
            tokens = list(tokenizer.tokenize(row[2].lower()))
            filtered_tokens = list()
            for term in tokens:
                if term not in stop_words:
                    filtered_tokens.append(lemmatizer.lemmatize(term))
            tokens = self.ExtractElements(vocabulary, filtered_tokens)

            score = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for c in [1, 2, 3, 4, 5]:
                # score[c] = math.log2(prior[c])
                score[c] = prior[c]
                for term in tokens:
                    # score[c] = score[c] + math.log2(cond_prob[c][term])
                    score[c] = score[c] + cond_prob[c][term]
            df_test.at[index, 'Prediction'] = max(score.items(), key=operator.itemgetter(1))[0]
        # if called from test function, then execute evaluation
        if flag == "test":
            self.evaluation(df_test)
        return df_test

    def ExtractElements(self, vocab, tokens):
        return list(set(vocab) & set(tokens))

    def evaluation(self, df_test):
        actual = df_test['reviews.rating'].tolist()
        predicted = df_test['Prediction'].tolist()
        classes = [1, 2, 3, 4, 5]
        # print(confusion_matrix(actual, predicted, labels=classes))
        # print(classification_report(actual, predicted))
        from math import sqrt
        # Calculating MSE
        mse = mean_squared_error(actual, predicted)
        print("Mean Squared Error:", mse)
        print("RMSE:", sqrt(mse))

        # Calculating accuracy score
        print("Accuracy score:", "{:.0%}".format(accuracy_score(actual, predicted)))

    def test(self):
        print("Test")
        os.chdir(r"C:\Users\Ambar\PycharmProjects\FlaskDemo\Classifier")
        df = pd.read_csv(r"F:\UTA\1st sem\DM\hotel-reviews (1)\Hotel_Reviews_Jun19_reduced.csv",
                         usecols=['reviews.id', 'reviews.text', 'reviews.rating'])
        from sklearn.model_selection import train_test_split

        df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)
        result = self.load()
        self.apply_multinomial_nb(vocabulary=result[0], prior=result[1], cond_prob=result[2], df_test=df_test,
                                  flag="test")


if __name__ == "__main__":
    inst = Classifier()
    inst.test()
