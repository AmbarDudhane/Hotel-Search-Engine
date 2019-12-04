# Classify using star based rating
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pickle
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def train_multinomial_nb(class_text, df_train, docs_count):
    tokenizer = RegexpTokenizer(r'\w+')  # getting rid of punctuation while tokenizing
    # removing stop words
    # stop_words = set(stopwords.words('english'))

    # Training multinomial NB
    tokens = list(tokenizer.tokenize(class_text[0] + " " + class_text[1] + " " + class_text[2] + " " +
                                     class_text[3] + " " + class_text[4]))

    temp = list(set(tokens))
    print("temp len:", len(temp))
    vocabulary = list()
    for term in temp:
        if term not in stop_words:
            vocabulary.append(lemmatizer.lemmatize(term))
    print("Vocab len:", len(vocabulary))
    if " " in vocabulary:
        print("Space is there")
    N = len(df_train)
    classes = [1, 2, 3, 4, 5]
    # class probabilities
    prior = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    # term probablities
    cond_prob = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

    for c in classes:
        print("c = ", c)
        Nc = docs_count[c-1]
        prior[c] = float("{:.5f}".format(Nc / N))

        for term in vocabulary:
            Tct = class_text[c-1].count(term)   # class_text[1-1] i.e selecting text of one_star rating which is at 0th position
            prob = (Tct + 0.4) / (len(class_text[c-1]) + len(vocabulary))
            cond_prob[c][term] = float("{:.8f}".format(prob))
            # cond_prob_positive[term] = prob

    store(vocabulary, prior, cond_prob)
    print("Training completed...")
    return [vocabulary, prior, cond_prob]

def store(vocab, prior, cond_prob):
    f1 = open(r"C:\Users\Ambar\PycharmProjects\FlaskDemo\Classifier\vocab.pkl", "wb")
    f2 = open(r"C:\Users\Ambar\PycharmProjects\FlaskDemo\Classifier\prior.pkl", "wb")
    f3 = open(r"C:\Users\Ambar\PycharmProjects\FlaskDemo\Classifier\cond_prob.pkl", "wb")

    pickle.dump(vocab, f1)
    pickle.dump(prior, f2)
    pickle.dump(cond_prob, f3)
    f1.close()
    f2.close()
    f3.close()

# def apply_multinomial_nb(vocabulary,prior,cond_prob, df_test):
#     df_test["Prediction"] = ""
#     tokenizer = RegexpTokenizer(r'\w+')  # getting rid of punctuation while tokenizing
#     for index, row in df_test.iterrows():
#         tokens = list(tokenizer.tokenize(row[2].lower()))
#         tokens = ExtractElements(vocabulary, tokens)
#
#         score = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
#         for c in [1,2,3,4,5]:
#             score[c] = prior[c]
#             for term in tokens:
#                 score[c] = score[c] * cond_prob[c][term]
#
#         df_test.at[index,'Prediction'] = max(score.items(), key=operator.itemgetter(1))[0]
#
#     evaluation(df_test)
#
# def ExtractElements(vocab, tokens):
#     return list(set(vocab) & set(tokens))
#
# def evaluation(df_test):
#     actual = df_test['reviews.rating'].tolist()
#     prediced = df_test['Prediction'].tolist()
#
#     # Calculating MSE
#     print("Mean Squared Error:", mean_squared_error(actual, prediced))
#
#     # Calculating accuracy score
#     print("Accuracy score:", accuracy_score(actual, prediced))

os.chdir("F:\\UTA\\1st sem\\DM\\hotel-reviews (1)")

df = pd.read_csv("Hotel_Reviews_Jun19_reduced.csv", usecols=['reviews.id', 'reviews.text', 'reviews.rating'])
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

# print(df_train)
one_star_text = ""
two_star_text = ""
three_star_text = ""
four_star_text = ""
five_star_text = ""

one_star_count = 0
two_star_count = 0
three_star_count = 0
four_star_count = 0
five_star_count = 0

for index, row in df_train.iterrows():
    # print("row[2]=", row[2])
    if row[0] == 1:
        one_star_text = one_star_text + " " + row[2].lower()
        one_star_count = one_star_count + 1
    if row[0] == 2:
        two_star_text = two_star_text + " " + row[2].lower()
        two_star_count = two_star_count + 1
    if row[0] == 3:
        three_star_text = three_star_text + " " + row[2].lower()
        three_star_count = three_star_count + 1
    if row[0] == 4:
        four_star_text = four_star_text + " " + row[2].lower()
        four_star_count = four_star_count + 1
    if row[0] == 5:
        five_star_text = five_star_text + " " + row[2].lower()
        five_star_count = five_star_count + 1

class_text = [one_star_text, two_star_text, three_star_text, four_star_text, five_star_text]
docs_count = [one_star_count, two_star_count, three_star_count, four_star_count, five_star_count]

result = train_multinomial_nb(class_text, df_train, docs_count)
# apply_multinomial_nb(vocabulary=result[0], prior=result[1], cond_prob=result[2], df_test=df_test)
