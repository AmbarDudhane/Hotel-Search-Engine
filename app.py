import threading

from flask import Flask, request, render_template
import searchIndex
import time
import pandas as pd
from Classifier.Classifier_Star_Apply import Classifier
from image_caption.ic_search_index import ICSearchIndex

app = Flask(__name__)
s = searchIndex.SearchIndex()
ic = ICSearchIndex()

@app.route('/')
def hello_world():
    return render_template('Homepage.html')


@app.before_first_request
def activate_job():
    ts = time.gmtime()
    print("In run job before loading index", time.strftime("%Y-%m-%d %H:%M:%S", ts))
    s.loadFiles()
    ic.loadFiles()


@app.route('/submit', methods=['POST'])
def submit():
    searchText = request.form['search']
    print("Search text=", searchText)
    terms = searchText.lower().split(" ")  # performing lower case and tokenizing
    s.filterQuery(terms)
    output = s.returnResults()

    # s.readTotalDocs()
    # s.readIndexFile()
    # s.computeTFIDF()
    # results = s.calculateCosine()

    # return 'Search Results are' + str(results)
    return render_template('Results.html', data=output, query=searchText)


@app.route('/classify')
def classifier_display():
    return render_template('Classifier.html', data=["",""])

@app.route('/submit_classify', methods=['POST'])
def submit_classify():
    print("in classify")
    c = Classifier()
    searchText = request.form['search']
    print("Search text=", searchText)
    dict1 = {'reviews.rating': [0], 'reviews.id': [1], 'reviews.text': [searchText]}
    df = pd.DataFrame(dict1)
    result = c.load()
    result_df = c.apply_multinomial_nb(result[0], result[1], result[2], df_test=df, flag="not test")
    print("Prediction:", result_df.get_value(0, 'Prediction'))
    return render_template('Classifier.html', data=[result_df.get_value(0, 'Prediction'), searchText])

@app.route('/image_captioning')
def image_captioning():
    return render_template("ImageCaptioning.html", data="")

@app.route('/show_images', methods=['POST'])
def show_images():
    searchText = request.form['search']
    print("Search text=", searchText)
    terms = searchText.lower().split(" ")  # performing lower case and tokenizing
    ic.filterQuery(terms)
    op = ic.returnResults()
    return render_template("ImageCaptioning.html", data=op)

if __name__ == '__main__':
    app.run()
