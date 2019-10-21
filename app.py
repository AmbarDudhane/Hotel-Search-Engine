import threading

from flask import Flask, request, render_template
import searchIndex
import time
import requests

app = Flask(__name__)
s = searchIndex.SearchIndex()


@app.route('/')
def hello_world():
    return render_template('Homepage.html')


@app.before_first_request
def activate_job():
    ts = time.gmtime()
    print("In run job before loading index", time.strftime("%Y-%m-%d %H:%M:%S", ts))
    s.loadFiles()


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


if __name__ == '__main__':
    app.run()
