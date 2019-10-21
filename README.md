# Hotel-Search-Engine
This web application lets you retrieve text information on hotels. Hotel Search Engine provides a search bar by using which users can enter free text query and get text results.

### Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
To start with the project, below frameworks are needed:
1. jetBrains pycharm
2. Python 3.x
3. Python Flask

### Installing
1. Install python 3.x from https://www.python.org/downloads/release/python-373/. Many people prefer using Anaconda which provides lot of libraries of python, but I was not familiar with Anaconda at the development time of this project. Hence I chose plain python.
2. Install jetBrains pycharm from https://www.jetbrains.com/pycharm/download/.
3. Install flask web framework from http://flask.palletsprojects.com/en/1.1.x/installation/.

Next step will be to import code from github repository.
After importing, if pycharm shows errors on packages, then pycharm gives suggestion to install the packages. Install those packages.
Run app.py which will run application on 127.0.0.1:5000.

### Program files:
1. createIndex.py : used to create inverted index of terms in the documents.
2. searchIndex.py : used to load the index stored on disk and calculates tf-idf and cosine similarity values. 
3. app.py : This file launches the server and also maps URL's to different methods. This file calls searchIndex.py after user click's 
"Go" button on UI.
4. Homepage.html : basic html UI having search bar and go button.
5. Results.html : displays the search results
6. hilitor.js : third party library to highlight the search keywords

### Deployment
I deployed the web app on web hosting platform pythonanywhere.com. You can deploy it to any python hosting platform. For deployment, you need to create "new flask web app". Pythonanywhere will take care of every installation. On pythonaywhere, you need to install nltk package by executing command "pip install --user -U nltk". NLTK package is used in stemming and removing stopwords.



### Authors
Ambar Dudhane - Entire Development -  Graduate Student at Univeristy of Texas at Arlington

### References:
While building the application I used below references:
1. http://www.ardendertat.com/2011/07/17/how-to-implement-a-search-engine-part-3-ranking-tf-idf/
2. http://www.site.uottawa.ca/~diana/csi4107/cosine_tf_idf_example.pdf
