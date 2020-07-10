import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim import similarities
#using Skilearn
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')

"""
Cleaning the texts for topic modelling, Panctuation and stop words will be removed
"""
def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text

# For gensim we need to tokenize the data and filter out stopwords
def tokenizeDocuments(Docs):
    tokenized_data = []
    for text in Docs:
        tokenized_data.append(clean_text(text))

    return tokenized_data

def buildLDA_model(tokenized_data,n_topics):
    # Build a Dictionary - association word to numeric id
    dictionary = corpora.Dictionary(tokenized_data)

    # Transform the collection of texts to a numerical form
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    # Have a look at how the 20th document looks like: [(word_id, count), ...]
    print(len(dictionary),len(tokenized_data))
    # Build the LDA model
    lda_model = models.LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)

    return lda_model

def print_topMF(model,num_topics,top):
    for idx in range(num_topics):
        # Print the first 10 most representative topics
        print("Topic :%s:" % idx, model.print_topic(idx, top))
        print("#"*100)