from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec,doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem.snowball import SnowballStemmer
import Preprocessing as pp
import numpy as np
import utility as ut
import pandas as pd

#from flair.data import Sentence
#from flair.models import SequenceTagger
#tag=SequenceTagger.load('ner')


#model_sent = SentenceTransformer('bert-base-nli-mean-tokens')


def doc2vecEmbed(sceneTranscript):
    sceneLevelEmbedings=[]
    for i in range(len(sceneTranscript)):
        #print(i)
        if sceneTranscript[i]!='':
            #txtblb=TextBlob(str(sceneTranscript[i]))
            #sent=[str(j) for j in txtblb.sentences]
            sceneSentEmd=embedSentences([sceneTranscript[i]])
            sceneLevelEmbedings.append(sceneSentEmd)
    docEmbed=[]
    for i in range(len(sceneLevelEmbedings)):
        docEmbed.append(sceneLevelEmbedings[i][0])

    simDoc2vec=cosine_similarity(docEmbed)

    return np.array(docEmbed), simDoc2vec

def getScene_word_emb(scene_st_pos=0,scene_end_pos=444):
    return 0

def getSceneEmbeding(sceneTranscript):
    model= Doc2Vec.load('/people/berhe/Bureau/TLP_thesis/codes/d2v.model')
    #Document Embedding data
    doc2vec_Emb=[]
    for i in range(len(sceneTranscript)):
        doc2vec_Emb.append( model.infer_vector(sceneTranscript[i].lower()))
    doc2vec_Emb=np.array(doc2vec_Emb)

    return doc2vec_Emb

def list_oneHot_encode(scene_listName):
    allNames=[item for sublist in scene_listName for item in sublist]
    uniqueNames=list(set(allNames))
    names = [name for name in uniqueNames if '#' not in name]
    data = {v: k for k, v in enumerate(names)}
    existingName_rep=[]
    for sln in scene_listName:
        sceneName_rep=np.zeros(len(uniqueNames))
        for n in sln:
            if n in data:
                key=data[n]
                sceneName_rep[key]=1
        existingName_rep.append(sceneName_rep)
    existingName_Data=np.array(existingName_rep)
    avg_ent = np.mean(existingName_Data, axis=0)
    for i in range(existingName_Data.shape[0]):
        if not np.any(existingName_Data[i]):
            existingName_Data[i]=avg_ent

    return existingName_Data, data

def getSpeakingCharacters(dataframe):
    speakingChars = dataframe['Speaking_Chararacters'].tolist()
    speakingChars = [i.split(',') for i in speakingChars]
    sp_char, sp_char_data = list_oneHot_encode(speakingChars)
    return sp_char
def getAppearingCharacters(dataframe):
    speakingChars = dataframe['Appearing_Characters'].tolist()
    speakingChars = [i.split(',') for i in speakingChars]
    sp_char, sp_char_data = list_oneHot_encode(speakingChars)
    return sp_char

def getKeywords(dataframe):
    keywords = dataframe.Scene_Keybords.tolist()
    keywordsList = [i.split() if i != np.nan else [] for i in keywords]
    keywords_onehot, keyword_data = list_oneHot_encode(keywordsList)
    return keywords_onehot

def getEntities(dataframe):
    entities = ut.getNormalizedEntities(dataframe)
    entities_onehot,data=list_oneHot_encode(entities)
    return entities_onehot

def extract_topN_Keywords_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

def tfidfSklearn(corpus,stem=False):
    """
    The coupus is list of sequence of words separated by space. This can be preprocessed using the Preprocessing file.
    This can be applied to represent the characters tfidf representation which inturn can capture the cooccurence of characters.
    When using text of each scene. This will assume the elements of corpus are scenes.
    We can have embeding of episodes and their respctive scenes here we cosider the text or list of character names of an episode as the corpus.
    We can have the option of stemming in the coupus
    """
    if stem:
        corpus1 = []
        for cor in corpus:
            corpus1.append(pp.sceneStemText(cor))
        corpus = corpus1
    stop_words = set(stopwords.words("english"))
    tfidf = TfidfVectorizer(stop_words=stop_words)
    tfidf_vectors=tfidf.fit_transform(corpus)
    tfidf_features = tfidf.get_feature_names()
    #feature_names = vectorizer.get_feature_names()

    return tfidf_vectors,tfidf_features

def tfidfEmbedding(corpus,stem=True):
    if stem:
        corpus1=[]
        for cor in corpus:
            corpus1.append(pp.sceneStemText(cor))
        corpus=corpus1
    stop_words = set(stopwords.words("english"))
    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
    X=cv.fit_transform(corpus)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)# get feature names
    feature_names=cv.get_feature_names()


    sceneTFIDF=[]
    for i in range(len(corpus)):
        doc=corpus[i]
        tf_idf_vector=tfidf_transformer.transform(cv.transform(doc))
        sceneTFIDF.append(tf_idf_vector)
    return np.array(sceneTFIDF)

def getScene_keywords(scenesCorpus):
    with open(codesFile+"python_files/keywordsCorpus", "rb") as input_file:
        episodesCorpus= pickle.load(input_file)

    stop_words = set(stopwords.words("english"))
    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
    X=cv.fit_transform(episodesCorpus)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)# get feature names
    feature_names=cv.get_feature_names()

    sceneTFIDF=[]
    episodeStemwords=pp.sceneStemText(scenesCorpus)
    for i in range(len(episodeStemwords)):
        doc=episodeStemwords[i]

        #generate tf-idf for the given document
        #tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
        tf_idf_vector=tfidf_transformer.transform(cv.transform(sceneStemwords))
        sceneTFIDF.append(tf_idf_vector)

    return np.array(sceneTFIDF)

"""
creating a topic model based on a given corpus (Can be episode level corpus or Tv series level corpus) to capture different labele of granularity.
It is composed of scene data.
"""
def getTopic_model(corpus, num_topics):
    vectorize=CountVectorizer(min_df=5,max_df=0.9,lowercase=True,stop_words='english',token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    data_vector=vectorize.fit_transform(episodesCorpus)
    lda_model = LatentDirichletAllocation(n_components=num_top, max_iter=15, learning_method='online')
    lda_model.fit(data_vector)

    return lda_model,vectorize

"""
predict the topic/s of each scene in the corpus
"""
def getScene_topic(corpus):
    lda_model,vect=getTopic_model(corpus=corpus,num_topics=20)
    topic=list(lda_model.fit_transform(vect.transform(scenetext))[0])
    topic=[topic.index(element) for element in topic if element>0.10 ]
    return topic

#pd.read_csv('Scenes_Dataset_New_Keywords.csv') current clean dataset
def generateEmbeddings(dataframe):
    """
    This function generates the mebeddings of the dataset (text, speaking characters, entities, keywords) and save the files in embeddings folder as numpy array
    It saves the row embeddings
    """
    print("hello")
