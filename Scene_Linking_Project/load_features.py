import numpy as np
import pandas as pd
import utility as ut
import Embeddings as emb
from sklearn.metrics.pairwise import cosine_similarity
import itertools
#normDF=pd.read_csv('/home/berhe/Desktop/Scene_Linking_Project/Data/Scene_Dataset_Normalized.csv')
class Rawdata:
    def __init__(self,dataframe):
        self.sp_char=emb.getSpeakingCharacters(dataframe)
        self.app_char=emb.getAppearingCharacters(dataframe)
        self.entities=emb.getEntities(dataframe)
        self.keywords=emb.getKeywords(dataframe)
        self.w_sp_ch_lines=np.load('Data/Wieghted_Speaking_Charcters_lines.npy')
        self.w_sp_ch_words=np.load('Data/Wieghted_Speaking_Charcters_words.npy')
        self.doc2vec=np.load('Data/Scene_Doc2Vec_Embedding.npy')
        self.d2v_bert=np.load('Data/dataset_D2V_embedding.npy')
        sceneTranscripts = dataframe['Scene_Texts'].tolist()
        tfidfVect, featName = emb.tfidfSklearn(sceneTranscripts, stem=True)
        self.tfidf_text=tfidfVect.toarray()
class Rawdata_Gran:
    def __init__(self,dataframe,start,end):
        self.sp_char=emb.getSpeakingCharacters(dataframe)
        self.app_char=emb.getAppearingCharacters(dataframe)
        self.entities=emb.getEntities(dataframe)
        self.keywords=emb.getKeywords(dataframe)
        w_sp_ch_lines = np.load('Data/Wieghted_Speaking_Charcters_lines.npy')
        w_sp_ch_lines_g=w_sp_ch_lines[start:end]
        self.w_sp_ch_lines=w_sp_ch_lines_g
        w_sp_ch_words = np.load('Data/Wieghted_Speaking_Charcters_words.npy')
        w_sp_ch_words_g=w_sp_ch_words[start:end]
        self.w_sp_ch_words=w_sp_ch_words_g
        doc2vec = np.load('Data/Scene_Doc2Vec_Embedding.npy')
        doc2vec_g=doc2vec[start:end]
        self.doc2vec=doc2vec_g
        d2v_bert = np.load('Data/dataset_D2V_embedding.npy')
        d2v_bert_g=d2v_bert[start:end]
        self.d2v_bert=d2v_bert_g
        sceneTranscripts = dataframe['Scene_Texts'].tolist()
        tfidfVect, featName = emb.tfidfSklearn(sceneTranscripts, stem=True)
        self.tfidf_text=tfidfVect.toarray()

class Simdata:
    def __init__(self,rawData):
        self.sp_char = cosine_similarity((rawData.sp_char))
        self.app_char = cosine_similarity((rawData.app_char))
        self.entities = cosine_similarity((rawData.entities))
        self.keywords = cosine_similarity((rawData.keywords))
        self.w_sp_ch_lines = cosine_similarity((rawData.w_sp_ch_lines))
        self.w_sp_ch_words = cosine_similarity((rawData.w_sp_ch_words))
        self.doc2vec = cosine_similarity((rawData.doc2vec))
        self.d2v_bert=cosine_similarity((rawData.d2v_bert))
        self.tfidf_text=cosine_similarity((rawData.tfidf_text))


class Groundtruth:
    def __init__(self,dataframe):
        ref_array_stories, ref_array_Substories = ut.getLinkedDFNew(dataframe)
        self.ref_link_stories=ref_array_stories
        self.ref_link_substories=ref_array_Substories

        sceneStorycode, sceneSubStorycode = ut.getSceneStoryCode(dataframe)
        self.ref_code_stories=sceneStorycode
        self.ref_code_substories=sceneSubStorycode

        sceneNumber, directLinked = ut.getDirectLinks(dataframe=dataframe)
        firstDirectedLinks = []
        for i in range(len(directLinked)):
            firstDirectedLinks.append(ut.getDL_At(directLinked, sceneNumber, i, 4))
        firstLinkArray = ut.get_adjucencyMat(firstDirectedLinks, len(firstDirectedLinks))
        self.ref_first_n_link=firstLinkArray

        storiesGrouped, substoriesGrouped = ut.scenesGroup(dataframe)
        storiesGrouped.sort()
        substoriesGrouped.sort()
        firstDirectedLinks.sort()
        SceneGroupesStories = list(storiesGrouped for storiesGrouped, _ in itertools.groupby(storiesGrouped))
        SceneGroupesSubStories = list(substoriesGrouped for substoriesGrouped, _ in itertools.groupby(substoriesGrouped))
        firstDirectedLinksGroup = list(firstDirectedLinks for firstDirectedLinks, _ in itertools.groupby(firstDirectedLinks))

        self.scenes_grouped_stories=SceneGroupesStories
        self.scenes_grouped_substories=SceneGroupesSubStories
        self.scenes_grouped_first_n_links=firstDirectedLinksGroup

class Groundtruth_gran:
    def __init__(self,dataframe):
        dataframe.reset_index(drop=True)
        dataframe['Scene'] = [i for i in range(len(dataframe))]
        ref_array_stories, ref_array_Substories = ut.getLinkedDFNew(dataframe)
        self.ref_link_stories=ref_array_stories
        self.ref_link_substories=ref_array_Substories

        sceneStorycode, sceneSubStorycode = ut.getSceneStoryCode(dataframe)
        self.ref_code_stories = sceneStorycode
        self.ref_code_substories = sceneSubStorycode


        sceneNumber, directLinked = ut.getDirectLinks(dataframe=dataframe)
        firstDirectedLinks = []
        for i in range(len(directLinked)):
            firstDirectedLinks.append(ut.getDL_At(directLinked, sceneNumber, i, 4))
        firstLinkArray = ut.get_adjucencyMat(firstDirectedLinks, len(firstDirectedLinks))
        self.ref_first_n_link=firstLinkArray

        storiesGrouped, substoriesGrouped = ut.scenesGroup(dataframe)
        storiesGrouped.sort()
        substoriesGrouped.sort()
        firstDirectedLinks.sort()
        SceneGroupesStories = list(storiesGrouped for storiesGrouped, _ in itertools.groupby(storiesGrouped))
        SceneGroupesSubStories = list(substoriesGrouped for substoriesGrouped, _ in itertools.groupby(substoriesGrouped))
        firstDirectedLinksGroup = list(firstDirectedLinks for firstDirectedLinks, _ in itertools.groupby(firstDirectedLinks))

        self.scenes_grouped_stories=SceneGroupesStories
        self.scenes_grouped_substories=SceneGroupesSubStories
        self.scenes_grouped_first_n_links=firstDirectedLinksGroup

