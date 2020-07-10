import numpy as np
import sys
import pandas as pd
#from MyPlots import MyPlots
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import itertools
#import Clustering_Algos as ca
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import doc2vec
import clustering
import Embeddings as emb

import warnings
warnings.filterwarnings('ignore')


"""
getPosition gets the closest index of aan elemenent from a list; for example the to find the closest boundary of a scene.
This helps with the alignment of words
"""
def getPosition(man_boundry,bdry):
    closetS=min(man_boundry, key=lambda x:abs(x-bdry))
    indices = [int(i) for i, s in enumerate(man_boundry) if s==closetS]
    if len(indices)>0:
    #print(indices)
        return indices[-1]
    else:
        return 0

def getEntities(dataframe):
    nameMentions=[]
    for i in dataframe['Scene_Entities']:
        nameMentions.append(i)
    cleanNM = []
    print(nameMentions)
    for i in nameMentions:
        if i == '[]' or i=='none':
            # print('Empty')
            cleanNM.append(['none'])
        else:
            print(i)
            l = [k.split(":")[0].replace("['", "").replace("'", "") + ":" + k.split(":")[1] for k in i.split(',')]
            # print(l)
            cleanNM.append(l)

    entities = []
    for nm in cleanNM:
        tmp = []
        # print(nm)
        for n in nm:
            #print(n)
            if ',' in n and n != 'none' and n.split(':')[1] == 'PER':
                tmp.extend(n.split(':')[0].lower().strip().split(' ').replace("'","").replace("]","").strip())
            else:
                if n == 'none':
                    tmp.append(n)
                else:
                    tmp.append(n.lower().strip().replace(' ', '_').split(':')[0])
        entities.append(tmp)
    return entities

def get_adjucencyMat(grScenNUmb,totalScene):
    linkArray=np.zeros((totalScene,totalScene))
    for gr in grScenNUmb:
        for i in gr:
            for j in gr:
                #print(i,j)
                linkArray[i][j]=1
    return linkArray

def getNormalizedEntities(dataframe):
    nameMentions = []
    for i in dataframe['Scene_Entities']:
        nameMentions.append(i)
    entities = []
    for i in nameMentions:
        if i == '[]' or i == 'none':
            # print('Empty')
            entities.append([])
        else:
            #print(i)
            l = [k.replace("['", "").replace("]'", "").replace("'", "").replace("]", "").strip() for k in i.split(',')]
            # print(l)
            entities.append(l)
    return entities

def get_groupedSceneNum(groupedScenes,episodeNumb):
    grScenNUmb=[]
    for gr in groupedScenes:
        grTemp=[]
        for i in gr:
            #print(i)
            if int(i[3:5])-1!=0:
                grTemp.append(int(i.split('S')[1])-1)
            else:
                grTemp.append(int(i.split('S')[1]))
        grScenNUmb.append(grTemp)
    linkArray=get_adjucencyMat(grScenNUmb,episodeNumb)
    return grScenNUmb,linkArray

def groupedSceneCls(clsData,dataframe):
    uni=list(set(clsData))
    goupedClusters=[]
    for c in uni:
        group=[]
        for i in range(len(clsData)):
            if c == clsData[i]:
                #print(c,clsData[i],i)
                group.append(i)
        goupedClusters.append(group)

    groupedScenesNam=[]
    for i in goupedClusters:
        group=[]
        for j in i:
            k=int(j)
            if dataframe['Specific_Stories'][k]!='Non':
                group.append('0'+str(dataframe['Season'][k])+'E0'+str(dataframe['Episode'][k])+'S'+str(dataframe['Scene'][k]))
        groupedScenesNam.append(group)

    return groupedScenesNam,goupedClusters

def getLinkedDFNew(dfread):
    dfread=dfread.reset_index(drop=True)
    #print(dfread)
    ref_array_stories=np.zeros((len(dfread),len(dfread)))
    ref_array_Substories=np.zeros((len(dfread),len(dfread)))
    for i in range(len(dfread)):
        storyList=dfread['Specific_Stories'][i].split(',')
        sceneStories=[item.split('-')[0] for item in storyList]
        sceneSubStories=[item.split('-')[1:] if len(item.split('-'))>1 else item.split('-')  for item in storyList]
        sceneSubStories=[it for sublist in sceneSubStories for it in  sublist]
        for j in range(i,len(dfread)):
            if i==j:
                ref_array_stories[i][j]=1
                ref_array_Substories[i][j]=1
            else:
                for ss in sceneStories:
                    #print(ss,j)
                    #print(str(i)+' '+str(j)+' '+ss.strip()+':'+dfread['Scene Titles'][j]+':'+str(ss.strip() in dfread['Scene Titles'][j]))
                    if ss.strip() in dfread['Specific_Stories'][j]:
                        ref_array_stories[i][j]=1
                        ref_array_stories[j][i]=1
                        break
                    else:
                        continue
                #print('substories')
                #print('here')
                for sss in sceneSubStories:
                    #print(str(i)+' '+str(j)+' '+sss.strip()+':'+dfread['Scene Titles'][j].strip()+':'+str(sss.strip() in dfread['Scene Titles'][j]))
                    if sss.strip() in dfread['Specific_Stories'][j]:
                        ref_array_Substories[i][j]=1
                        ref_array_Substories[j][i]=1
                        break
                    else:
                        continue
    return ref_array_stories, ref_array_Substories

def getGranularity(df,season,episode=None):
    if episode!=None:
        dfGran=df[(df['Season']==season) & (df['Episode']==episode)]
    else :
        dfGran=df[df['Season']==season]

    #dfGran.reset_index(drop=True)
    dfGran=dfGran.reset_index()
    start=dfGran['Unnamed: 0'][0]
    end=dfGran['Unnamed: 0'][len(dfGran)-1]+1
    return dfGran, start, end

def getGranularity_Res(df,season,episode=None):
    dfGran=df
    if episode!=None:
        dfGran=dfGran[(dfGran['Season']==season) & (dfGran['Episode']==episode)]
    else :
        dfGran=dfGran[dfGran['Season']==season]

    #dfGran.reset_index(drop=True)
    dfGran=dfGran.reset_index()
    linkArray, linkArray_Sub=getLinkedDFNew(dfGran)

    speakingChars=dfGran['Speaking_Chararacters'].tolist()
    speakingChars=[i.split(',') for i in speakingChars]
    starts=dfGran['Scene_Start'].tolist()
    end=dfGran['Scene_End'].tolist()
    ext_Chars_df=dfGran['Appearing_Characters'].tolist()
    ext_Chars_df=[str(i).split(',') for i in ext_Chars_df]
    keywords=dfGran['Scene_Keybords']
    #print(keywords)
    keywords=[i.split(',') if not pd.isnull(i) else 'none' for i in keywords]
    sceneTranscript=dfGran['Scene_Texts'].tolist()
    doc2vec_Emb=getSceneEmbeding(sceneTranscript)


    nameMentions=[]
    for i in dfGran['Scene_Entities']:
        nameMentions.append(i)
    cleanNM=[]
    for i in nameMentions:
        if i=='[]':
            #print('Empty')
            cleanNM.append(['none'])
        else:
            #print(i.split(','))
            l=[k.split(":")[0].replace("['","").replace(" '","")+":"+k.split(":")[1][0:3] for k in i.split(',')]
            #print(l)
            cleanNM.append(l)

    entities=[]
    for nm in cleanNM:
        tmp=[]
        #print(nm)
        for n in nm:
            #print(n)
            if ',' in n and n!='none' and n.split(':')[1]=='PER':
                tmp.extend(n.split(':')[0].lower().strip().split(' '))
            else:
                if n=='none':
                    tmp.append(n)
                else:
                    tmp.append(n.lower().strip().replace(' ','_').split(':')[0])
        entities.append(tmp)

    x=dfGran['Unnamed: 0'][0]
    y=dfGran['Unnamed: 0'][len(dfGran)-1]

    EpisodeRowData={}
    existingChars_data,sceneAN=list_oneHot_encode(ext_Chars_df)
    speakingChars_data,sceneSN=list_oneHot_encode(speakingChars)
    sceneKeywords,sceneTFIDF=list_oneHot_encode(keywords)
    sceneNameMentions,sceneNM=list_oneHot_encode(entities)
    doc2vec=np.load('Scene_Doc2Vec_Embedding.npy')
    EpisodeRowData['oneHotApp']=existingChars_data
    EpisodeRowData['oneHotSpk']=speakingChars_data
    EpisodeRowData['oneHotKeyWords']=sceneKeywords
    EpisodeRowData['oneHotEntitie']=sceneNameMentions
    EpisodeRowData['doc2vec']=doc2vec_Emb#doc2cev[x:y,x:y]

    combApp_Ent=np.concatenate((existingChars_data,sceneNameMentions),axis=1)
    combSpk_Ent=np.concatenate((speakingChars_data,sceneNameMentions),axis=1)
    combApp_KW=np.concatenate((existingChars_data,sceneKeywords),axis=1)
    combSpk_KW=np.concatenate((speakingChars_data,sceneKeywords),axis=1)
    combSpk_Ent_KW=np.concatenate((combSpk_Ent,sceneKeywords),axis=1)
    combApp_Ent_KW=np.concatenate((combApp_Ent,sceneKeywords),axis=1)
    combSpk_KW2=np.concatenate((existingChars_data,2*sceneKeywords),axis=1)
    combApp_KW2=np.concatenate((speakingChars_data,2*sceneKeywords),axis=1)

    simAvgSentenceEmb=np.load('/vol/work3/berhe/SceneLinking/SentenceEmbeddings/Average_Sentences_Embedding_Similarity_New.npy')
    data={}

    data['sceneDf']=dfGran
    data['simApp_Characters']=cosine_similarity(existingChars_data)
    data['simSpk_Characters']=cosine_similarity(speakingChars_data)
    data['simSceneKeywords']=cosine_similarity(sceneKeywords)
    data['simSceneNameMentions']=cosine_similarity(sceneNameMentions)
    data['simAvgSent']=simAvgSentenceEmb[x:y,x:y]
    #data['doc2vec']=cosine_similarity(doc2vec[x:y,x:y])

    data['simcombSpk_Ent']=cosine_similarity(combSpk_Ent)
    data['simcombSpk_KW']=cosine_similarity(combSpk_KW)
    data['simcombSpk_Ent_KW']=cosine_similarity(combSpk_Ent_KW)
    data['simcombApp_Ent']=cosine_similarity(combApp_Ent)
    data['simcombApp_KW']=cosine_similarity(combApp_KW)
    data['simcombApp_Ent_KW']=cosine_similarity(combApp_Ent_KW)
    data['simcombApp_KW2']=cosine_similarity(combApp_KW2)
    data['simcombSpk_KW2']=cosine_similarity(combSpk_KW2)

    data['linkArray']=linkArray
    data['linkArray_Sub']=linkArray_Sub



    return data,EpisodeRowData

def getDirectLinks(dataframe):
    links=dataframe['Scene_Links'].tolist()
    directLinkled=[]
    sceneNumber=[]
    for i in range(len(links)):
        linked=links[i].split(',')
        retList=[]
        for j in linked:
            #print(j)
            if j!='Non':
                s=int(j.strip()[0])
                e=int(j.strip()[2:4])
                scene=int(j.split('S')[1])
                lst=dataframe.Scene[(dataframe['Season']==s) & (dataframe['Episode']==e) & (dataframe['Scene_on_Video']==scene)].tolist()
                #print(lst)
                if lst!=[]:
                    retList.append(lst[0])
                else:
                    retList.append('Non')
            else:
                retList.append('Non')
        directLinkled.append(retList)
        sceneNumber.append(i)
    return sceneNumber,directLinkled

def getDL_At(directLinked,sceneNumber,scene,firstLink=2):
    #sceneNumber,directLinkled=getDirectLinks(dataframe)
    linkAt=[sceneNumber[scene]]
    for i in range(firstLink):
        #print(sceneNumber[scene])
        if directLinked[scene][0]=='Non':
            return linkAt
        else:
            linkAt.append(directLinked[scene][0])
            scene=sceneNumber.index(directLinked[scene][0])
    return linkAt

#Check if a scene belong to a group based on the feature and threshold value
def belongIn(eltpos,sceneCluster,data,threshold=0.5):
    for sc in sceneCluster:
        if round(cosine_similarity([data[eltpos]],[data[sc]])[0][0],3)>threshold:
            return True
    return False
#Check if a scene belong to a group based on the feature and threshold value, and other features related ness (commonfeature)
def belongIn2(eltpos,sceneCluster,threshold=0.5):
    for sc in sceneCluster:
        cosSimSpk=round(cosine_similarity([dataRow.oneHotSpk[eltpos]],[dataRow.oneHotSpk[sc]])[0][0],3)
        cosSimKey=round(cosine_similarity([dataRow.oneHotKeywords[eltpos]],[dataRow.oneHotKeywords[sc]])[0][0],3)
        cosSimEnt=round(cosine_similarity([dataRow.oneHotEntities[eltpos]],[dataRow.oneHotEntities[sc]])[0][0],)
        cosSimD2V=round(cosine_similarity([dataRow.d2v[eltpos]],[dataRow.d2v[sc]])[0][0],2)
        if cosSimSpk>threshold and cosSimEnt>threshold:
            return True
    return False

#get scene goupings based on a threshold value and feature
def getSceneCluster(length,threshold=0.5):
    sceneCluster=[[0]]
    for i in range(1,length):
        t=0
        for j in sceneCluster:
            if belongIn2(i,j,threshold):
                j.append(i)
                t=1
        if t==0:
            sceneCluster.append([i])
    lArray=get_adjucencyMat(sceneCluster,length)
    return lArray,sceneCluster

#get scene goupings based on a threshold value and feature, and considering other features
def getSceneCluster2(length,data,threshold=0.5):
    sceneCluster=[[0]]
    for i in range(1,length):
        t=0
        for j in sceneCluster:
            if belongIn(i,j,data,threshold):
                j.append(i)
                t=1
        if t==0:
            sceneCluster.append([i])
    lArray=get_adjucencyMat(sceneCluster,length)
    return lArray,sceneCluster

def sceneBasedInfo(scene,data,threshold=0.5):
    relatedScene=[]
    for i in range(scene,data.shape[0]):
        if round(cosine_similarity([data[scene]],[data[i]])[0][0],3)>threshold:
            relatedScene.append(i)
    return relatedScene

def sceneBasedFeatures(feature,sc,eltpos,dataRow):
    cosSimSpk=round(cosine_similarity([dataRow["oneHotSpk"][eltpos]],[dataRow["oneHotSpk"][sc]])[0][0],3)
    cosSimKey=round(cosine_similarity([dataRow["oneHotKeyWords"][eltpos]],[dataRow["oneHotKeyWords"][sc]])[0][0],3)
    cosSimEnt=round(cosine_similarity([dataRow["oneHotEntitie"][eltpos]],[dataRow["oneHotEntitie"][sc]])[0][0],)
    cosSimD2V=round(cosine_similarity([dataRow["doc2vec"][eltpos]],[dataRow["doc2vec"][sc]])[0][0],2)
    if feature=='s':
        if cosSimEnt>0.0 or cosSimKey>0.0:
            return True
        else:
            return False
    elif feature=='e':
        if cosSimEnt>0.0 or cosSimKey>0.0:
            return True
        else:
            return False
    elif feature=='k':
        if cosSimEnt>0.0 or cosSimSpk>0.0:
            return True
        else:
            return False
    else:
        if cosSimEnt>0.0 or cosSimKey>0.0 or cosSimSpk>0.0:
            return True
        else:
            return False

def sceneBasedInfo2(scene,data,alldata,f='s',threshold=0.5):
    relatedScene=[]
    for i in range(scene,data.shape[0]):
        if round(cosine_similarity([data[scene]],[data[i]])[0][0],3)>threshold and sceneBasedFeatures(f,scene,i,alldata):
            relatedScene.append(i)
    return relatedScene

def getsceneRelatedness(data,threshold=0.5):
    scenesRelated=[]
    for i in range(data.shape[0]):
        relatedScene=sceneBasedInfo(i,data,threshold)
        scenesRelated.append(relatedScene)
    linkingArray=get_adjucencyMat(scenesRelated,len(scenesRelated))
    return scenesRelated,linkingArray

def getsceneRelatedness2(data,alldata, f='s',threshold=0.5):
    scenesRelated=[]
    for i in range(data.shape[0]):
        relatedScene=sceneBasedInfo2(i,data,alldata,f,threshold)
        scenesRelated.append(relatedScene)
    linkingArray=get_adjucencyMat(scenesRelated,len(scenesRelated))
    return scenesRelated,linkingArray

def groupedScene_Sk_Fuzzy(clsData,df):
    cltrs=[item for sublist in clsData for item in sublist]
    uni=list(set(cltrs))
    goupedClusters=[]
    for c in clsData[0]:
        group=[]
        for i in range(len(clsData)):
            if c in clsData[i]:
                #print(c,clsData[i],i)
                group.append(i)
            else:
                continue
        goupedClusters.append(group)

    sceneTitleDF=df#pd.read_csv('/people/berhe/Bureau/TLP_thesis/codes/Scene_Linking_project/Scenes_Dataset_New_Keywords.csv')
    #print(goupedClusters,len(goupedClusters))
    groupedScenesNam=[]
    for i in goupedClusters:
        group=[]
        for j in i:
            k=int(j)
            if sceneTitleDF['Specific_Stories'][k]!='Non':
                group.append('0'+str(sceneTitleDF['Season'][k])+'E0'+str(sceneTitleDF['Episode'][k])+'S'+str(sceneTitleDF['Scene'][k]))
        groupedScenesNam.append(group)
    return groupedScenesNam,goupedClusters

def fuzzygrouping(membershipList,df):
    uni=[i for sublist in membershipList for i in sublist]
    uni=list(set(uni))
    grouped=[]
    for i in range(len(uni)):
        groupTemp=[]
        for s in range(len(membershipList)):
            if uni[i] in membershipList[s]:
                groupTemp.append(s)
        grouped.append(groupTemp)

    sceneTitleDF=df
    #print(goupedClusters,len(goupedClusters))
    groupedScenesNam=[]
    for i in grouped:
        group=[]
        for j in i:
            k=int(j)
            if sceneTitleDF['Specific_Stories'][k]!='Non':
                group.append('0'+str(1)+'E0'+str(1)+'S'+str(k))
        groupedScenesNam.append(group)

    return groupedScenesNam,grouped

def storywise_group(dataframe):
    stories = []
    substories = []
    for i in range(len(dataframe)):
        storyList = dataframe['Specific_Stories'][i].split(',')
        scene_stories = [item.split('-')[0].strip() for item in storyList]
        stories.append(scene_stories)

        scene_substories = [item.split('-')[1].strip() if '-' in item else [] for item in storyList]
        substories.append(scene_substories)

    unique_stories = [i.strip() for sublist in stories for i in sublist]
    unique_Substories = [i.strip() for sublist in substories for i in sublist if i]

    unique_stories=list(set(unique_stories))
    unique_Substories=list(set(unique_Substories))

    storiesGrouped = [[]] * len(unique_stories)
    for i, st in enumerate(unique_stories):
        for scNum, scSt in enumerate(stories):
            if st in scSt:
                storiesGrouped[i] = storiesGrouped[i] + [scNum]
    substoriesGrouped = [[]] * len(unique_Substories)
    for i, st in enumerate(unique_Substories):
        for scNum, scSt in enumerate(substories):
            if st in scSt:
                substoriesGrouped[i] = substoriesGrouped[i] + [scNum]

    return storiesGrouped,substoriesGrouped,stories,substories
def scenesGroup(df):
    storiesGrouped=[]
    substoriesGrouped=[]
    for i in range(len(df)):
        storyList=df['Specific_Stories'][i].split(',')
        sceneStories=[item.split('-')[0] for item in storyList]
        sceneSubStories=[item.split('-')[1:] if len(item.split('-'))>1 else item.split('-')  for item in storyList]
        sceneSubStories=[it for sublist in sceneSubStories for it in  sublist]
        for z in sceneStories:
            tempstory=[]
            for j in range(len(df)):
                if z in df['Specific_Stories'][j]:
                    tempstory.append(j)
            storiesGrouped.append(tempstory)
        for z in sceneSubStories:
            tempSub=[]
            for j in range(len(df)):
                if z in df['Specific_Stories'][j]:
                    tempSub.append(j)
            substoriesGrouped.append(tempSub)

    return storiesGrouped,substoriesGrouped

#getting Story and Sub Stories codes : Results will be used for regular clustering metrics.

def getSceneStoryCode(dataframe):

    """
    getting Story and Sub Stories codes : Results will be used for regular clustering metrics.
    """
    stories=dataframe['Specific_Stories'].tolist()
    st = [i.split(',')[0].split('-')[0].strip() for i in stories]
    subSt = [i.split(',')[0].split('-')[1].strip() if '-' in i.split(',')[0].strip() else i.split(',')[0].strip() for i in stories]
    uniStories = list(set(st))
    uniSubStories = list(set(subSt))

    sceneStorycode=[]
    sceneSubStorycode=[]
    for i in stories:
        story=i.split(',')[0].split('-')[0].strip()
        if '-' in i.split(',')[0]:
            subStory=i.split(',')[0].split('-')[1].strip()
        else:
            subStory=i.split(',')[0].strip()
        #try:
        sceneStorycode.append(uniStories.index(story))
        sceneSubStorycode.append(uniSubStories.index(subStory))
        #except:
            #print('error',i)
    return sceneStorycode, sceneSubStorycode

def get_spcific_rawData(rawData,scene_list):
    rawData_specific=[]
    for i in scene_list:
        rawData_specific.append(rawData[i])
    return np.array(rawData_specific)

def cluster_postprecsess(algo,cluster_num,data,dataframe,simThreshold=0.5):
    if 'skfuzzy' in algo:
        avgValue, uniPredValues, scenemembership = clustering.getClusters(algo, cluster_num, data)
        groupedScenesNam, goupedClusters = fuzzygrouping(scenemembership, dataframe)
    else:
        computedClusters = clustering.getClusters(algo, cluster_num, data)
        groupedScenesNam, goupedClusters = groupedSceneCls(computedClusters, dataframe)
    print(data.shape)
    grdScenes, linkArrayComputed = get_groupedSceneNum(groupedScenesNam, data.shape[0])
    grdScenes.sort()
    scenesRelated = list(grdScenes for grdScenes, _ in itertools.groupby(grdScenes))
    return scenesRelated,linkArrayComputed
"""
This build the co-occurence matrix from a list of grouped scenes. It retuns a matrix of co-occurence.
"""
def cooccurence_matrix(groupedScenes):
    u = (pd.get_dummies(pd.DataFrame(groupedScenes), prefix='', prefix_sep=''),groupby(level=0, axis=1).sum())
    v=u.T.dot(u)
    return v


"""
The following function are used to generate links form the scene linking column.
and recursively go from a scene to its previous linked scene.
"""
def getLinked_all(dataframe):
    links = dataframe.Scene_Links.tolist()
    scenesLinks=[]
    for i,l in enumerate(links):
        for ll in l.split(','):
            if ll == 'Non':
                #print(i,ll, 'None')
                scenesLinks.append((i, 'Non'))
            else:
                #print(i,ll)
                #print(i, dataframe.Scene[(dataframe.Scene_on_Video==int(ll[5:]))&(dataframe.Episode==int(ll[2:4]))&(dataframe.Season==int(ll[0]))].tolist())
                a=dataframe.Scene[(dataframe.Scene_on_Video==int(ll[5:]))&(dataframe.Episode==int(ll[2:4]))&(dataframe.Season==int(ll[0]))].tolist()
                if a==[]:
                    aa='Non'
                else:
                    aa=a[0]
                scenesLinks.append((i, aa))
    episode_links = [[]] * len(dataframe)
    for i, c in scenesLinks:
        #print(i, c)
        episode_links[i] = episode_links[i] + [c]
    episode_links = [i[0:len(i) - 1] if 'Non' in i and len(i) != 1 else i for i in episode_links]
    return episode_links,scenesLinks

def getLinked_previous(dataframe,season=1,episode=1):
    #dfGran, start, end = getGranularity(dataframe, season=season, episode=episode)
    links = dataframe.Scene_Links.tolist()
    original_scene = dataframe.Scene_on_Video.tolist()
    scenesLinks = []
    for i, l in enumerate(links):
        for ll in l.split(','):
            if ll == 'Non':
                print(i, 'None')
                scenesLinks.append((i, 'Non'))
            elif int(ll[5:]) not in original_scene:
                print(i, 'None')
                scenesLinks.append((i, 'Non'))
            else:
                print(i, original_scene.index(int(ll[5:])))
                scenesLinks.append((i, original_scene.index(int(ll[5:]))))
    episode_links = [[]] * len(dataframe)
    for i, c in scenesLinks:
        print(i, c)
        episode_links[i] = episode_links[i] + [c]
    episode_links = [i[0:len(i) - 1] if 'Non' in i and len(i) != 1 else i for i in episode_links]
    return episode_links

def recurse_link(sc,n):
    if sc[n]==[]:
        return []
    if sc[n][0] =='Non' :
        return []
    else:
        #print(n,sc[n])
        return [n]+sc[n] + recurse_link(sc[0:sc[n][0]+1],sc[n][0])

def get_recursiveLinks_unique(dataframe):
    episode_links, scenesLinks = getLinked_all(dataframe)
    rec_links = []
    for i in range(len(dataframe)):
        l = list(set(recurse_link(episode_links, i)))
        rec_links.append(l)
    rec_links = [i for i in rec_links if i != []]
    unique_links = [[]]
    for i, links in enumerate(rec_links):
        #print('length ', len(unique_links))
        t = 0
        for j, ul in enumerate(unique_links):
            if set(ul) <= set(links):
                unique_links[j] = links
                t = 1
                break
        if t != 1:
            unique_links.append(links)
            #print(ul, links)
    linkedArray_rec = get_adjucencyMat(unique_links, len(dataframe))
    return unique_links,linkedArray_rec
"""
episode_links,scenesLinks
rec_links=[]
for i in range(len(dataframe)):
    l=list(set(recurse_link(episode_links, i)))
    rec_links.append(l)
"""

"""
Get all paired links between episodes 
"""


def pairedLinkesBetweenEpisodes(dataframe, unique_links):
    links_bn_episodes = []
    link_e7_e8 = []
    links_bn_episodes_cl = []
    for s in range(1, 3):
        for e in range(1, 10):
            # for ee in range(e+1,11):
            # print("episode {} and episode {}".format(e,e))
            s1 = dataframe.Scene[(dataframe.Season == s) & (dataframe.Episode == e)].tolist()
            s2 = dataframe.Scene[(dataframe.Season == s) & (dataframe.Episode == e + 1)].tolist()
            print("episode {} from {} {} and episode {} from {} {}".format(e, s1[0], s1[-1], e + 1, s2[0], s2[-1]))

            link_e7_e8 = []
            for i in unique_links:
                t1 = 0
                t2 = 0
                for j in i:
                    if s1[0] < j < s1[-1]:
                        t1 = 1
                        break
                for j in i:
                    if s2[0] < j < s2[-1]:
                        t2 = 1
                        break
                if t1 == 1 and t2 == 1:
                    link_e7_e8.append(i)
            #print(link_e7_e8)
            link_e7_e8_found = []
            for in_link in link_e7_e8:
                link_e7_e8_found.append([sc for sc in in_link if s1[0] <= sc and sc <= s2[-1]])
            pair_link_bn_episodes = []
            for i in link_e7_e8_found:
                # print("Printing I ", i)
                a = i
                a.sort()
                further = getPosition(a, s1[-1])
                closer = getPosition(a, s2[0])
                #print(closer, further)
                if further != closer:
                    link = [a[further], a[closer]]
                else:
                    if a[closer] >= s2[0]:
                        link = [a[further - 1], a[closer]]
                    else:
                        link = [a[further - 1], a[closer + 1]]
                #print(a, link, a[further], a[closer], s1[-1], s2[0])
                pair_link_bn_episodes.append(link)
            clear_pair = []
            for i in pair_link_bn_episodes:
                if i not in clear_pair:
                    clear_pair.append(i)
            links_bn_episodes_cl.append(link_e7_e8_found)
            links_bn_episodes.append(clear_pair)

    return links_bn_episodes,links_bn_episodes_cl

def getpairedlinks_2_episodes(dataframe,unique_links,season=1,episode=1):
    s1 = dataframe.Scene[(dataframe.Season == season) & (dataframe.Episode == episode)].tolist()
    s2 = dataframe.Scene[(dataframe.Season == season) & (dataframe.Episode == episode+1)].tolist()
    #print(s1[0],s[-1],s2[0],s2[-1])
    link_bn_eps = []
    for i in unique_links:
        t1 = 0
        t2 = 0
        for j1 in i:
            if s1[0] < j1 < s1[-1]:
                t1 = 1
                break
        for j2 in i:
            if s2[0] < j2 < s2[-1]:
                t2 = 1
                break
        if t1 == 1 and t2 == 1:
            link_bn_eps.append(i)
    print(link_bn_eps)
    link_bn_eps_found = []
    for in_link in link_bn_eps:
        link_bn_eps_found.append([sc for sc in in_link if s1[0] <= sc <= s2[-1]])
    pair_link_bn_episodes = []
    #print(link_bn_eps_found)
    for i in link_bn_eps_found:
        #print("Printing I ", i)
        a = i
        a.sort()
        further = getPosition(a, s1[-1])
        closer = getPosition(a, s2[0])
        #print(closer, further)
        if further != closer and a[further]<s2[0]:
            link = [a[further], a[closer]]
        else:
            if a[closer] >= s2[0]:
                link = [a[further - 1], a[closer]]
            else:
                link = [a[further], a[closer + 1]]
        #print(a, link, a[further], a[closer], s1[-1], s2[0])
        pair_link_bn_episodes.append(link)
    clear_pair = []
    for i in pair_link_bn_episodes:
        if i not in clear_pair:
            clear_pair.append(i)

    return clear_pair,link_bn_eps