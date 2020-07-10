"""
Online clustering of scenes based on a similairty threshold.
It try to use all scenes narrative features step by step
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import Evaluations as ev
import utility as ut
import itertools
"""
This function tries to cluster scenes on the fly (online) based on a similarity threshold values.
It takes two arguments: 1, the features to be clustered and the similarity threshold values
It return the cluster centres, features grouped in the same cluster and clustered scenes index.
"""

def clusterOnline(features,simThreshold):
    clstrCenters = [features[0]]
    clstrFeatures = [[features[0]]]
    clstrSceneNum = [[0]]
    for i in range(1, features.shape[0]):
        t=0
        for indx, cc in enumerate(clstrCenters):
            if cosine_similarity([features[i]], [cc])[0][0] > simThreshold:
                clstrFeatures[indx].extend([features[i]])
                clstrSceneNum[indx].extend([i])
                #print(len(clstrFeatures[indx]))
                clstrCenters[indx] = np.mean(np.array(clstrFeatures[indx]), axis=0)
                t=1
        if t==0:
            clstrCenters.append(features[i])
            clstrSceneNum.append([i])
            clstrFeatures.append([features[i]])
    return clstrCenters, clstrFeatures, clstrSceneNum

def cluster_subStories(rawData,clstScenNum,threshold=0.5):
    sub_strory_clusters=[]
    for scene_list in clstScenNum:
        rawdata_cltr=ut.get_spcific_rawData(rawData,scene_list)
        #simData_cltr=cosine_similarity(rawdata_cltr)
        clstrCenters, clstrFeatures, sub_clstScenNum=clusterOnline(rawdata_cltr,simThreshold=threshold)
        origin_scene_num=[]
        for s_sub in sub_clstScenNum:
            temp=[]
            for s in s_sub:
                temp.append(scene_list[s])
            origin_scene_num.append(temp)
        sub_strory_clusters.append(origin_scene_num)
    return sub_strory_clusters

def oc_combFeat(rawData,clstScenNum,comb_feat,threshold=0.9):
    sub_clusters_entities = []
    for scene_list in clstScenNum:
        rawdata_cltr = ut.get_spcific_rawData(rawData, scene_list)
        #simData_cltr = cosine_similarity(rawdata_cltr)
        clstrCenters, clstrFeatures, sub_clstScenNum = clusterOnline(rawdata_cltr, simThreshold=threshold)
        origin_scene_num = []
        for s_sub in sub_clstScenNum:
            temp = []
            for s in s_sub:
                temp.append(scene_list[s])
            origin_scene_num.append(temp)
        sub_clusters_entities.append(origin_scene_num)
    return sub_clusters_entities


def onlineCl_res(rawData,ref_array_stories,ref_array_Substories,ref_first_n_links,fileIdent=''):
    res = pd.DataFrame(
        columns=['threshold','clstr_num', 'Recall', 'Precision', 'F1_score', 'Acc', 'Recall_sub', 'Precision_sub', 'F1_score_sub',
                 'Acc_sub','Recall_n', 'Precision_n', 'F1_score_n',
                 'Acc_n'])
    for attr, value in rawData.__dict__.items():
        fileName=attr+'_'+fileIdent+'online.csv'
        print(attr)
        j=0
        for i in range(1,20):
            print(i)
            if i < 10:
                thr=i*0.1
                #clstrCenters, clstrFeatures, clstrSceneNum = clusterOnline(value, i*0.1)
            else:
                thr=0.9+((i%10)*0.01)
            print(thr)
            try:
                clstrCenters, clstrFeatures, clstrSceneNum = clusterOnline(value,thr)
                linkArrayComputed = ut.get_adjucencyMat(clstrSceneNum, value.shape[0])
                rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, ref_array_stories, upperDiagonal=True)
                rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP( linkArrayComputed,ref_array_Substories, upperDiagonal=True)
                rec_n, pre_n, f1_n, acc_n = ev.linkingArrayRP(linkArrayComputed, ref_first_n_links,upperDiagonal=True)
                results = [thr, len(clstrSceneNum),rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s,rec_n, pre_n, f1_n, acc_n]
                res.loc[j] = results
                j=j+1
            except:
                print('Error',i)
                break
        res.to_csv('Results/' + fileName)

def onlineCl_pairwise_res(rawData, scenes_grouped_stories,scenes_grouped_substories,scenes_grouped_first_n_links,fileIdent=''):
    res = pd.DataFrame(
        columns=['threshold', 'Recall', 'Precision', 'F1_score', 'Recall_sub', 'Precision_sub',
                 'F1_score_sub','Recall_n', 'Precision_n', 'F1_score_n'])
    for attr, value in rawData.__dict__.items():
        fileName = attr + '_' + 'online.csv'
        print(attr)
        j = 0
        for i in range(1, 20):
            print(i)
            if i < 10:
                thr = i * 0.1
                clstrCenters, clstrFeatures, clstrSceneNum = clusterOnline(value, i * 0.1)
            else:
                thr = 0.9 + (i * 0.001)
            #try:
            clstrCenters, clstrFeatures, clstrSceneNum = clusterOnline(value, thr)
            clstrSceneNum.sort()
            scenesRelated = list(clstrSceneNum for clstrSceneNum, _ in itertools.groupby(clstrSceneNum))
            #linkArrayComputed = ut.get_adjucencyMat(clstrSceneNum, value.shape[0])
            rec, pre, f1= ev.countpairRP(scenesRelated, scenes_grouped_stories)
            rec_s, pre_s, f1_s= ev.countpairRP(scenesRelated, scenes_grouped_substories)
            rec_n, pre_n, f1_n= ev.countpairRP(scenesRelated, scenes_grouped_first_n_links)
            results = [thr, rec, pre, f1, rec_s, pre_s, f1_s, rec_n, pre_n, f1_n]
            res.loc[j] = results
            j = j + 1
            #except:
            #break
        res.to_csv('Results/paired_' + fileName)
