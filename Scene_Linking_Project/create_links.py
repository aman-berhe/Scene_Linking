import numpy as np
import pandas as pd
import Embeddings as emb
import utility as ut
import onlineClustering as oc
import load_features as lf
import Evaluations as ev
def getLinks_otherfeat(data, episode_clusters, threshold=0.4):
    linkedClusters = []
    # episode_loc_onehot=emb.list_oneHot_encode(episode_locations)
    episode_entities = []
    for clstr in episode_clusters:
        tmp = []
        for scene in clstr:
            tmp.append(data[scene])
        episode_entities.append(np.mean(np.array(tmp), axis=0))

    for i in range(len(episode_entities) - 1):
        for j in range(i + 1, len(episode_entities)):
            avgSim = emb.cosine_similarity(episode_entities[i], episode_entities[j])
            if avgSim >= threshold:
                print(i, j, avgSim)
                linkedClusters.append(list(set((episode_clusters[i] + episode_clusters[j]))))
    return linkedClusters


def getLinksIoU(episode_locations,episode_clusters):
    linkedClusters=[]
    #episode_loc_onehot=emb.list_oneHot_encode(episode_locations)
    for i in range(len(episode_clusters)-1):
        for j in range(i+1,len(episode_clusters)):
            iOu=len(list(set(episode_locations[i]).intersection(episode_locations[j])))/(len(episode_locations[i]+episode_locations[j]))
            if iOu>=0.5:
                print(i, j, iOu)
                linkedClusters.append(list(set((episode_clusters[i]+episode_clusters[j]))))
    return linkedClusters

def getepisodeOnlineClusters(dataframe,season=1,episode=1,feat='sp_char',threshold=0.7):
    dfGran, start, end = ut.getGranularity(dataframe, season=season, episode=episode)
    rawData_gran = lf.Rawdata_Gran(dfGran, start, end)
    simData_Gran = lf.Simdata(rawData_gran)
    groundTruth_gran = lf.Groundtruth_gran(dfGran)
    feat=rawData_gran.__getattribute__(feat)
    clstrCenters, clstrFeatures, clstrSceneNum = oc.clusterOnline(feat, threshold)
    return clstrCenters, clstrFeatures, clstrSceneNum, len(dfGran)

def getEpisodeSubClusters(episode_clusters,rawData_4_sub):
    sub_clstrs=[]
    for i in episode_clusters:
        pos=i
        cltr_data=ut.get_spcific_rawData(rawData_4_sub,i)
        clstrCenters, clstrFeatures, sub_clstr=oc.clusterOnline(cltr_data,simThreshold=0.2)
        org_scene=[]
        for s in sub_clstr:
            tmp=[]
            for sc in s:
                tmp.append(pos[sc])
            org_scene.append(tmp)
        sub_clstr.append(org_scene)
    return sub_clstr



def get_cluster_locations(dataframe,clstrSceneNum):
    cluster_loc=[]
    for scenes in clstrSceneNum:
        tmp=[]
        for scene in scenes:
            #print(scene,type(scene))
            tmp.append(dataframe['Scene_Locations'][scene])
        cluster_loc.append(list(set(tmp)))
    return cluster_loc

def get_All_episode_cluster(dataframe,threshold=0.7):
    episode_clusters = []
    episode_locations = []
    for s in range(1, 3):
        for e in range(1, 11):
            clstrCenters, clstrFeatures, clstrSceneNum,len= getepisodeOnlineClusters(dataframe, season=s, episode=e,threshold=threshold)
            cluster_loc = get_cluster_locations(dataframe, clstrSceneNum)
            episode_clusters.append(clstrSceneNum)
            episode_locations.append(cluster_loc)
    #episode_clusters = [i for sublist in episode_clusters for i in sublist]
    episode_locations = [i for sublist in episode_locations for i in sublist]
    return episode_clusters, episode_locations

def cluster_onlinecluster(data,clstrSceneNum,threshold):
    clstr_clusters=[]
    for clstr in clstrSceneNum:
        clstr_entities = []
        pos=[]
        for scene in clstr:
            clstr_entities.append(data[scene])
            pos.append(scene)
        clstr_entities=np.array(clstr_entities)
        clstrCenters, clstrFeatures, clstrSceneNum2 = oc.clusterOnline(clstr_entities, threshold)
        scen_origin=[]
        for i in clstrSceneNum2:
            tmp=[]
            for j in i:
                tmp.append(pos[j])
            scen_origin.append(tmp)
        clstr_clusters.append(scen_origin)
    clstr_clusters=[i for sublist in clstr_clusters for i in sublist]
    return clstr_clusters

def link_window_episode(episodes_clusters,data_4_linking,current_episode=0,window_size=1):
    windowlinked_clusters=[]
    for i in range(len(episode_clusters)):
        for j in range(i,len(episode_clusters)-1):
            linkedEpisodes=link2episode(episode_clusters[i],episode_clusters[j],data_4_linking)
            windowlinked_clusters.append(linkedEpisodes)
    return windowlinked_clusters

def merge_clusters(episode_clusters,start_episode=0,window_size=1):
    assert window_size>0, "Window size should be greater than zero"
    assert start_episode-window_size <0, "window size too big"
    assert start_episode+window_size >len(episode_clusters), "window size too big"
    merged_clusters=[]
    for i in range(window_size):
        computed_links=compute_clusters_links(episode_clusters[i],episode_clusters[start_episode],rawData,threshold=0.5)
        merged_clusters.append(computed_links)

    merged_clusters=[i for i in merge_clusters]

    return merged_clusters

def merge_using_centers(episodes_clusters,cluster_centres,threshold):
    merged_clstrs=[]
    for i in range(len(cluster_centres)-1):
        tmp_merge=[]
        for j in range(i,len(cluster_centres)):
            if emb.cosine_similarity(cluster_centre[i],cluster_centres[j])[0][0] > threshold:
                tmp_merge.append(episodes_clusters[i]+episodes_clusters[j])
        if tmp_merge==[]:
            merged_clstrs.append(cluster_centres[i])
        else:
            merged_clstrs.append(tmp_merge)
    return merged_clstrs

def get_linking_results(clsuters,ref_groundtruth,shape=444):
    linkArrayComputed = ut.get_adjucencyMat(clsuters, shape)
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, ref_groundtruth, upperDiagonal=False)
    return [rec,pre,f1,acc]

def get_merge_result(ctrs_corr,rawdata,groundTruth):
    data = []
    for i in ctrs_corr:
        data.append(ut.get_spcific_rawData(rawdata, i))
    centers = []
    for i in data:
        centers.append(np.mean(i, axis=0))
    for i in range(10):
        merged_clstrs = merge_using_centers(ctrs_corr, centers, threshold=i * 0.1)
        merged_clstrs = [i for sublist in merged_clstrs for i in sublist]
        linkArrayComputed = ut.get_adjucencyMat(merged_clstrs, rawData.entities.shape[0])
        rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_link_stories, upperDiagonal=False)
        rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_link_substories,
                                                  upperDiagonal=False)
        print("{} & {} & {} & {} & {} & {} & {} & {}".format(i * 0.1, len(merged_clstrs), rec, pre, f1, rec_s, pre_s, f1_s))
        print('\hline')

def get_window_merging(ctrs_corr,rawdata,groundTruth):
    data = []
    for i in ctrs_corr:
        data.append(ut.get_spcific_rawData(rawdata, i))
    centers = []
    for i in data:
        centers.append(np.mean(i, axis=0))
    for i in range(1, 10):
        windowlinked_clusters = link_window_episode(ctrs_corr, centers, current_episode=0, window_size=5,threshold=i * 0.1)
        linkArrayComputed = ut.get_adjucencyMat(windowlinked_clusters, rawData.entities.shape[0])
        rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_link_stories, upperDiagonal=False)
        rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_link_substories, upperDiagonal=False)
        print("{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(i, len(windowlinked_clusters), rec, pre, f1, rec_s,pre_s, f1_s))
        print('\hline')


def get_cluster_link(cluster1,cluster2,threshold=0.5):
    scene_pair_link=[]
    for c1 in cluster1:
        for c2 in cluster2:
            if emb.cosine_similarity(c1.reshape(1,-1),c2.reshape(1,-1))[0][0]>threshold:
                scene_pair_link.append((c1,c2))
    return scene_pair_link

def create_cluster_link(cluster1Data,cluster2Data,threshold=0.5):
    scene_pair_link=[]
    for i,c1 in enumerate(cluster1Data):
        for j,c2 in enumerate(cluster2Data):
            if emb.cosine_similarity(c1.reshape(1,-1),c2.reshape(1,-1))[0][0]>threshold:
                scene_pair_link.append((i,j))
    return scene_pair_link

"""
dfGran, start, end = ut.getGranularity(normDF, season=1, episode=1)
links=dfGran.Scene_Links.tolist()
original_scene=dfGran.Scene_on_Video.tolist()
scenesLinks=[]
for i ,l in enumerate(links):
    for ll in l.split(','):
        if ll=='Non':
            print(i,'None')
            scenesLinks.append((i,'Non'))
        elif int(ll[5:]) not in original_scene:
            print(i, 'None')
            scenesLinks.append((i, 'Non'))
        else:
            print(i,original_scene.index(int(ll[5:])))
            scenesLinks.append((i, original_scene.index(int(ll[5:]))))
sc=[[]]*len(dfGran) 
for i,c in scenesLinks:
    print(i,c)
    sc[i]=sc[i]+[c]
    sc=[i[0:len(i)-1] if 'Non' in i and len(i)!=1 else i for i in sc]
"""

def getclusterSim(cltr_data1,cltr_data2,clstrs1,clstrs2,threshold=0.5):
    links1=[]
    #print(len(cltr_data1),len(cltr_data2))
    simlist = []
    for i in range(len(cltr_data1)):
        scenes_sim = []
        links=[]
        for j in range(len(cltr_data2)):
            #print(i,j)
            #print(cltr_data1[i].reshape(-1, 1),cltr_data1[i].shape)
            sceneSim=round(emb.cosine_similarity(cltr_data1[i].reshape(1,-1),cltr_data2[j].reshape(1,-1))[0][0],3)
            scenes_sim.append(sceneSim)
            #simlist.append(sceneSim)
            #print(sceneSim)
        #print("max similairty ",max(scenes_sim))
        simlist.append(max(scenes_sim))
        links1.append(clstrs2[scenes_sim.index(max(scenes_sim))])
    #print(simlist,links1,simlist.index(max(simlist)))
    if max(simlist)>threshold:
        #links1.append([simlist.index(max(simlist)),links1[simlist.index(max(simlist))]])
        return clstrs1[simlist.index(max(simlist))],links1[simlist.index(max(simlist))]
    else:
        return [],[]

def getclusterLinkedResults(episodes_clusters,rawData,groundTruth,threshold=0.4):
    cltr_data = []
    for i in episodes_clusters:
        cltr_data.append(ut.get_spcific_rawData(rawData, i))
    cltr_data = [i for sublist in cltr_data for i in sublist]
    clstrs = [i for sublist in episodes_clusters for i in sublist]

    new_links = []
    for i in range(len(cltr_data)):
        for j in range(i + 1, len(cltr_data)):
            scenes_sim, links = getclusterSim(cltr_data[i], cltr_data[j], clstrs[i], clstrs[j], threshold=threshold)
            if links!= [] :
                new_links.append([scenes_sim, links])

    linked_clusters = clstrs + new_links
    sum_mem=sum([len(i) for i in new_links])
    linkArrayComputed1 = ut.get_adjucencyMat(clstrs, rawData.shape[0])
    linkArrayComputed = ut.get_adjucencyMat(linked_clusters, rawData.shape[0])
    print(linkArrayComputed.shape,groundTruth.ref_link_substories.shape)
    rec_1, pre_1, f1_1, acc_1 = ev.linkingArrayRP(linkArrayComputed1, groundTruth.ref_link_stories, upperDiagonal=False)
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_link_stories, upperDiagonal=False)
    rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_link_substories,upperDiagonal=False)
    print(round(threshold,2),len(clstrs), len(new_links),sum_mem,rec_1, pre_1, f1_1, acc_1,rec,pre, f1, acc, rec_s, pre_s, f1_s, acc_s)
    return [round(threshold,2),len(clstrs), len(new_links),sum_mem,rec, pre, f1, rec_s, pre_s, f1_s]