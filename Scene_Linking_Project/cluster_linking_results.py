import pandas as pd
import numpy as np
import utility as ut
import Embeddings as emb
import Save_Results as sr
import load_features as lf
import onlineClustering as oc
import create_links as cl
import itertools
import Evaluations as ev

"""
Load data
"""
dataframe=pd.read_csv('/home/berhe/Desktop/Scene_Linking_Project/Data/Scene_Dataset_Normalized_locations.csv')
rawData=lf.Rawdata(dataframe)
simData=lf.Simdata(rawData)
groundTruth=lf.Groundtruth(dataframe)

"""
compute episode clusters
"""
def clusterEpisodePair(dataframe,season1=1,season2=1,episode1=1,episode2=2):
    clsCenters = []
    comp_cluster = []
    episodes_length = [0]
    gt_story = []
    gt_links = []
    thr = [0.6, 0.5,0.6, 0.5,0.6, 0.5,0.5,0.6, 0.5,0.5,0.6, 0.5]
    for s in range(season1, season2+1):
        for e in range(episode1, episode2+1):
            dfGran, start, end = ut.getGranularity(dataframe, season=s, episode=e)
            #print(len(dfGran))
            rawData_gran = lf.Rawdata_Gran(dfGran, start, end)
            simData_Gran = lf.Simdata(rawData_gran)
            groundTruth_gran = lf.Groundtruth_gran(dfGran)
            clstrCenters, clstrFeatures, clstrSceneNum, leng = cl.getepisodeOnlineClusters(dataframe, s, e, 'sp_char',
                                                                                                thr[e - 1])
            linkArrayComputed = ut.get_adjucencyMat(clstrSceneNum, dfGran.shape[0])
            # rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, groundTruth_gran.ref_link_stories, upperDiagonal=False)
            # recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, groundTruth_gran.ref_link_stories, upperDiagonal=True)
            # print(rec,pre,f1,recT, preT, f1T)
            episodes_length.append(leng)
            unique_links, linkedArray_rec = ut.get_recursiveLinks_unique(dfGran)
            gt_links.append([[i + episodes_length[e - 1] for i in sublist] for sublist in unique_links])
            storiesGrouped, substoriesGrouped, stories, substories = ut.storywise_group(dfGran)
            gt_story.append([[i + episodes_length[e - 1] for i in sublist] for sublist in storiesGrouped])
            comp_cluster.append([[i + episodes_length[e - 1] for i in sublist] for sublist in clstrSceneNum])
            clsCenters.append(clstrCenters)
            linkArrayRecursive = ut.get_adjucencyMat(unique_links, episodes_length[e])
            linkArrayStory = ut.get_adjucencyMat(storiesGrouped, episodes_length[e])
            rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=False)
            recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=True)
            print(" {} & {} & {} & {} & {} & {} \\\\".format(rec, pre, f1,recT, preT, f1T))
            rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=False)
            recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=True)
            print(" {} & {} & {} & {} & {} & {} \\\\".format(rec, pre, f1,recT, preT, f1T))
    
    return episodes_length,comp_cluster,gt_links,gt_story

def get_cluster_data_all(comp_cluster):
    epsds_cltr_data=[]
    epsds_cltrs=[]
    for i in comp_cluster:
        tmp_data=[]
        for j in i:
            tmp_data.append(ut.get_spcific_rawData(rawData.entities, i))
        epsds_cltr_data.append(tmp_data)
        epsds_cltrs.append(i)
    return epsds_cltr_data,epsds_cltrs

def get_cluster_data(comp_cluster):
    ep1_cltr_data = []
    for i in comp_cluster[0]:
        ep1_cltr_data.append(ut.get_spcific_rawData(rawData.entities, i))
    ep1_clstrs = comp_cluster[0]
    ep2_cltr_data = []
    for i in comp_cluster[1]:
        ep2_cltr_data.append(ut.get_spcific_rawData(rawData.entities, i))
    ep2_clstrs = comp_cluster[1]
    return ep1_cltr_data,ep2_cltr_data,ep1_clstrs,ep2_clstrs

def get_episode_cluster_links_all(comp_cluster):
    epsds_cltr_data,epsds_cltrs=get_cluster_data_all(comp_cluster)
    comp_cluster_pair_links = []
    for i in range(len(epsds_cltr_data)-1):
        for c1 in range(len(epsds_cltr_data[i])):
            for c2 in range(len(epsds_cltr_data[i+1])):
            #print(ep1_clstrs[c1],ep2_cltr_data[c2])
                scene_pair_link = cl.create_cluster_link(epsds_cltr_data[i][c1], epsds_cltr_data[i+1][c2], threshold=0.2)
                if scene_pair_link:
                    comp_cluster_pair_links.append(epsds_cltrs[i][c1] + epsds_cltrs[i+1][c1])

    return comp_cluster_pair_links

def get_episode_cluster_links(comp_cluster):
    ep1_cltr_data, ep2_cltr_data,ep1_clstrs,ep2_clstrs=get_cluster_data(comp_cluster)
    comp_cluster_pair_links = []
    for c1 in range(len(ep1_cltr_data)):
        for c2 in range(len(ep2_cltr_data)):
            #print(ep1_clstrs[c1],ep2_cltr_data[c2])
            scene_pair_link = cl.create_cluster_link(ep1_cltr_data[c1], ep2_cltr_data[c2], threshold=0.7)
            if scene_pair_link:
                comp_cluster_pair_links.append(ep1_clstrs[c1] + ep2_clstrs[c2])

    return comp_cluster_pair_links

def compute_linking_results(dataframe,s1=1,s2=1,e1=1,e2=2):
    #for e in range(e1,e2):
    episodes_length, comp_cluster, gt_links, gt_story=clusterEpisodePair(dataframe,season1=s1,season2=s2,episode1=e1,episode2=e2)
    #print("Comuted clusters \n",comp_cluster[0])
    #print("Comuted clusters \n", comp_cluster[1])
    #print("*"*40)
    print(len(comp_cluster))
    #for i in range(len(comp_cluster)):
    comp_cluster_pair_links=get_episode_cluster_links(comp_cluster)
    gt_links_mer=get_episode_cluster_links(gt_links)
    gt_story_mer = get_episode_cluster_links(gt_story)
    print("computed cluster pairs:",comp_cluster_pair_links)

    gt_links = [item for sublist in gt_links for item in sublist]
    gt_story = [item for sublist in gt_story for item in sublist]
    comp_cluster = [item for sublist in comp_cluster for item in sublist]

    unique_links, linkedArray_rec = ut.get_recursiveLinks_unique(dataframe)
    storiesGrouped, substoriesGrouped, stories, substories = ut.storywise_group(dataframe)
    #clstrCenters, clstrFeatures, clstrSceneNum = oc.clusterOnline(rawData.sp_char, 0.6)

    clear_pair_rec, link_bn_eps_rec = ut.getpairedlinks_2_episodes(dataframe, unique_links, season=s1, episode=e1)
    clear_pair_story, link_bn_eps_story = ut.getpairedlinks_2_episodes(dataframe, storiesGrouped, season=s1, episode=e1)
    clear_pair_comp, link_bn_eps = ut.getpairedlinks_2_episodes(dataframe, comp_cluster_pair_links, season=s1, episode=e1)
    print("clear pairs")
    print(clear_pair_comp)
    print(clear_pair_rec)
    print(clear_pair_story)
    episodes_rec_links = gt_links + clear_pair_rec
    story_groups_link = gt_story + clear_pair_story
    comp_cluster_links = comp_cluster + clear_pair_comp

    linkArrayComputed = ut.get_adjucencyMat(comp_cluster_links, episodes_length[1] + episodes_length[2])
    linkArrayRecursive = ut.get_adjucencyMat(episodes_rec_links, episodes_length[1] + episodes_length[2])
    linkArrayStory = ut.get_adjucencyMat(story_groups_link, episodes_length[1] + episodes_length[2])
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=False)
    recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=True)
    print(" {} & {} & {} & {} & {} & {} \\\\".format(rec, pre, f1, recT, preT, f1T))
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=False)
    recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=True)
    print(" {} & {} & {} & {} & {} & {} \\\\".format(rec, pre, f1, recT, preT, f1T))

    #print("Only the link pairs")
    linkArrayComputed = ut.get_adjucencyMat(clear_pair_comp, episodes_length[1] + episodes_length[2])
    linkArrayRecursive = ut.get_adjucencyMat(clear_pair_rec, episodes_length[1] + episodes_length[2])
    linkArrayStory = ut.get_adjucencyMat(clear_pair_story, episodes_length[1] + episodes_length[2])
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=False)
    recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=True)
    #print(rec, pre, f1, recT, preT, f1T)
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=False)
    recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=True)
    #print(rec, pre, f1, recT, preT, f1T)

    #print("Merged clusters")
    linkArrayComputed = ut.get_adjucencyMat(comp_cluster_pair_links, episodes_length[1] + episodes_length[2])
    linkArrayRecursive = ut.get_adjucencyMat(gt_links_mer, episodes_length[1] + episodes_length[2])
    linkArrayStory = ut.get_adjucencyMat(gt_story_mer, episodes_length[1] + episodes_length[2])
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=False)
    recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=True)
    #print(rec, pre, f1, recT, preT, f1T)
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=False)
    recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=True)
    #print(rec, pre, f1, recT, preT, f1T)

def compute_linking_results_all(dataframe,s1=1,s2=1,e1=1,e2=2):
    episodes_length, comp_cluster, gt_links, gt_story = clusterEpisodePair(dataframe, season1=s1, season2=s2,
                                                                           episode1=e1, episode2=e2)
    print("Comuted clusters \n", comp_cluster[0])
    print("Comuted clusters \n", comp_cluster[1])
    print("*" * 40)
    comp_cluster_pair_links = get_episode_cluster_links_all(comp_cluster)
    gt_links_mer = get_episode_cluster_links_all(gt_links)
    gt_story_mer = get_episode_cluster_links_all(gt_story)
    print("computed cluster pairs:", comp_cluster_pair_links)

    gt_links = [item for sublist in gt_links for item in sublist]
    gt_story = [item for sublist in gt_story for item in sublist]
    comp_cluster = [item for sublist in comp_cluster for item in sublist]

    unique_links, linkedArray_rec = ut.get_recursiveLinks_unique(dataframe)
    storiesGrouped, substoriesGrouped, stories, substories = ut.storywise_group(dataframe)
    # clstrCenters, clstrFeatures, clstrSceneNum = oc.clusterOnline(rawData.sp_char, 0.6)

    clear_pair_rec, link_bn_eps_rec = ut.getpairedlinks_2_episodes(dataframe, unique_links, season=s1, episode=e1)
    clear_pair_story, link_bn_eps_story = ut.getpairedlinks_2_episodes(dataframe, storiesGrouped, season=s1, episode=e1)
    clear_pair_comp, link_bn_eps = ut.getpairedlinks_2_episodes(dataframe, comp_cluster_pair_links, season=s1,
                                                                episode=e1)
    episodes_rec_links = gt_links + clear_pair_rec
    story_groups_link = gt_story + clear_pair_story
    comp_cluster_links = comp_cluster + clear_pair_comp

    linkArrayComputed = ut.get_adjucencyMat(comp_cluster_links, episodes_length[1] + episodes_length[2])
    linkArrayRecursive = ut.get_adjucencyMat(episodes_rec_links, episodes_length[1] + episodes_length[2])
    linkArrayStory = ut.get_adjucencyMat(story_groups_link, episodes_length[1] + episodes_length[2])
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=False)
    recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayRecursive, upperDiagonal=True)
    print(rec, pre, f1, recT, preT, f1T)
    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=False)
    recT, preT, f1T, acc = ev.linkingArrayRP(linkArrayComputed, linkArrayStory, upperDiagonal=True)
    print(rec, pre, f1, recT, preT, f1T)