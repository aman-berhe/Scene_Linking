"""
One format is creating a datframe for each results
"""
import Embeddings as emb
import pandas as pd
import numpy as np
import utility as ut
import clustering
import Evaluations as ev
import numpy as np
import pandas as pd
import load_features as lf
import itertools

def saveResults(fileName,data):
    results = pd.DataFrame(index = ['Word Embedding', 'TF-IDF','Speaking characters','Entities'],
          columns = ['Recall','Precision', 'F1 score', 'Recall','Precision', 'F1 score']
          )
    results.loc['Word Embedding'] = [rec,prec,f1,recSub,precSub,f1Sub]
    results.loc['TF-IDF'] = [rec,prec,f1,recSub,precSub,f1Sub]
    results.loc['Speaking characters'] = [rec, prec, f1, recSub, precSub, f1Sub]
    results.loc['Entities'] = [rec, prec, f1, recSub, precSub, f1Sub]
    results.to_csv(fileName)

def featureResults(data,normDF,fileName,ref_array_stories,ref_array_Substories,ref_first_n_link,algo='sp',latex=False):
    res = pd.DataFrame(columns=['clstr_num', 'Recall', 'Precision', 'F1_score', 'Acc', 'Recall_sub', 'Precision_sub', 'F1_score_sub', 'Acc_sub','Recall_n', 'Precision_n', 'F1_score_n', 'Acc_n'])
    j=0
    for i in range(2, 21):
        try:
            scenesRelated, linkArrayComputed = ut.cluster_postprecsess(algo, i, data, normDF, simThreshold=0.5)
            rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, ref_array_stories, upperDiagonal=True)
            rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP(linkArrayComputed, ref_array_Substories, upperDiagonal=True)
            rec_n, pre_n, f1_n, acc_n = ev.linkingArrayRP(linkArrayComputed, ref_first_n_link, upperDiagonal=True)
            results = [i, rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s, rec_n, pre_n, f1_n, acc_n]
            res.loc[j] = results
            j=j+1
            if latex:
                print("{} & {} & {} & {} & {} & {} & {} & {} & {}".format(i, rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s))
        except:
            break
    for i in range(2, 20):
        try:
            scenesRelated, linkArrayComputed = ut.cluster_postprecsess(algo, i, data, normDF, simThreshold=0.5)
            rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, ref_array_stories, upperDiagonal=False)
            rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP(linkArrayComputed, ref_array_Substories, upperDiagonal=False)
            rec_n, pre_n, f1_n, acc_n = ev.linkingArrayRP(linkArrayComputed, ref_first_n_link, upperDiagonal=False)
            results = [i, rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s,rec_n, pre_n, f1_n, acc_n]
            res.loc[j] = results
            j=j+1
            if latex:
                print("{} & {} & {} & {} & {} & {} & {} & {} & {}".format(i, rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s))
        except:
            break
    res.to_csv('Results/'+fileName+'.csv')
    return  res

def pairwiseFeatureResults(data,normDF,fileName,scenes_grouped_stories,scenes_grouped_substories,scenes_grouped_first_n_links,algo='sp',latex=False):
    res = pd.DataFrame(columns=['clstr_num', 'Recall', 'Precision', 'F1_score', 'Recall_sub', 'Precision_sub', 'F1_score_sub','Recall_n', 'Precision_n', 'F1_score_n'])
    j=0
    for i in range(2, 21):
        scenesRelated_hyp, linkArrayComputed = ut.cluster_postprecsess(algo, i, data, normDF, simThreshold=0.5)
        rec, pre, f1= ev.countpairRP(scenesRelated_hyp,scenes_grouped_stories)
        rec_s, pre_s, f1_s=ev.countpairRP(scenesRelated_hyp,scenes_grouped_substories)
        rec_n, pre_n, f1_n=ev.countpairRP(scenesRelated_hyp,scenes_grouped_first_n_links)
        results = [i, rec, pre, f1, rec_s, pre_s, f1_s, rec_n, pre_n, f1_n,]
        res.loc[j] = results
        j=j+1
        if latex:
            print("{} & {} & {} & {} & {} & {} & {} & {} & {}".format(i, rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s))
    res.to_csv('Results/pair_'+fileName+'.csv')
    return res

def getBestRows(feat,algos):
    df = pd.read_csv('Results/pair_' + feat + '_' + algos[0] + '.csv')
    col=df.columns.tolist()
    res = pd.DataFrame(columns=col)
    for i in algos:
        try:
            if i=='online':
                df = pd.read_csv('Results/paired_' + feat + '_' + i + '.csv')
            else:
                df = pd.read_csv('Results/pair_' + feat + '_' + i + '.csv')
            bestRow = df.iloc[df.F1_score.tolist().index(max(df.F1_score.tolist()))].tolist()
            # print(bestRow)
            res.loc[i] = bestRow
        except:
            print("clustering of {} based on {} not computed".format(feat, i))
    res.to_csv('Results/best_pair_'+feat+'.csv')
    return res

def getResults(simData,fileInden):
    for attr, value in simData.__dict__.items():
        print(attr)
        for alg in algos_full:
            filename = attr + '_fileInden_' + alg
            res=pairwiseFeatureResults(value, normDF, filename, groundTruth.scenes_grouped_stories,groundTruth.scenes_grouped_substories, groundTruth.scenes_grouped_first_n_links,algo=alg)


def granularity_avg_result(dataframe,algos,clstr_num,featName='sp_char'):
    res = pd.DataFrame(
        columns=['Recall', 'Precision', 'F1_score', 'Acc', 'Recall_sub', 'Precision_sub', 'F1_score_sub',
                 'Acc_sub', 'Recall_n', 'Precision_n', 'F1_score_n', 'Acc_n'])
    j = 0
    for algo in algos:
        res_all=[]
        for s in range(1,3):
            for e in range(1,11):
                try:
                    dfGran, start, end = ut.getGranularity(dataframe, season=s, episode=e)
                    rawData_gran = lf.Rawdata_Gran(dfGran, start, end)
                    simData_Gran = lf.Simdata(rawData_gran)
                    groundTruth_gran = lf.Groundtruth_gran(dfGran)
                    feat=simData_Gran.__getattribute__(featName)
                    groundTruth_gran.ref_link_stories,groundTruth_gran.ref_link_substorie,groundTruth_gran.ref_first_n_link
                    scenesRelated, linkArrayComputed = ut.cluster_postprecsess(algo, clstr_num, feat, dfGran, simThreshold=0.5)
                    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, groundTruth_gran.ref_link_stories, upperDiagonal=True)
                    rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP(linkArrayComputed, groundTruth_gran.ref_link_substorie, upperDiagonal=True)
                    rec_n, pre_n, f1_n, acc_n = ev.linkingArrayRP(linkArrayComputed, groundTruth_gran.ref_first_n_link, upperDiagonal=True)
                    results = [rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s,rec_n, pre_n, f1_n, acc_n]
                    res_all.append(results)
                    print("Season {} episode {}".format(s, e))
                except:
                    print("Season {} episode {} --> ERROR".format(s, e))
                    continue

        res.loc[algo]=list(np.mean(res_all, axis=0))
    res.to_csv('Results/average_episodes_' + featName + '.csv')

import onlineClustering as oc
def granularity_evg_online(dataframe,featName='sp_char'):
    res = pd.DataFrame(
        columns=['threshold','clstr_num','Recall', 'Precision', 'F1_score', 'Acc', 'Recall_sub', 'Precision_sub', 'F1_score_sub',
                 'Acc_sub', 'Recall_n', 'Precision_n', 'F1_score_n', 'Acc_n'])
    res_p = pd.DataFrame(
        columns=['threshold','clstr_num','Recall', 'Precision', 'F1_score', 'Recall_sub', 'Precision_sub', 'F1_score_sub', 'Recall_n', 'Precision_n', 'F1_score_n'])
    j = 0
    for i in range(10):
        res_all=[]
        res_all_p=[]
        for s in range(1, 3):
            for e in range(1, 11):
                try:
                    dfGran, start, end = ut.getGranularity(dataframe, season=s, episode=e)
                    rawData_gran = lf.Rawdata_Gran(dfGran, start, end)
                    simData_Gran = lf.Simdata(rawData_gran)
                    groundTruth_gran = lf.Groundtruth_gran(dfGran)
                    feat = simData_Gran.__getattribute__(featName)
                    clstrCenters, clstrFeatures, clstrSceneNum =oc.clusterOnline(feat, i*0.1)
                    clstrSceneNum.sort()
                    scenesRelated = list(clstrSceneNum for clstrSceneNum, _ in itertools.groupby(clstrSceneNum))
                    linkArrayComputed = ut.get_adjucencyMat(clstrSceneNum, feat.shape[0])
                    rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, groundTruth_gran.ref_link_stories, upperDiagonal=False)
                    rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP(linkArrayComputed, groundTruth_gran.ref_link_substories,upperDiagonal=False)
                    rec_n, pre_n, f1_n, acc_n = ev.linkingArrayRP(linkArrayComputed, groundTruth_gran.ref_first_n_link,upperDiagonal=False)
                    results = [rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s, rec_n, pre_n, f1_n, acc_n]
                    res_all.append(results)
                    rec_p, pre_p, f1_p = ev.countpairRP(scenesRelated, groundTruth_gran.scenes_grouped_stories)
                    rec_s_p, pre_s_p, f1_s_p= ev.countpairRP(scenesRelated, groundTruth_gran.scenes_grouped_substories)
                    rec_n_p, pre_n_p, f1_n_p = ev.countpairRP(scenesRelated, groundTruth_gran.scenes_grouped_first_n_links)
                    res_all_p.append([rec_p, pre_p, f1_p,rec_s_p, pre_s_p, f1_s_p,rec_n_p, pre_n_p, f1_n_p])
                    print("Season {} episode {}".format(s, e))
                except:
                    print("Season {} episode {} -->ERROR".format(s, e))
                    continue

        res.loc[i] = ([i*0.1]+list(np.mean(res_all, axis=0).round(3)))
        res_p.loc[i] = ([i*0.1]+list(np.mean(res_all_p, axis=0).round(3)))

    res.to_csv('Results/average_episodes_' + featName + '_online.csv')
    res_p.to_csv('Results/average_episodes_Pair_' + featName + '_online.csv')

def getcombinedFeatures(features,groundTruth):
    for i,feat in enumerate(features):
        clstrCenters, clstrFeatures, clstrSceneNum = oc.clusterOnline(feat, 0.4)
        clstrSceneNum.sort()
        scenesRelated = list(clstrSceneNum for clstrSceneNum, _ in itertools.groupby(clstrSceneNum))
        linkArrayComputed = ut.get_adjucencyMat(clstrSceneNum, feat.shape[0])
        rec, pre, f1, acc = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_link_stories, upperDiagonal=False)
        rec_s, pre_s, f1_s, acc_s = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_link_substories,
                                                      upperDiagonal=False)
        rec_n, pre_n, f1_n, acc_n = ev.linkingArrayRP(linkArrayComputed, groundTruth.ref_first_n_link,
                                                      upperDiagonal=False)
        results = [rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s, rec_n, pre_n, f1_n, acc_n]
        res_all.append(results)
        rec_p, pre_p, f1_p = ev.countpairRP(scenesRelated, groundTruth.scenes_grouped_stories)
        rec_s_p, pre_s_p, f1_s_p = ev.countpairRP(scenesRelated, groundTruth.scenes_grouped_substories)
        rec_n_p, pre_n_p, f1_n_p = ev.countpairRP(scenesRelated, groundTruth.scenes_grouped_first_n_links)
        res_all_p.append([rec_p, pre_p, f1_p, rec_s_p, pre_s_p, f1_s_p, rec_n_p, pre_n_p, f1_n_p])
        res.loc[i] = list(np.mean(res_all, axis=0).round(3))
        res_p.loc[i] = list(np.mean(res_all_p, axis=0).round(3))

        res.to_csv('Results/average_episodes_' + featName + '_online.csv')
        res_p.to_csv('Results/average_episodes_Pair_' + featName + 'online.csv')


def getResults_Mirinda(data,normDF,fileName,ref_stories,ref_Substories,algo='sp',latex=False):
    res = pd.DataFrame(columns=['clstr_num', 'Recall', 'Precision', 'F1_score', 'Acc', 'Recall_sub', 'Precision_sub', 'F1_score_sub','Acc_sub'])
    j=0
    ms = ev.MerindaScore()
    for i in range(2, 30):
        #try:
        cls = clustering.getClusters(algo, i, data=data, simThreshold=0.5)
        #print("{}: {}".format(i, ms.scoreSet(ref_stories, cls)))
        #scenesRelated, linkArrayComputed = ut.cluster_postprecsess(algo, i, data, normDF, simThreshold=0.5)
        #cls=list(cls)
        rec, pre, f1, acc = ms.scoreSet(ref_stories, cls)
        rec_s, pre_s, f1_s, acc_s = ms.scoreSet(ref_Substories, cls)
        results = [i, rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s]
        res.loc[j] = results
        j=j+1
        if latex:
            print("{} & {} & {} & {} & {} & {} & {} & {} & {}".format(i, rec, pre, f1, acc, rec_s, pre_s, f1_s, acc_s))
        #except:
        #   print("Error ",type(cls))
        #  break
    res.to_csv('Results/mirinda_'+fileName+'.csv')
    return res