"""
Aman
This file have many clustering algorithms. Both fuzzy and crisp clustering techniques.
The functions need an input data and cluster numbers.
"""

import scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering,KMeans,SpectralClustering,FeatureAgglomeration,DBSCAN
import onlineClustering as oc
import sys
import skfuzzy as fuzz
sys.path.insert(1,'fuzzycmeans/fuzzycmeans/')
import fuzzy_clustering as fc
import numpy as np
import utility as ut

"""
get clusters labels using different clustering algorithms:
Input data, algorithm and data. For online clustering it also requires similarity threshold 
"""
#['hc','ag','fa','km','sp','fcm','fcm2','fuzzy']
#algos_full=['agglomerative','kmeans','specteral','fcluster','fcm','skfuzzy','online']
def getClusters(algo,n_cltrs,data,simThreshold=0.5):
    if algo=='hc' or algo.lower()=='hierarchical':
        dend = hc.dendrogram(hc.linkage(data, method='ward'))
        cls=getHierarchClusterAssignments(data, dend)
        return cls
    if algo=='ag' or algo.lower()=='agglomerative':
        cluster = AgglomerativeClustering(n_clusters=n_cltrs, affinity='euclidean', linkage='ward')
        cls=cluster.fit_predict(data)
        return cls
    if algo=='fa' or algo=='feature agglomerative':
        cluster=FeatureAgglomeration(n_clusters=n_cltrs, affinity='euclidean', linkage='ward')
        cls=cluster.fit_predict(data)
        return cls
    if algo=='fcluster':
        cls = fcluster(hc.linkage(data, method='ward'),n_cltrs,criterion='maxclust')
        return cls
    if algo=='km' or algo=='kmeans':
        cluster=KMeans(n_clusters=n_cltrs,random_state=0).fit(data)
        cls=cluster.labels_
        return cls
    if algo=='sp' or algo=='specteral':
        cluster= SpectralClustering(n_clusters=n_cltrs,assign_labels="discretize",random_state=0).fit(data)
        cls=cluster.labels_
        return cls
    if algo=='fuzzycmeans' or algo=='fcm':
        avgValue,uniPredValues,membership=getAvgFCM(n_cltrs,data)
        return avgValue,uniPredValues,membership
    if algo=='skfuzzy' or algo=='skfcm' or algo=='fuzzy':
        avgValue,uniPredValues,scenemembership=getAvgSk_Fuzzy(n_cltrs,data)
        #grdScenes, arrayclst = ut.fuzzygrouping(scenemembership, dataframe)
        #grdScenes, linkArray = get_groupedSceneNum(grdScenes, referenceArray.shape[0])
        return avgValue,uniPredValues,scenemembership
    if algo=='online':
        clstrCenters,clstrFeatures,clstrSceneNum=oc.clusterOnline(data,simThreshold=0.5)
        return clstrCenters,clstrFeatures,clstrSceneNum
    else:
        print('Clustering algorithms not Specified')

"""
Fuzzy Clustering helping functions
"""

def getAvgFCM(n_clusters,data):
    fcm=fc.FCM(n_clusters=n_clusters)

    #fcm.set_logger(tostdout=True,level=logging.DEBUG)
    fcm.fit(data)

    pred=fcm.predict(data)
    prdList=[round(i,4) for sublist in pred for i in sublist]
    uniPredValues=list(set(prdList))
    avgValue=sum(uniPredValues)/len(uniPredValues)

    membership=[]
    for i in pred:
        mem=[]
        for j,v in enumerate(i):
            if round(v,4)>0.20005:
                mem.append(j)
        if len(mem)==0:
            membership.append(list(np.where(i==max(i))[0]))
        else:
            membership.append(mem)

    return avgValue,uniPredValues,membership


def getAvgSk_Fuzzy(n_clusters,data):
    """
        Fuzzy clustering using skfuzzy (sklearn fuzzy clustering Cmeans)
        consideres the average of the membership values of data elements to cluster them in multiple or one cluster.
    """
    cntrs,u,u0,d,jm,p,fpc=fuzz.cluster.cmeans(data,n_clusters,2,0.005,100,metric='euclidean',init=None,seed=None)
    prdList=[round(i,4) for sublist in u for i in sublist]
    uniPredValues=list(set(prdList))
    avgValue=sum(prdList)/len(prdList)

    membership=[]
    for i in range(u.shape[0]):
        mem=[]
        for j,v in enumerate(u[i]):
            if round(v,4)>avgValue:
                mem.append(j)
        if len(mem)==0:
            membership.append(list(np.where(u[i]==max(u[i]))[0]))
        else:
            membership.append(mem)
    scenemembership=[]
    for i in range(len(data)):
        tmpMemb=[]
        for c in range(len(membership)):
            if i in membership[c]:
                tmpMemb.append(c)
        scenemembership.append(tmpMemb)
    return avgValue,uniPredValues,scenemembership