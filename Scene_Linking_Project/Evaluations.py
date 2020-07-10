import pandas as pd
from scipy.misc import comb
import numpy as np


def countTruthValue(linkArrayRef,linkArrayComputed, upperDiagonal=False):
    """
    The input of this function is two linking arrays. The function is not symmetrical. This means the order of the arguments matters.
    Therefore the order of the input is
    1. LinkArrayRef : The ground truth linking array
    2. LinkArrayCls : The computed linking array
    """
    TP=0
    TN=0
    FP=0
    FN=0
    if upperDiagonal==False:
        for i in range(len(linkArrayRef)):
            for j in range(len(linkArrayRef)):
                if linkArrayRef[i][j]==linkArrayComputed[i][j]==1:
                    TP=TP+1
                elif linkArrayRef[i][j]==linkArrayComputed[i][j]==0:
                    TN=TN+1
                elif linkArrayRef[i][j]==1 and linkArrayComputed[i][j]==0:
                    FP=FP+1
                elif linkArrayRef[i][j]==0 and linkArrayComputed[i][j]==1:
                    FN=FN+1
    else:
        #We only take the upper diagonall of the linking arrays. Since they are symmetric
        for i in range(len(linkArrayRef)-1):
            for j in range(i+1,len(linkArrayRef)):
                if linkArrayRef[i][j]==linkArrayComputed[i][j]==1:
                    TP=TP+1
                elif linkArrayRef[i][j]==linkArrayComputed[i][j]==0:
                    TN=TN+1
                elif linkArrayRef[i][j]==1 and linkArrayComputed[i][j]==0:
                    FP=FP+1
                elif linkArrayRef[i][j]==0 and linkArrayComputed[i][j]==1:
                    FN=FN+1
        #TP=TP-len(linkArrayRef)#Removing the diagonal which is always Trueon the computed diagonals
    return TP,TN,FP,FN

def linkingArrayRP(linkArrayRef,linkArrayComputed,upperDiagonal=False):
    """
    The input of this function is two linking arrays. The function make use of countTruthValue().
    The function is not symmetrical which means the order of the arguments matters:
    """
    TP,TN,FP,FN=countTruthValue(linkArrayRef,linkArrayComputed,upperDiagonal)
    try:
        prec=TP/(TP+FP)
        recall=TP/(TP+FN)
        f1=2*(recall*prec/(recall+prec))
        acc=(TP+TN)/(TP+TN+FP+FN)
        return round(recall,3),round(prec,3),round(f1,3),round(acc,3)
    except ZeroDivisionError:
        return 0

"""
Computing pairwise correctness of computed linked scenes.
Count the number correctly paired, incorrectly paired and missed pairs of scenes in a group fo scene (Linking Clusters)
"""

def intersection(sceneGrouping1, scenesGrouping2):
    """
    intesection of two grouping of scene. The first grouping is the Ground truth and the second one is computed grouping
    """
    lst3 = [value for value in sceneGrouping1 if value in scenesGrouping2]
    return lst3

def computeMissedLinks(scenesGrouped, intersect):
    """
    Compute missing links: Missed pairs from the ground truth grouping
    """
    paired=[]
    for sg in scenesGrouped:
        tmpMax=0
        for i in range(len(intersect)):
            if tmpMax<len(intersection(sg,intersect[i])):
                tmpMax=len(intersection(sg,intersect[i]))
        paired.append(tmpMax)
    missedLinks=[len(scenesGrouped[p])-paired[p]-1 if paired[p]>1 else len(scenesGrouped[p])-paired[p] for p in range(len(paired))]

    return missedLinks


def countStastics(scenesRelated,scenesGrouped):
    """
    The function return the total number of correctly paired, incorrectly paired and missed pairs. using the above counting functions.
    For counting correct pairs, incorrectPairs and missing pairs. The function does not consider the current scene in the counting.
    It only counts pairs which are paired with the current scenes.
    """
    intesect=[]
    for i in range(len(scenesRelated)):#computed group
        tmp=[]
        for sg in scenesGrouped:
            if len(tmp)<len(intersection(scenesRelated[i],sg)):
                tmp=intersection(scenesRelated[i],sg)
        intesect.append(tmp)
    correctPairs=[len(p)-1 if len(p)>1 else len(p) for p in intesect]
    incorrectPairs=[len(scenesRelated[p])-len(intesect[p])-1 if len(scenesRelated[p])>len(intesect[p]) else len(scenesRelated[p])-len(intesect[p]) for p in range(len(intesect))]
    missedLinks=computeMissedLinks(scenesGrouped,intesect)
    #return intesect,correctPairs,incorrectPairs,missedLinks
    return intesect,sum(correctPairs),sum(incorrectPairs),sum(missedLinks)

def countpairRP(scenesRelated_hyp,scenesGrouped_gt):
    intesect,correctPairs,incorrectPairs,missedPairs=countStastics(scenesRelated_hyp,scenesGrouped_gt)
    recall=correctPairs/(correctPairs+incorrectPairs)
    precision=correctPairs/(correctPairs+missedPairs)
    f1=2*(recall*precision/(recall+precision))
    return round(recall,3),round(precision,3),round(recall,3)

class MerindaScore:
    def myComb(self,a, b):
        return comb(a, b, exact=True)

    def get_cooccurrence_matrix(selg,true_labels, pred_labels):
        assert len(true_labels) == len(pred_labels)
        true_label_map = {}
        i = 0
        for l in true_labels:
            if not l in true_label_map:
                true_label_map[l] = i
                i += 1
        hyp_label_map = {}
        i = 0
        for l in pred_labels:
            if not l in hyp_label_map:
                hyp_label_map[l] = i
                i += 1
        m = [[0 for i in range(len(hyp_label_map))] for j in range(len(true_label_map))]
        for i in range(len(true_labels)):
            m[true_label_map[true_labels[i]]][hyp_label_map[pred_labels[i]]] += 1
        return (np.array(m), true_label_map, hyp_label_map)

    def get_tp_fp_tn_fn(self,cooccurrence_matrix):
        vComb = np.vectorize(self.myComb)
        tp_plus_fp = vComb(cooccurrence_matrix.sum(0, dtype=int), 2).sum()
        tp_plus_fn = vComb(cooccurrence_matrix.sum(1, dtype=int), 2).sum()
        tp = vComb(cooccurrence_matrix.astype(int), 2).sum()
        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp
        tn = comb(cooccurrence_matrix.sum(), 2) - tp - fp - fn

        return [tp, fp, tn, fn]

    def scoreSet(self,true_labels, pred_labels):
        cooccurrence_matrix, true_label_map, pred_label_map = self.get_cooccurrence_matrix(true_labels, pred_labels)
        tp, fp, tn, fn = self.get_tp_fp_tn_fn (cooccurrence_matrix)

        acc = 1. * (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        p = 1. * tp / (tp + fp) if tp + fp > 0 else 0
        r = 1. * tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2. * p * r / (p + r) if p + r > 0 else 0
        return r, p, f1, acc