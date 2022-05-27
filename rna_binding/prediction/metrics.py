"""
Metrics used for model evaluation

Author: Lukas Moeller
Date: 01/2022
"""



import math
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics.pairwise import euclidean_distances



FAIL_DIST = math.sqrt(3) * 80 # maximal distance within cube with edge length 80
FAIL_SET = 0



def get_single_site_centers(cube, threshold, quantile=0.5):
    # get predicted binding site coordinates 
    coords = np.array(np.asarray(cube > threshold).nonzero()).transpose()

    if coords.shape[0] == 0:
        labels, cluster_centers, cluster_num = None, None, 0
        return coords, labels, cluster_centers, cluster_num

    else:
        # estimate bandwith parameter for mean-shift clustering
        bandwidth = estimate_bandwidth(coords, quantile)

        if bandwidth is not None and bandwidth > 0:
            # perform mean-shift clustering
            # note that orphan points (not within any kernel) will receive label -1
            ms = MeanShift(bandwidth=bandwidth).fit(coords)
            
            # get labels for all clusters
            labels = ms.labels_

            # get indices of binding site coordinates without orphan points & update coords, labels
            not_orphan = np.asarray(labels != -1).nonzero()[0]
            updated_coords = coords[not_orphan]
            updated_labels = labels[not_orphan]
            
            # get center of clusters
            cluster_centers = ms.cluster_centers_

            # get number of clusters
            cluster_num = len(np.unique(updated_labels))

            # return results
            return updated_coords, updated_labels, cluster_centers, cluster_num
        
        else:
            labels, cluster_centers, cluster_num = None, None, 0
            return coords, labels, cluster_centers, cluster_num


def jaccard_coefficient(prediction, target, threshold):
    prediction = np.where(prediction > threshold, 1, 0)
    intersection = np.sum(np.logical_and(prediction, target))
    union = np.sum(np.logical_or(prediction, target))
    return intersection/union


def overlap_coefficient(prediction, target, threshold):
    prediction = np.where(prediction > threshold, 1, 0)
    intersection = np.sum(np.logical_and(prediction, target))
    prediction, target = np.sum(prediction), np.sum(target)
    return intersection/np.min([prediction, target])


def sorensen_dice(prediction, target, threshold):
    prediction = np.where(prediction > threshold, 1, 0)
    intersection = np.sum(np.logical_and(prediction, target))
    prediction, target = np.sum(prediction), np.sum(target)
    return 2 * intersection/(prediction + target)


def dcc(prediction, pcoord, plabel, tcenter, pcenter, tnum, pnum):
    # get top tnum predicted sites and calculate distance to closest target site
    if pnum < tnum: # evaluate DCC for all predicted binding sites
        likelihood_list = np.zeros(pnum)
        center_list = np.zeros((pnum, 3))
    else: # evaluate DCC for top "tnum" predicted binding sites
        likelihood_list = np.zeros(tnum)
        center_list = np.zeros((tnum, 3))
    
    for p in np.unique(plabel): # select top "tnum" (or if "pnum" < "tnum": top "pnum") sites
        index_pred = plabel == p
        tmp_udp = pcoord[index_pred].transpose()
        likelihood = np.mean(prediction[tmp_udp[0], tmp_udp[1], tmp_udp[2]]) # highest average prediction
        addition_indices = np.argwhere(likelihood_list < likelihood)
        if len(addition_indices) > 0:
            likelihood_list[addition_indices[0]] = likelihood
            center_list[addition_indices[0]] = pcenter[p]
    
    distance_matrix = euclidean_distances(tcenter, center_list)
    min_distance_2_target = np.min(distance_matrix, axis=0)
    return min_distance_2_target


def calculate_metrics(target, prediction, tcoord, pcoord, tlabel, plabel, tcenter, pcenter, tnum, pnum, threshold):
    """
    script to calculate metrics after making predictions
    """

    # case 1: no target available after clustering or no binding site predicted
    if tnum == 0 or pnum == 0:
        # define results dict
        results = {
            'jaccard': FAIL_SET,
            'overlap': FAIL_SET,
            'sorensen': FAIL_SET,
            'cc_distance': np.array([FAIL_DIST]),
            'cc_distance_mean': FAIL_DIST,
            'tnum': tnum,
            'pnum': pnum
        }
        return results
    
    # case 2: calculate metrics
    else:
        min_distance_2_target = dcc(prediction, pcoord, plabel, tcenter, pcenter, tnum, pnum)

        # define results dict
        results = {
            'jaccard': jaccard_coefficient(prediction, target, threshold),
            'overlap': overlap_coefficient(prediction, target, threshold),
            'sorensen': sorensen_dice(prediction, target, threshold),
            'cc_distance': min_distance_2_target,
            'cc_distance_mean': np.mean(min_distance_2_target),
            'tnum': tnum,
            'pnum': pnum
        }
        return results
