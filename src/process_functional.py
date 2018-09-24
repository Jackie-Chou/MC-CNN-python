"""
    processing functions used in stereo matching
"""
import os
import util
import time
import cv2
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
from model import NET
from tqdm import tqdm

def compute_features(left_image, right_image, patch_height, patch_width, checkpoint):

    height, width = left_image.shape[:2]

    # pad images to make the final feature map size = (height, width..)
    auged_left_image = np.zeros([1, height+patch_height-1, width+patch_width-1, 1], dtype=np.float32)
    auged_right_image = np.zeros([1, height+patch_height-1, width+patch_width-1, 1], dtype=np.float32)
    row_start = (patch_height - 1)/2
    col_start = (patch_width - 1)/2
    auged_left_image[0, row_start: row_start+height, col_start: col_start+width] = left_image
    auged_right_image[0, row_start: row_start+height, col_start: col_start+width] = right_image

    # TF placeholder for graph input
    x = tf.placeholder(tf.float32, shape=[1, height+patch_height-1, width+patch_width-1, 1])  

    # Initialize model
    model = NET(x, input_patch_size = patch_height, batch_size=1)
    saver = tf.train.Saver(max_to_keep=10)

    features = model.features

    # compute features on both images
    with tf.Session(config=tf.ConfigProto(
                        log_device_placement=False, \
                        allow_soft_placement=True, \
                        gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        print "{}: restoring from {}...".format(datetime.now(), checkpoint)
        saver.restore(sess, checkpoint)

        print "{}: features computing...".format(datetime.now())
        '''
        # this is used when a whole image is too big to fit in the memory
        featureslul = sess.run(features, feed_dict = {x: auged_left_image[:, 0: height/2+patch_height-1, 0: width/2+patch_width-1]}) 
        featureslur = sess.run(features, feed_dict = {x: auged_left_image[:, 0: height/2+patch_height-1, width/2: width+patch_width-1]}) 
        featureslbl = sess.run(features, feed_dict = {x: auged_left_image[:, height/2: height+patch_height-1, 0: width/2+patch_width-1]}) 
        featureslbr = sess.run(features, feed_dict = {x: auged_left_image[:, height/2: height+patch_height-1, width/2: width+patch_width-1]}) 

        featuresrul = sess.run(features, feed_dict = {x: auged_right_image[:, 0: height/2+patch_height-1, 0: width/2+patch_width-1]}) 
        featuresrur = sess.run(features, feed_dict = {x: auged_right_image[:, 0: height/2+patch_height-1, width/2: width+patch_width-1]}) 
        featuresrbl = sess.run(features, feed_dict = {x: auged_right_image[:, height/2: height+patch_height-1, 0: width/2+patch_width-1]}) 
        featuresrbr = sess.run(features, feed_dict = {x: auged_right_image[:, height/2: height+patch_height-1, width/2: width+patch_width-1]}) 

        featuresl = np.concatenate((np.concatenate((featureslul, featureslur), axis=2), np.concatenate((featureslbl, featureslbr), axis=2)), axis=1)
        featuresr = np.concatenate((np.concatenate((featuresrul, featuresrur), axis=2), np.concatenate((featuresrbl, featuresrbr), axis=2)), axis=1)
        '''

        featuresl = sess.run(features, feed_dict = {x: auged_left_image}) 
        featuresr = sess.run(features, feed_dict = {x: auged_right_image}) 
        print featuresl.shape

        featuresl = np.squeeze(featuresl, axis=0)
        featuresr = np.squeeze(featuresr, axis=0) # (height, width, 64)
        print "{}: features computed done...".format(datetime.now())

    # clear the used gpu memory
    tf.reset_default_graph()

    return featuresl, featuresr

# form cost volume
# max possible disparity is ndisp
# cost_volume[d, x, y] = -correlation between pixel (x, y) in left image and pixel (x, y - d) in right image
def compute_cost_volume(featuresl, featuresr, ndisp):

    print "{}: computing cost_volume for left image...".format(datetime.now())
    height, width = featuresl.shape[:2]
    left_cost_volume = np.zeros([ndisp, height, width], dtype=np.float32)

    # NOTE: since y - d may < 0, so some pixels may not have corresponding pixels
    tem_xl = featuresl
    tem_xr = featuresr
    for d in range(ndisp):
        print "{}: disparity {}...".format(datetime.now(), d)
        left_cost_volume[d, :, d:] = np.sum(np.multiply(tem_xl, tem_xr), axis=-1)
        tem_xl = tem_xl[:, 1:]
        tem_xr = tem_xr[:, :tem_xr.shape[1]-1]

    # use average cost to fill in those not calculated
    for d in range(ndisp-1, 0, -1):
        left_cost_volume[d:ndisp, :, d-1] = np.mean(left_cost_volume[d:ndisp, :, d:d+3], axis=-1)

    print "{}: cost_volume for left image computed...".format(datetime.now())

    # do it for right image again
    # NOTE: just copy from left_cost_volume since dot product is symmetric
    print "{}: computing cost_volume for right image...".format(datetime.now())
    right_cost_volume = np.zeros([ndisp, height, width], dtype=np.float32)
    for d in range(ndisp):
        right_cost_volume[d, :, :width-d] = left_cost_volume[d, :, d:]
    for d in range(ndisp-1, 0, -1):
        right_cost_volume[d:ndisp, :, width-d] = np.mean(right_cost_volume[d:ndisp, :, width-d-3:width-d], axis=-1)
    print "{}: cost_volume for right image computed...".format(datetime.now())

    # convert from matching score to cost
    # match score larger = cost smaller
    left_cost_volume =  -1. * left_cost_volume
    right_cost_volume = -1. * right_cost_volume
    return left_cost_volume, right_cost_volume

# cost volume aggregation
# use cross-based cost aggregation
def cost_volume_aggregation(left_image, right_image, left_cost_volume, right_cost_volume, intensity_threshold, distance_threshold, max_average_time):

    ndisp, height, width = left_cost_volume.shape
    left_union_region, left_union_region_num = compute_cross_region(left_image, intensity_threshold, distance_threshold)
    right_union_region, right_union_region_num = compute_cross_region(right_image, intensity_threshold, distance_threshold)
    """
    # NOTE: this is too large and is impractical to run :)
    # then compute disparity-dependent union regions considering support regions of both images
    print "{}: cost volume aggragation for left image...".format(datetime.now())
    max_num = (2*distance_threshold)**2
    dis_left_union_region = np.ndarray([ndisp, height, width, max_num, 2], dtype=np.int32)
    dis_left_union_region_num = np.ndarray([ndisp, height, width], dtype=np.int32)
    # compute for all disparities
    for d in range(ndisp):
        dis_left_union_region[d], dis_left_union_region_num[d] = \
                                        compute_disparity_union_region(left_union_region, left_union_region_num, \
                                                                      right_union_region, right_union_region_num, d, "L")
    # do the same for right
    print "{}: cost volume aggragation for right image...".format(datetime.now())
    max_num = (2*distance_threshold)**2
    dis_right_union_region = np.ndarray([ndisp, height, width, max_num, 2], dtype=np.int32)
    dis_right_union_region_num = np.ndarray([ndisp, height, width], dtype=np.int32)
    # compute for all disparities
    for d in range(ndisp):
        dis_right_union_region[d], dis_right_union_region_num[d] = \
                                        compute_disparity_union_region(left_union_region, left_union_region_num, \
                                                                       right_union_region, right_union_region_num, d, "R")
    """

    # then compute average match cost using union regions
    # NOTE: the averaging can be done several times
    print "{}: cost averaging for left cost_volume...".format(datetime.now())
    for _ in range(max_average_time):
        print "\t{}: averaging No.{} time".format(datetime.now(), _)

        agg_cost_volume = np.ndarray(left_cost_volume.shape, dtype=np.float32)
        for h in range(height):
            for w in range(width):
                aver_num = left_union_region_num[h, w]
                aver_regions = left_union_region[h, w, :aver_num]
                cost_sum = np.zeros([ndisp], dtype=np.float32)
                for v in range(aver_num):
                    h_, w_ = aver_regions[v]
                    cost_sum += left_cost_volume[:, h_, w_]
                agg_cost_volume[:, h, w] = cost_sum / aver_num

        left_cost_volume = agg_cost_volume

    print "{}: cost averaging for right cost_volume...".format(datetime.now())
    for _ in range(max_average_time):
        print "\t{}: averaging No.{} time".format(datetime.now(), _)

        agg_cost_volume = np.ndarray(right_cost_volume.shape, dtype=np.float32)
        for h in range(height):
            for w in range(width):
                aver_num = right_union_region_num[h, w]
                aver_regions = right_union_region[h, w, :aver_num]
                cost_sum = np.zeros([ndisp], dtype=np.float32)
                for v in range(aver_num):
                    h_, w_ = aver_regions[v]
                    cost_sum += right_cost_volume[:, h_, w_]
                agg_cost_volume[:, h, w] = cost_sum / aver_num

        right_cost_volume = agg_cost_volume

    print "{}: cost average done...".format(datetime.now())
    return left_cost_volume, right_cost_volume

# semi-global matching for four directions and taking average
# NOTE: after SGM, doing cost aggregation again
def SGM_average(left_cost_volume, right_cost_volume, left_image, right_image, \
                sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, sgm_V):

    # along four directions do dynamic programming and take average
    print "{}: semi-global matching for left image...".format(datetime.now())
    # right
    print "{}: right".format(datetime.now())
    r = (0, 1)
    left_cost_volume_right = semi_global_matching(left_image, right_image, left_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "L")
    # left
    print "{}: left".format(datetime.now())
    r = (0, -1)
    left_cost_volume_left = semi_global_matching(left_image, right_image, left_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "L")
    # for two vertical directions, P1 should be further devided by sgm_V
    # up
    print "{}: up".format(datetime.now())
    r = (-1, 0)
    left_cost_volume_up = semi_global_matching(left_image, right_image, left_cost_volume, r, sgm_P1/sgm_V, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "L")
    # bottom
    print "{}: bottom".format(datetime.now())
    r = (1, 0)
    left_cost_volume_bottom = semi_global_matching(left_image, right_image, left_cost_volume, r, sgm_P1/sgm_V, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "L")
    # taken average
    left_cost_volume = (left_cost_volume_right + left_cost_volume_left + left_cost_volume_up + left_cost_volume_bottom) / 4.

    # doing the same for right cost volume
    print "{}: semi-global matching for right image...".format(datetime.now())
    # right
    print "{}: right".format(datetime.now())
    r = (0, 1)
    right_cost_volume_right = semi_global_matching(left_image, right_image, right_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "R")
    # left
    print "{}: left".format(datetime.now())
    r = (0, -1)
    right_cost_volume_left = semi_global_matching(left_image, right_image, right_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "R")
    # for two vertical directions, P1 should be further devided by sgm_V
    # up
    print "{}: up".format(datetime.now())
    r = (-1, 0)
    right_cost_volume_up = semi_global_matching(left_image, right_image, right_cost_volume, r, sgm_P1/sgm_V, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "R")
    # bottom
    print "{}: bottom".format(datetime.now())
    r = (1, 0)
    right_cost_volume_bottom = semi_global_matching(left_image, right_image, right_cost_volume, r, sgm_P1/sgm_V, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "R")
    # taken average
    right_cost_volume = (right_cost_volume_right + right_cost_volume_left + right_cost_volume_up + right_cost_volume_bottom) / 4.
    print "{}: semi-global matching done...".format(datetime.now())

    return left_cost_volume, right_cost_volume

# disparity prediction
# simple "Winner-take-All"
def disparity_prediction(left_cost_volume, right_cost_volume):

    print "{}: left disparity map making...".format(datetime.now())
    ndisp, height, width = left_cost_volume.shape
    left_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            min_cost = float("inf")
            min_disparity = -1
            for d in range(ndisp):
                if left_cost_volume[d, h, w] < min_cost:
                    min_cost = left_cost_volume[d, h, w]
                    min_disparity = d
            assert min_disparity >= 0
            left_disparity_map[h, w] = min_disparity

    # same for right 
    print "{}: right disparity map making...".format(datetime.now())
    right_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            min_cost = float("inf")
            min_disparity = -1
            for d in range(ndisp):
                if right_cost_volume[d, h, w] < min_cost:
                    min_cost = right_cost_volume[d, h, w]
                    min_disparity = d
            assert min_disparity >= 0
            right_disparity_map[h, w] = min_disparity
    print "{}: disparity map done...".format(datetime.now())

    return left_disparity_map, right_disparity_map

# interpolation: left-right consistency check
# every pixel disparity has 3 status
# 0: match
# 1: mismatch
# 2: occlusion
def interpolation(left_disparity_map, right_disparity_map, ndisp):

    print "{}: doing left-right consistency check...".format(datetime.now())
    height, width = left_disparity_map.shape
    consistency_map = np.zeros([height, width], dtype=np.int32)

    for h in range(height):
        for w in range(width):
            left_disparity = int(left_disparity_map[h, w])
            # no corresponding pixel, takes as occlusion
            if w < left_disparity:
                consistency_map[h, w] = 2
                continue

            right_disparity = right_disparity_map[h, w-left_disparity]
            if abs(left_disparity - right_disparity) <= 1:
                # match
                continue

            # check if mismatch
            for d in range(min(w+1, ndisp)):
                if abs(d - right_disparity_map[h, w-d]) <= 1:
                    # mismatch
                    consistency_map[h, w] = 1
                    break

            # otherwise take as occlusion
            if consistency_map[h, w] == 0:
                consistency_map[h, w] = 2
     
    print "{}: doing interpolation...".format(datetime.now())
    int_left_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            if consistency_map[h, w] == 0:
                int_left_disparity_map[h, w] = left_disparity_map[h, w]
            elif consistency_map[h, w] == 1:
                # mismatch, taken median value from nearest match neighbours in 4 directions
                # NOTE: in origin paper, they use 16 directions
                count = 0
                neighbours = []

                # right
                for w_ in range(w+1, width):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        neighbours.append(left_disparity_map[h, w_])
                        break

                # left
                for w_ in range(w-1, -1, -1):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        neighbours.append(left_disparity_map[h, w_])
                        break

                # bottom
                for h_ in range(h+1, height):
                    if consistency_map[h_, w] == 0:
                        count += 1
                        neighbours.append(left_disparity_map[h_, w])
                        break

                # up
                for h_ in range(h-1, -1, -1):
                    if consistency_map[h_, w] == 0:
                        count += 1
                        neighbours.append(left_disparity_map[h_, w])
                        break

                neighbours = np.array(neighbours, dtype=np.float32)

                # no nearest match, use the raw value
                if count == 0:
                    int_left_disparity_map[h, w] = left_disparity_map[h, w]
                else:
                    int_left_disparity_map[h, w] = np.median(neighbours)

            else:
                # occlusion
                # just use the nearest match neighbour value on the right
                # NOTE: in the origin paper, they use left rather than left

                # right
                count = 0
                for w_ in range(w+1, width):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        int_left_disparity_map[h, w] = left_disparity_map[h, w_]
                        break

                # no match neighbour found, use the raw value
                if count == 0:
                    int_left_disparity_map[h, w] = left_disparity_map[h, w]

    left_disparity_map = int_left_disparity_map
    print "{}: interpolation done...".format(datetime.now())

    return left_disparity_map

# subpixel enhancement
def subpixel_enhance(left_disparity_map, left_cost_volume):

    print "{}: doing subpixel enhancement...".format(datetime.now())
    ndisp, height, width = left_cost_volume.shape
    se_left_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            d = left_disparity_map[h, w]
            if d == 0 or d == ndisp - 1:
                se_left_disparity_map[h, w] = d
            else:
                C_m = left_cost_volume[d - 1, h, w]
                C_p = left_cost_volume[d + 1, h, w]
                C = left_cost_volume[d, h, w]
                se_left_disparity_map[h, w] = d - (C_p - C_m) / (2. * (C_p - 2. * C + C_m))

    print "{}: subpixel enhancement done...".format(datetime.now())
    
    return se_left_disparity_map

# refinement1: median filter
def median_filter(left_disparity_map, filter_height, filter_width):

    print "{}: doing median filter...".format(datetime.now())
    height, width = left_disparity_map.shape
    med_left_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            patch_hs = max(0, h - (filter_height-1)/2)
            patch_he = min(height, h + (filter_height-1)/2 + 1)
            patch_ws = max(0, w - (filter_width-1)/2)
            patch_we = min(width, w + (filter_width-1)/2 + 1)
            patch = left_disparity_map[patch_hs:patch_he, patch_ws:patch_we]
            median = np.median(patch)
            med_left_disparity_map[h, w] = median

    print "{}: median filtering done...".format(datetime.now())

    return med_left_disparity_map

# refinement2: bilateral filter
def bilateral_filter(left_image, left_disparity_map, filter_height, filter_width, mean, std_dev, blur_threshold):

    print "{}: doing bilateral filter...".format(datetime.now())
    height, width = left_disparity_map.shape
    g = util.normal(mean, std_dev)

    # precompute filter weight
    center_h = (filter_height - 1)/2
    center_w = (filter_width - 1)/2
    bi_filter = np.zeros([filter_height, filter_width], dtype=np.float32)
    for h in range(filter_height):
        for w in range(filter_width):
            bi_filter[h, w] = g(np.sqrt((h - center_h)**2 + (w - center_w)**2))

    # filter
    bi_left_disparity_map = np.ndarray([height, width], dtype=np.float32)
    for h in range(height):
        for w in range(width):
            patch_hs = max(0, h - (filter_height-1)/2)
            patch_he = min(height, h + (filter_height-1)/2 + 1)
            patch_ws = max(0, w - (filter_width-1)/2)
            patch_we = min(width, w + (filter_width-1)/2 + 1)

            patch = left_disparity_map[patch_hs:patch_he, patch_ws:patch_we]
            
            filter_hs = center_h - (h - patch_hs)
            filter_he = center_h + (patch_he - h)
            filter_ws = center_w - (w - patch_ws)
            filter_we = center_w + (patch_we - w)
            tem_filter = bi_filter[filter_hs:filter_he, filter_ws:filter_we]
            assert tem_filter.shape == patch.shape

            image_patch = left_image[patch_hs:patch_he, patch_ws:patch_we]
            cur_inten = left_image[h, w]
            image_patch = image_patch - cur_inten
            image_patch = np.linalg.norm(image_patch, axis=-1)
            image_patch = (image_patch < blur_threshold).astype(np.float32)
            assert image_patch.shape == tem_filter.shape
            final_filter = np.multiply(image_patch, tem_filter)
            Wsum = np.sum(final_filter)
            
            final_patch = np.multiply(final_filter, patch)
            bi_left_disparity_map[h, w] = np.sum(final_patch) / Wsum

    print "{}: bilateral filtering done...".format(datetime.now())

    return bi_left_disparity_map

# do semi-global matching for one direction r
# choice is used to specify whether it's left cost volume or right cost volume
# NOTE: this implementation only supports SGM along axis-directions as the origin MC-CNN used, for other directions like digional, 
# it is approximated by alternative horizontal and vertical steps
def semi_global_matching(left_image, right_image, cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, choice):

    ndisp, height, width = cost_volume.shape
    assert choice == "R" or choice == "L"

    rh = r[0]
    rw = r[1]

    assert rh*rw == 0
    if rh >= 0:
        starth = rh
        endh = height
        steph = 1
    else:
        starth = height+rh-1
        endh = -1
        steph = -1

    if rw >= 0:
        startw = rw
        endw = width
        stepw = 1
    else:
        startw = width+rw-1
        endw = -1
        stepw = -1

    # first compute penalty factors P1 and P2 for all disparities of every pixel
    P1 = sgm_P1*np.ones([ndisp, height, width], dtype=np.float32)
    P2 = sgm_P2*np.ones([ndisp, height, width], dtype=np.float32)
    D1 = np.zeros([height, width], dtype=np.float32)
    D2 = np.zeros([ndisp, height, width], dtype=np.float32)

    if choice == "L":
        for h in range(starth, endh, steph):
            for w in range(startw, endw, stepw):
                D1[h, w] = np.linalg.norm(left_image[h, w] - left_image[h - rh, w - rw])

        for h in range(starth, endh, steph):
            for w in range(startw, endw, stepw):
                for d in range(ndisp):
                    if w - d < 0 or w - rw - d < 0:
                        continue
                    
                    D2[d, h, w] = np.linalg.norm(right_image[h, w - d] - right_image[h - rh, w - rw - d])

    else:
        for h in range(starth, endh, steph):
            for w in range(startw, endw, stepw):
                D1[h, w] = np.linalg.norm(right_image[h, w] - right_image[h - rh, w - rw])

        for h in range(starth, endh, steph):
            for w in range(startw, endw, stepw):
                for d in range(ndisp):
                    if w + d >= width or w - rw + d >= width:
                        continue
                    
                    D2[d, h, w] = np.linalg.norm(left_image[h, w + d] - left_image[h - rh, w - rw + d])

    condition1 = np.logical_and(D1 < sgm_D, D2 < sgm_D)
    condition2 = np.logical_and(D1 >= sgm_D, D2 >= sgm_D)
    condition3 = np.logical_not(np.logical_or(condition1, condition2))
    P1[condition2] = P1[condition2] / sgm_Q2
    P2[condition2] = P2[condition2] / sgm_Q2
    P1[condition3] = P1[condition3] / sgm_Q1
    P2[condition3] = P2[condition3] / sgm_Q1

    # dynamic programming optimization
    cost_volume_rd = cost_volume
    for h in range(starth, endh, steph):
        for w in range(startw, endw, stepw):
            # d = 0
            d = 0
            item1 = cost_volume_rd[d, h-rh, w-rw]
            item3 = cost_volume_rd[d+1, h-rh, w-rw] + P1[d, h, w]
            item4 = np.amin(cost_volume_rd[:, h-rh, w-rw]) + P2[d, h, w]
            cost_volume_rd[d, h, w] = cost_volume_rd[d, h, w] + min(item1, min(item3, item4)) - np.amin(cost_volume_rd[:, h-rh, w-rw])

            for d in range(1, ndisp-1):
                item1 = cost_volume_rd[d, h-rh, w-rw]
                item2 = cost_volume_rd[d-1, h-rh, w-rw] + P1[d, h, w]
                item3 = cost_volume_rd[d+1, h-rh, w-rw] + P1[d, h, w]
                item4 = np.amin(cost_volume_rd[:, h-rh, w-rw]) + P2[d, h, w]
                cost_volume_rd[d, h, w] = cost_volume_rd[d, h, w] + min(min(item1, item2), min(item3, item4)) - np.amin(cost_volume_rd[:, h-rh, w-rw])

            # d = ndisp-1
            d = ndisp - 1
            item1 = cost_volume_rd[d, h-rh, w-rw]
            item2 = cost_volume_rd[d-1, h-rh, w-rw] + P1[d, h, w]
            item4 = np.amin(cost_volume_rd[:, h-rh, w-rw]) + P2[d, h, w]
            cost_volume_rd[d, h, w] = cost_volume_rd[d, h, w] + min(min(item1, item2), item4) - np.amin(cost_volume_rd[:, h-rh, w-rw])

    return cost_volume_rd

# compute union region in cross-based cost aggregation
def compute_cross_region(image, intensity_threshold, distance_threshold):

    # the cross union region can be decomposed into vertical and horizontal region
    # and is more efficient
    height, width= image.shape[:2]
    union_region_v = np.ndarray([height, width, (2*distance_threshold), 2], dtype=np.int32)
    union_region_v_num = np.zeros([height, width], dtype=np.int32)

    # compute vertical regions of every pixel
    for h in range(height):
        for w in range(width):
            count = 0
            cur_inten = image[h, w]
            # extend the top arm 
            for h_bias in range(min(distance_threshold, h+1)):
                h_ = h - h_bias
                tem_inten = image[h_, w]
                if np.linalg.norm(cur_inten - tem_inten) >= intensity_threshold:
                    break
                union_region_v[h, w, count] = np.array([h_, w])
                count += 1
            # extend the bottom arm 
            for h_bias in range(1, min(distance_threshold, height-h)):
                h_ = h + h_bias
                tem_inten = image[h_, w]
                if np.linalg.norm(cur_inten - tem_inten) >= intensity_threshold:
                    break
                union_region_v[h, w, count] = np.array([h_, w])
                count += 1
            # update count, at least its self
            assert count >= 1 and count < 2 * distance_threshold
            union_region_v_num[h, w] = count

    union_region_h = np.ndarray([height, width, (2*distance_threshold), 2], dtype=np.int32)
    union_region_h_num = np.zeros([height, width], dtype=np.int32)
    # compute horizontal regions of every pixel
    for h in range(height):
        for w in range(width):
            count = 0
            cur_inten = image[h, w]
            # extend the left arm 
            for w_bias in range(min(distance_threshold, w+1)):
                w_ = w - w_bias
                tem_inten = image[h, w_]
                if np.linalg.norm(cur_inten - tem_inten) >= intensity_threshold:
                    break
                union_region_h[h, w, count] = np.array([h, w_])
                count += 1
            # extend the right arm 
            for w_bias in range(1, min(distance_threshold, width-w)):
                w_ = w + w_bias
                tem_inten = image[h, w_]
                if np.linalg.norm(cur_inten - tem_inten) >= intensity_threshold:
                    break
                union_region_h[h, w, count] = np.array([h, w_])
                count += 1
            # update count, at least its self
            assert count >= 1 and count < 2 * distance_threshold
            union_region_h_num[h, w] = count
                
    # compute the cross union region using vertical and horizontal regions
    # shape like this, see paper for details
    # +++++++|+++++++
    #   +++++|++++
    #    ++++|+
    #       +|+++
    max_num = (2*distance_threshold)**2
    union_region = np.ndarray([height, width, max_num, 2], dtype=np.int32)
    union_region_num = np.zeros([height, width], dtype=np.int32)
    for h in range(height):
        for w in range(width):
            count = 0
            v_num = union_region_v_num[h, w]
            for v in range(v_num):
                h_, w_ = union_region_v[h, w, v]
                hz_num = union_region_h_num[h_, w_]
                for hz in range(hz_num):
                    _h, _w = union_region_h[h_, w_, hz]
                    union_region[h, w, count] = np.array([_h, _w])
                    count += 1
            # update count
            assert count >= 1 and count < max_num
            union_region_num[h, w] = count
            # padding at invalid position with (-1, -1)
            union_region[h, w, count:max_num] = np.array([-1, -1])

    return union_region, union_region_num

# union region when consideing disparity
# this function can be used to shrink both left and right union regions based on choice("R" or "L")
def compute_disparity_union_region(left_union_region, left_union_region_num, \
                                   right_union_region, right_union_region_num, disparity, choice):

    assert choice == "R" or choice == "L"
    height, width, max_num = left_union_region.shape[0:3]
    assert disparity < width
    d_union_region = np.ndarray([height, width, max_num, 2], dtype=np.int32)
    d_union_region_num = np.zeros([height, width], dtype=np.int32)

    # for pixels approaching left/right boundary such that no according pixel for the disparity, 
    # just copy the raw union region
    if choice == "L":
        d_union_region[:, 0:disparity] = left_union_region[:, 0:disparity]
        d_union_region_num[:, 0:disparity] = left_union_region_num[:, 0:disparity]
        startw = disparity
        endw = width
        for h in range(height):
            for w in range(startw, endw):
                count = 0
                raw_num = left_union_region_num[h, w]
                for v in range(raw_num):
                    h_, w_ = left_union_region[h, w, v]
                    # for pixels without according pixel, just take it in
                    if w_ < disparity:
                        d__union_region[h, w, count] = np.array([h_, w_])
                        count += 1
                        continue
                    # judge whether the according pixel of (h_, w_)(i.e. (h_, w_-d)) 
                    # is in the right union_region of (h ,w-d)
                    pos = np.array([h_, w_-disparity], dtype=np.int32)
                    cur_right_union = right_union_region[h, w-disparity]
                    exist_num = cur_right_union[cur_right_union == pos].shape[0]
                    if exist_num > 0:
                        d_union_region[h, w, count] = np.array([h_, w_])
                        count += 1
                # update count, at least one and at most raw union region num
                assert count >= 1 and count <= raw_num
                d_union_region_num[h, w] = count
                d_union_region[h, w, count: max_num] = np.array([-1,-1])
    else:
        d_union_region[:, width-disparity:width] = right_union_region[:, width-disparity:width]
        d_union_region_num[:, width-disparity:width] = right_union_region_num[:, width-disparity:width]
        startw = 0
        endw = width-disparity
        for h in range(height):
            for w in range(startw, endw):
                count = 0
                raw_num = right_union_region_num[h, w]
                for v in range(raw_num):
                    h_, w_ = right_union_region[h, w, v]
                    # for pixels without according pixel, just take it in
                    if w_ + disparity >= width:
                        d_union_region[h, w, count] = np.array([h_, w_])
                        count += 1
                        continue
                    # judge whether the according pixel of (h_, w_)(i.e. (h_, w_+d)) 
                    # is in the left union_region of (h ,w+d)
                    pos = np.array([h_, w_+disparity], dtype=np.int32)
                    cur_left_union = left_union_region[h, w+disparity]
                    exist_num = cur_left_union[cur_left_union == pos].shape[0]
                    if exist_num > 0:
                        d_union_region[h, w, count] = np.array([h_, w_])
                        count += 1
                # update count, at least one and at most raw union region num
                assert count >= 1 and count <= raw_num
                d_union_region_num[h, w] = count
                d_union_region[h, w, count: max_num] = np.array([-1,-1])

    return d_union_region, d_union_region_num

