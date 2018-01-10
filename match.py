import os
import util
import time
import cv2
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
from model import NET
from datagenerator import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num", type=int, required=True)
parser.add_argument("-i", "--inten", type=float, required=True)
parser.add_argument("-s", "--start", type=int, required=True)
parser.add_argument("-e", "--end", type=int, required=True)
parser.add_argument("-g", "--gpuid", type=int, required=True)
parser.add_argument("-c", "--choice", type=int, required=True)
args = parser.parse_args()
choice = args.choice
"""
Configuration settings
"""
#######################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

######################
file_prefix = '/home/zxz/stereo_matching'
data_prefix = '/local/MI/zxz/stereo_matching'
restore_epoch = 100000
# equal to size of receptive field
patch_height = 11
patch_width = 11

######################
# input left and right image path
intensity_threshold = args.inten
distance_threshold = 14
max_average_time = 4
######################
# restore checkpoint from
old_checkpoint_path = os.path.join(file_prefix, "record/old_tfrecordQ2")

left_image_list = os.path.join(data_prefix, 'data/alltrainlQ.in')
left_image_suffix = "im0.png"
left_gt_suffix = "disp0GT.pfm"
right_image_suffix = "im1.png"
right_gt_suffix = "disp1GT.pfm"
out_file = "disp0MCCNNzxz2.pfm"
out_img_file = "disp0MCCNNzxz2.pgm"
calib_suffix = "calib.txt"
out_time_file = "timeMCCNNzxz2.txt"
out_featurel_file = "featurel.npy"
out_featurer_file = "featurer.npy"

in_prefix = "/local/MI/zxz/stereo_matching/data/MiddEval3"
out_prefix = "/home/zxz/stereo_matching/data/submit{}_2".format(args.num)
out_img_prefix = "/home/zxz/stereo_matching/data/submit{}_2_imgs".format(args.num)
util.testMk(out_prefix)
util.testMk(out_img_prefix)

def compute_features(left_image, right_image):
    # pad images to make the final feature map size = (height, width..)
    auged_left_image = np.zeros([1, height+patch_height-1, width+patch_width-1, 3], dtype=np.float32)
    auged_right_image = np.zeros([1, height+patch_height-1, width+patch_width-1, 3], dtype=np.float32)
    row_start = (patch_height - 1)/2
    col_start = (patch_width - 1)/2
    auged_left_image[0, row_start: row_start+height, col_start: col_start+width] = left_image
    auged_right_image[0, row_start: row_start+height, col_start: col_start+width] = right_image
    # quarter size
    # TF placeholder for graph input
    #x = tf.placeholder(tf.float32, shape=[1, height/2+patch_height-1, width/2+patch_width-1, 3])  
    x = tf.placeholder(tf.float32, shape=[1, height+patch_height-1, width+patch_width-1, 3])  

    # Initialize model
    model = NET(x, batch_size=1)
    saver = tf.train.Saver()

    # Link variable to model output
    features = model.features
    #assert features.shape == (1, height/2, width/2, 112)
    assert features.shape == (1, height, width, 112)

    # compute features on both images
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, \
            allow_soft_placement=True)) as sess:
            #gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        restore_path = os.path.join(old_checkpoint_path, 'model_epoch%d.ckpt'%(restore_epoch))
        print "{}: restoring from {}...".format(datetime.now(), restore_path)
        saver.restore(sess, restore_path)

        print "{}: features computing...".format(datetime.now())
        '''
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

        featuresl = np.squeeze(featuresl, axis=0)
        featuresr = np.squeeze(featuresr, axis=0) # (height, width, 112)
        assert featuresl.shape == (height, width, 112)
        assert featuresr.shape == (height, width, 112)
        print "{}: features computed done...".format(datetime.now())
    tf.reset_default_graph()
    return featuresl, featuresr

# compute correlation between pixels, note max disparity is ndisp
# cost_volume[d, x, y] = correlation between pixel (x, y) in left image and pixel (x, y - d) in right image
# NOTE: y - d >= 0 so some pixels approaching boundary may have no matching score at some disparities
def compute_cost_volume(featuresl, featuresr):
    print "{}: computing cost_volume for left image...".format(datetime.now())
    left_cost_volume = np.zeros([ndisp, height, width], dtype=np.float32)

    '''
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        xl = tf.placeholder(tf.float32, shape=None)
        xr = tf.placeholder(tf.float32, shape=None)
        y = tf.reduce_sum(tf.multiply(xl, xr), axis=-1)
    '''
    tem_xl = featuresl
    tem_xr = featuresr
    for d in range(ndisp):
        print "{}: disparity {}...".format(datetime.now(), d)
        left_cost_volume[d, :, d:] = np.sum(np.multiply(tem_xl, tem_xr), axis=-1)
        tem_xl = tem_xl[:, 1:]
        tem_xr = tem_xr[:, :tem_xr.shape[1]-1]

    # NOTE: should not use 0 at these pixels since average is taken w.r.t. all pixels
    # change to use average cost over rightside 3 cols
    # NOTE: Ablation study, whether this can help
    print "{}: padding with average...".format(datetime.now())
    ''''
    x = tf.placeholder(tf.float32, shape=None)
    x_mean = tf.reduce_mean(x, axis=-1)
    '''
    for d in range(ndisp-1, 0, -1):
        print "{}: disparity {}...".format(datetime.now(), d)
        left_cost_volume[d:ndisp, :, d-1] = np.mean(left_cost_volume[d:ndisp, :, d:d+3], axis=-1)
    print "{}: cost_volume for left image computed...".format(datetime.now())

    # do it for right image again
    # NOTE: just copy from left_cost_volume since dot product is symmetric
    # NOTE: Ablation study, whether this can help
    print "{}: computing cost_volume for right image...".format(datetime.now())
    right_cost_volume = np.zeros([ndisp, height, width], dtype=np.float32)
    for d in range(ndisp):
        right_cost_volume[d, :, :width-d] = left_cost_volume[d, :, d:]
    for d in range(ndisp-1, 0, -1):
        right_cost_volume[d:ndisp, :, width-d] = np.mean(right_cost_volume[d:ndisp, :, width-d-3:width-d], axis=-1)
    print "{}: cost_volume for right image computed...".format(datetime.now())

    # convert from matching score to cost
    # match score larger = cost smaller
    left_cost_volume = np.amax(left_cost_volume) - left_cost_volume
    right_cost_volume = np.amax(right_cost_volume) - right_cost_volume
    return left_cost_volume, right_cost_volume

# cost volume aggregation
# use cross-based cost aggregation
# NOTE: Ablation study, whether this can help and how the result change w.r.t. different thresholds
def cost_volume_aggregation(left_image, right_image, left_cost_volume, right_cost_volume, intensity_threshold, distance_threshold, max_average_time):
    left_union_region, left_union_region_num = cross_cost_aggregation(left_image, intensity_threshold, distance_threshold)
    right_union_region, right_union_region_num = cross_cost_aggregation(right_image, intensity_threshold, distance_threshold)
    """
    this is too large and is impractical:)
    # then compute disparity-dependent union regions
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
        # averaging for every disparity
        #for d in range(ndisp):
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
        # averaging for every disparity
        #for d in range(ndisp):
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
                sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D):
    # use four directions dynamic programming and take average
    print "{}: semi-global matching for left image...".format(datetime.now())
    # right
    print "{}: right".format(datetime.now())
    r = (0, 1)
    left_cost_volume_right = semi_global_matching(left_image, right_image, left_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "L")
    # left
    print "{}: left".format(datetime.now())
    r = (0, -1)
    left_cost_volume_left = semi_global_matching(left_image, right_image, left_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "L")
    # up
    print "{}: up".format(datetime.now())
    r = (1, 0)
    left_cost_volume_up = semi_global_matching(left_image, right_image, left_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "L")
    # bottom
    print "{}: bottom".format(datetime.now())
    r = (-1, 0)
    left_cost_volume_bottom = semi_global_matching(left_image, right_image, left_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "L")
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
    # up
    print "{}: up".format(datetime.now())
    r = (1, 0)
    right_cost_volume_up = semi_global_matching(left_image, right_image, right_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "R")
    # bottom
    print "{}: bottom".format(datetime.now())
    r = (-1, 0)
    right_cost_volume_bottom = semi_global_matching(left_image, right_image, right_cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, "R")
    # taken average
    right_cost_volume = (right_cost_volume_right + right_cost_volume_left + right_cost_volume_up + right_cost_volume_bottom) / 4.
    print "{}: semi-global matching done...".format(datetime.now())
    return left_cost_volume, right_cost_volume

# disparity prediction
# simple "Winner-take-All"
def disparity_prediction(left_cost_volume, right_cost_volume):
    print "{}: left disparity map making...".format(datetime.now())
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
def interpolation(left_disparity_map, right_disparity_map):
    print "{}: doing left-right consistency check...".format(datetime.now())
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
                # top
                for h_ in range(h+1, height):
                    if consistency_map[h_, w] == 0:
                        count += 1
                        neighbours.append(left_disparity_map[h_, w])
                        break
                # bottom
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
                # just use the nearest left/right match neighbour value
                # left
                count = 0
                for w_ in range(w-1, -1, -1):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        int_left_disparity_map[h, w] = left_disparity_map[h, w_]
                        break
                if count == 0:
                    # right
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

# refinement1: median filter
def median_filter(left_disparity_map, filter_height, filter_width):
    print "{}: doing median filter...".format(datetime.now())
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
    left_disparity_map = med_left_disparity_map
    print "{}: median filtering done...".format(datetime.now())
    return left_disparity_map

# refinement2: bilateral filter
def bilateral_filter(left_disparity_map, filter_height, filter_width, mean, std_dev):
    print "{}: doing bilateral filter...".format(datetime.now())
    g = normal(mean, std_dev)
    cneter_h = (filter_height - 1)/2
    cneter_w = (filter_width - 1)/2
    bi_filter = np.zeros([filter_height, filter_width], dtype=np.float32)
    for h in range(filter_height):
        for w in range(filter_width):
            bi_filter[h, w] = g(np.sqrt((h - center_h)**2 + (w - center_w)**2))

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
            filter_ws = center_w + (patch_we - w)
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
            bi_left_dispatiy_map[h, w] = np.sum(final_patch) / Wsum

    left_disparity_map = bi_left_disparity_map
    print "{}: bilateral filtering done...".format(datetime.now())
    return left_disparity_map

def semi_global_matching(left_image, right_image, cost_volume, r, sgm_P1, sgm_P2, sgm_Q1, sgm_Q2, sgm_D, choice):
    assert choice == "R" or choice == "L"
    rh = r[0]
    rw = r[1]
    # just do axis-direction dp
    assert rh*rw == 0
    if rh >= 0:
        starth = rh
        endh = height
        steph = 1
    else:
        # NOTE: always [)
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

    P1 = sgm_P1*np.ones([ndisp, height, width], dtype=np.float32)
    P2 = sgm_P2*np.ones([ndisp, height, width], dtype=np.float32)
    D1 = np.zeros([height, width], dtype=np.float32)
    D2 = np.zeros([ndisp, height, width], dtype=np.float32)

    if choice == "L":
        D1[starth:(endh if endh != -1 else None):steph, startw:(endw if endw != -1 else None):stepw] = \
                np.linalg.norm(left_image[starth:(endh if endh != -1 else None):steph, startw:(endw if endw != -1 else None):stepw] - \
                               left_image[starth-rh:(endh-rh if endh-rh != -1 else None):steph, startw-rw:(endw-rw if endw-rw != -1 else None):stepw], axis=-1)
        for d in range(ndisp):
            startwd = min(startw+d, width-1)
            endwd = min(endw+d, width)
            '''
            print "d: {} startwd: {} endwd: {} rw: {}".format(d, startwd, endwd, rw)
            print "startwd-d:endwd-d {}:{}, startd-d-rw:endwd-d-rw {}:{}".format(startwd-d, (endwd-d if endwd-d != -1 else None), startwd-d-rw,(endwd-d-rw if endwd-d-rw != -1 else None))
            print right_image[starth:(endh if endh != -1 else None):steph, startwd-d:(endwd-d if endwd-d != -1 else None):stepw].shape
            print right_image[starth-rh:(endh-rh if endh-rh != -1 else None):steph, startwd-d-rw:(endwd-d-rw if endwd-d-rw != -1 else None):stepw].shape
            '''

            D2[d, starth:(endh if endh != -1 else None):steph, startwd:(endwd if endwd != -1 else None):stepw] = \
                    np.linalg.norm(right_image[starth:(endh if endh != -1 else None):steph, startwd-d:(endwd-d if endwd-d != -1 else None):stepw] - \
                                   right_image[starth-rh:(endh-rh if endh-rh != -1 else None):steph, startwd-d-rw:(endwd-d-rw if endwd-d-rw != -1 else None):stepw], axis=-1)
    else:
        D1[starth:(endh if endh != -1 else None):steph, startw:(endw if endw != -1 else None):stepw] = \
                np.linalg.norm(right_image[starth:(endh if endh != -1 else None):steph, startw:(endw if endw != -1 else None):stepw] - \
                               right_image[starth-rh:(endh-rh if endh-rh != -1 else None):steph, startw-rw:(endw-rw if endw-rw != -1 else None):stepw], axis=-1)
        for d in range(ndisp):
            startwd = max(startw-d, 0)
            endwd = max(endw-d, -1)
            D2[d, starth:(endh if endh != -1 else None):steph, startwd:(endwd if endwd != -1 else None):stepw] = \
                    np.linalg.norm(left_image[starth:(endh if endh != -1 else None):steph, startwd+d:(endwd+d if endwd+d != -1 else None):stepw] - \
                                   left_image[starth-rh:(endh-rh if endh-rh != -1 else None):steph, startwd+d-rw:(endwd+d-rw if endwd+d-rw != -1 else None):stepw], axis=-1)

    condition1 = np.logical_and(D1 < sgm_D, D2 < sgm_D)
    condition2 = np.logical_and(D1 >= sgm_D, D2 >= sgm_D)
    condition3 = np.logical_not(np.logical_or(condition1, condition2))
    P1[condition2] /= sgm_Q2
    P2[condition2] /= sgm_Q2
    P1[condition3] /= sgm_Q1
    P2[condition3] /= sgm_Q1

    # dynamic programming optimization
    cost_volume_rd = cost_volume
    for h in range(starth, endh, steph):
        for w in range(startw, endw, stepw):
            # d = 0
            d = 0
            item1 = cost_volume_rd[d, h-rh, w-rw]
            item3 = cost_volume_rd[d+1, h-rh, w-rw] + P1[d, h, w]
            item4 = np.amin(cost_volume_rd[d+2:, h-rh, w-rw]) + P2[d, h, w]
            cost_volume_rd[d, h, w] += min(item1, min(item3, item4))
            # d = 1
            d = 1
            item1 = cost_volume_rd[d, h-rh, w-rw]
            item2 = cost_volume_rd[d-1, h-rh, w-rw] + P1[d, h, w]
            item3 = cost_volume_rd[d+1, h-rh, w-rw] + P1[d, h, w]
            item4 = np.amin(cost_volume_rd[d+2:, h-rh, w-rw]) + P2[d, h, w]
            cost_volume_rd[d, h, w] += min(min(item1, item2), min(item3, item4))

            for d in range(2, ndisp-2):
                item1 = cost_volume_rd[d, h-rh, w-rw]
                item2 = cost_volume_rd[d-1, h-rh, w-rw] + P1[d, h, w]
                item3 = cost_volume_rd[d+1, h-rh, w-rw] + P1[d, h, w]
                item4 = np.minimum(np.amin(cost_volume_rd[:d-1, h-rh, w-rw]), np.amin(cost_volume_rd[d+2:, h-rh, w-rw])) + P2[d, h, w]
                cost_volume_rd[d, h, w] += min(min(item1, item2), min(item3, item4))

            # d = ndisp-2
            d = ndisp-2
            item1 = cost_volume_rd[d, h-rh, w-rw]
            item2 = cost_volume_rd[d-1, h-rh, w-rw] + P1[d, h, w]
            item3 = cost_volume_rd[d+1, h-rh, w-rw] + P1[d, h, w]
            item4 = np.amin(cost_volume_rd[:d-1, h-rh, w-rw]) + P2[d, h, w]
            cost_volume_rd[d, h, w] += min(min(item1, item2), min(item3, item4))
            # d = ndisp-1
            d = ndisp - 1
            item1 = cost_volume_rd[d, h-rh, w-rw]
            item2 = cost_volume_rd[d-1, h-rh, w-rw] + P1[d, h, w]
            item4 = np.amin(cost_volume_rd[:d-1, h-rh, w-rw]) + P2[d, h, w]
            cost_volume_rd[d, h, w] += min(min(item1, item2), item4)

    return cost_volume_rd


def cross_cost_aggregation(image, intensity_threshold, distance_threshold):
    # cost volume aggregation
    # use cross-based cost aggregation
    # the cross union region can be decomposed into vertical and horizontal region
    # and is more efficient
    height, width= image.shape[0:2]
    union_region_v = np.ndarray([height, width, (2*distance_threshold), 2], dtype=np.int32)
    union_region_v_num = np.zeros([height, width], dtype=np.int32)
    # compute vertical regions of every pixel
    for h in range(height):
        for w in range(width):
            count = 0
            cur_inten = image[h, w]
            # extend the bottom arm 
            for h_bias in range(min(distance_threshold, h+1)):
                h_ = h - h_bias
                tem_inten = image[h_, w]
                if np.linalg.norm(cur_inten - tem_inten) >= intensity_threshold:
                    break
                union_region_v[h, w, count] = np.array([h_, w])
                count += 1
            # extend the top arm 
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
    # shape is like a tree
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

# union region shrinkage considering disparity
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

index = 0
start = args.start
end = args.end
i = open(left_image_list, "r")
for left_path in i.readlines():
    print "index: ".format(index)
    if index < start:
        index += 1
        print "passed"
        continue
    if index > end:
        break
    index += 1

    left_path = left_path.strip()
    right_path = left_path.replace(left_image_suffix, right_image_suffix)
    calib_path = left_path.replace(left_image_suffix, calib_suffix)
    
    items = left_path.split('/')
    trainDir = items[-3]
    imageDir = items[-2]

    outSubDir = os.path.join(out_prefix, trainDir)
    outSubSubDir = os.path.join(outSubDir, imageDir)
    outPath = os.path.join(outSubSubDir, out_file)
    outTimePath = os.path.join(outSubSubDir, out_time_file)
    outFlPath = os.path.join(outSubSubDir, out_featurel_file)
    outFrPath = os.path.join(outSubSubDir, out_featurer_file)
    util.testMk(outSubDir)
    util.testMk(outSubSubDir)

    outImgSubDir = os.path.join(out_img_prefix, trainDir)
    outImgSubSubDir = os.path.join(outImgSubDir, imageDir)
    outImgPath = os.path.join(outImgSubSubDir, out_img_file)
    util.testMk(outImgSubDir)
    util.testMk(outImgSubSubDir)

    height, width, ndisp = util.parseCalib(calib_path)
    print "left_image: {}, right_image: {}".format(left_path, right_path)
    print "height: {}, width: {}, ndisp: {}".format(height, width, ndisp)
    print "outPath: {}, outTimePath: {}, outImgPath: {}".format(outPath, outTimePath, outImgPath)
    
    # reading images
    left_image = cv2.imread(left_path).astype(np.float32)
    right_image = cv2.imread(right_path).astype(np.float32)
    assert left_image.shape == (height, width, 3)
    assert right_image.shape == (height, width, 3)
    print "{}: images read".format(datetime.now())

    # choice 0: just compute feartures and save
    if choice == 0:
        left_feature, right_feature = compute_features(left_image, right_image)
        np.save(outFlPath, left_feature)
        np.save(outFrPath, right_feature)
        print "{}: features saved".format(datetime.now())

    elif choice == 1:
        # start doing computing
        # start timer for time file
        stTime = time.time()

        # compute features
        assert os.path.exists(outFlPath) and os.path.exists(outFrPath)
        left_feature = np.load(outFlPath)
        right_feature = np.load(outFrPath)
        print "{}: features loaded".format(datetime.now())

        # compute cost-volume
        left_cost_volume, right_cost_volume = compute_cost_volume(left_feature, right_feature)
        print "{}: cost-volume computed".format(datetime.now())

        # image process
        left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
        right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))
        print "{}: image processed".format(datetime.now())

        # cost-volume aggregation
        left_cost_volume, right_cost_volume = cost_volume_aggregation(left_image, right_image, left_cost_volume, right_cost_volume, 
                                                                        intensity_threshold, distance_threshold, max_average_time) 
        print "{}: cost-volume aggregated".format(datetime.now())

        # semi-global matching
        # ignoring this since the parameters are too many and there's no time to finetune all parameters

        # disparity map making 
        left_disparity_map, right_disparity_map = disparity_prediction(left_cost_volume, right_cost_volume)
        print "{}: disparity predicted".format(datetime.now())

        # interpolation
        left_disparity_map = interpolation(left_disparity_map, right_disparity_map)
        print "{}: disparity interpolated".format(datetime.now())

        # refinement
        # ignoring this

        # end timer
        endTime = time.time()

        # save as pgm and pfm
        util.saveDisparity(left_disparity_map, outImgPath)
        util.writePfm(left_disparity_map, outPath)
        util.saveTimeFile(endTime-stTime, outTimePath)
        print "{}: saved".format(datetime.now())
i.close()
