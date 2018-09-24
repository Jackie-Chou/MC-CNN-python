"""
    conduct stereo matching based on trained model + a series of post-processing
"""
import os
import util
import time
import cv2
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
from tqdm import tqdm
from process_functional import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="stereo matching based on trained model and post-processing")
parser.add_argument("-g", "--gpu", type=str, default="0", help="gpu id to use, \
                    multiple ids should be separated by commons(e.g. 0,1,2,3)")
parser.add_argument("-ps", "--patch_size", type=int, default=11, help="length for height/width of square patch")
parser.add_argument("--list_file", type=str, required=True, help="path to file containing left image list")
parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from. \
                    if None(default), model is initialized using default methods")
parser.add_argument("--data_dir", type=str, required=True, help="path to root dir to data.")
parser.add_argument("--save_dir", type=str, required=True, help="path to root dir to save results")
parser.add_argument("-t", "--tag", type=str, required=True, help="tag used to indicate one run")
parser.add_argument("-s", "--start", type=int, required=True, help="index of first image to do matching,\
                                                                    this is used for parallel matching of different images")
parser.add_argument("-e", "--end", type=int, required=True, help="index of last image to do matching")


# hyperparemeters, use suggested value from origin paper as default
parser.add_argument("--cbca_intensity", type=float, default=0.02, help="intensity threshold for cross-based cost aggregation")
parser.add_argument("--cbca_distance", type=float, default=14, help="distance threshold for cross-based cost aggregation")
parser.add_argument("--cbca_num_iterations1", type=float, default=2, help="distance threshold for cross-based cost aggregation")
parser.add_argument("--cbca_num_iterations2", type=float, default=16, help="distance threshold for cross-based cost aggregation")
parser.add_argument("--sgm_P1", type=float, default=2.3, help="hyperparemeter used in semi-global matching")
parser.add_argument("--sgm_P2", type=float, default=55.9, help="hyperparemeter used in semi-global matching")
parser.add_argument("--sgm_Q1", type=float, default=4, help="hyperparemeter used in semi-global matching")
parser.add_argument("--sgm_Q2", type=float, default=8, help="hyperparemeter used in semi-global matching")
parser.add_argument("--sgm_D", type=float, default=0.08, help="hyperparemeter used in semi-global matching")
parser.add_argument("--sgm_V", type=float, default=1.5, help="hyperparemeter used in semi-global matching")
parser.add_argument("--blur_sigma", type=float, default=6, help="hyperparemeter used in bilateral filter")
parser.add_argument("--blur_threshold", type=float, default=2, help="hyperparemeter used in bilateral filter")

# different file names
left_image_suffix = "im0.png"
left_gt_suffix = "disp0GT.pfm"
right_image_suffix = "im1.png"
right_gt_suffix = "disp1GT.pfm"
calib_suffix = "calib.txt"

out_file = "disp0MCCNN.pfm"
out_img_file = "disp0MCCNN.pgm"
out_time_file = "timeMCCNN.txt"

def main():
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    patch_height = args.patch_size
    patch_width = args.patch_size

    ######################
    left_image_list = args.list_file

    save_dir = args.save_dir
    data_dir = args.data_dir
    save_res_dir = os.path.join(save_dir, "submit_{}".format(args.tag))
    save_img_dir = os.path.join(save_dir, "submit_{}_imgs".format(args.tag))
    util.testMk(save_res_dir)
    util.testMk(save_img_dir)

    index = 0
    start = args.start
    end = args.end

    with open(left_image_list, "r") as i:
        img_paths = i.readlines()

    ####################
    # do matching
    for left_path in tqdm(img_paths):
        print "index: ".format(index)
        if index < start:
            index += 1
            print "passed"
            continue
        if index > end:
            break
        index += 1

        # get data path
        left_path = left_path.strip()
        right_path = left_path.replace(left_image_suffix, right_image_suffix)
        calib_path = left_path.replace(left_image_suffix, calib_suffix)
        
        # generate output path
        res_dir = left_path.replace(data_dir, save_res_dir)
        img_dir = left_path.replace(data_dir, save_img_dir)

        res_dir = res_dir[:res_dir.rfind(left_image_suffix)-1]
        img_dir = img_dir[:img_dir.rfind(left_image_suffix)-1]

        util.recurMk(res_dir)
        util.recurMk(img_dir)

        out_path = os.path.join(res_dir, out_file)
        out_time_path = os.path.join(res_dir, out_time_file)
        out_img_path = os.path.join(img_dir, out_img_file)

        height, width, ndisp = util.parseCalib(calib_path)
        print "left_image: {}\nright_image: {}".format(left_path, right_path)
        print "height: {}, width: {}, ndisp: {}".format(height, width, ndisp)
        print "out_path: {}\nout_time_path: {}\nout_img_path: {}".format(out_path, out_time_path, out_img_path)
        
        # reading images
        left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
        right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))
        left_image = np.expand_dims(left_image, axis=2)
        right_image = np.expand_dims(right_image, axis=2)
        assert left_image.shape == (height, width, 1)
        assert right_image.shape == (height, width, 1)
        print "{}: images read".format(datetime.now())

        # start timer for time file
        stTime = time.time()

        # compute features
        left_feature, right_feature = compute_features(left_image, right_image, patch_height, patch_width, args.resume)
        print left_feature.shape
        print "{}: features computed".format(datetime.now())

        # form cost-volume
        left_cost_volume, right_cost_volume = compute_cost_volume(left_feature, right_feature, ndisp)
        print "{}: cost-volume computed".format(datetime.now())

        # cost-volume aggregation
        print "{}: beginning cost-volume aggregation, this could take long".format(datetime.now())
        left_cost_volume, right_cost_volume = cost_volume_aggregation(left_image, right_image,left_cost_volume,right_cost_volume,\
                                                               args.cbca_intensity, args.cbca_distance, args.cbca_num_iterations1) 
        print "{}: cost-volume aggregated".format(datetime.now())

        '''
        # semi-global matching
        print "{}: beginning semi-global matching".format(datetime.now())
        left_cost_volume, right_cost_volume = SGM_average(left_cost_volume, right_cost_volume, left_image, right_image, \
                                                     args.sgm_P1, args.sgm_P2, args.sgm_Q1, args.sgm_Q2, args.sgm_D, args.sgm_V)
        print "{}: semi-global matched".format(datetime.now())
        '''

        '''
        # cost-volume aggregation afterhand
        print "{}: beginning cost-volume aggregation, this could take long".format(datetime.now())
        left_cost_volume, right_cost_volume = cost_volume_aggregation(left_image, right_image,left_cost_volume,right_cost_volume,\
                                                               args.cbca_intensity, args.cbca_distance, args.cbca_num_iterations2) 
        print "{}: cost-volume aggregated".format(datetime.now())
        '''

        # disparity map making 
        left_disparity_map, right_disparity_map = disparity_prediction(left_cost_volume, right_cost_volume)
        print "{}: disparity predicted".format(datetime.now())

        # interpolation
        left_disparity_map = interpolation(left_disparity_map, right_disparity_map, ndisp)
        print "{}: disparity interpolated".format(datetime.now())

        # refinement
        # 5*5 median filter 
        left_disparity_map = median_filter(left_disparity_map, 5, 5)

        """
        # bilateral filter
        left_disparity_map = bilateral_filter(left_image, left_disparity_map, 5, 5, 0, args.blur_sigma, args.blur_threshold)
        print "{}: refined".format(datetime.now())
        """

        # end timer
        endTime = time.time()

        # save as pgm and pfm
        util.saveDisparity(left_disparity_map, out_img_path)
        util.writePfm(left_disparity_map, out_path)
        util.saveTimeFile(endTime-stTime, out_time_path)
        print "{}: saved".format(datetime.now())

if __name__ == "__main__":
    main()
