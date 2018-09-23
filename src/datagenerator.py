"""
    data generator class
"""
import os
import numpy as np
import cv2
import copy
from util import readPfm
import random
from tensorflow import expand_dims

class ImageDataGenerator:
    """
        input image patch pairs generator
    """
    def __init__(self, left_image_list_file, shuffle=False, 
                 patch_size=(11, 11),
                 in_left_suffix='im0.png',
                 in_right_suffix='im1.png',
                 gt_suffix='disp0GT.pfm',
                 # tunable hyperparameters
                 # see origin paper for details
                 dataset_neg_low=1.5, dataset_neg_high=6,
                 dataset_pos=0.5
                 ):
        """
            left_image_list_file: path to text file containing training set left image PATHS, one path per line
            list of left image paths are formed directly by reading lines from file 
            list of corresponding right image and ground truth disparity image paths are 
            formed by replacing in_left_suffix with in_right_suffix and gt_suffix from every left image path
        """
                
        # Init params
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.in_left_suffix = in_left_suffix
        self.in_right_suffix = in_right_suffix
        self.gt_suffix = gt_suffix
        self.dataset_neg_low = dataset_neg_low
        self.dataset_neg_high = dataset_neg_high
        self.dataset_pos = dataset_pos

        # the pointer indicates which image are next to be used
        # a mini-batch is fully constructed using one image(pair)
        self.pointer = 0

        self.read_image_list(left_image_list_file)
        self.prefetch()
        if self.shuffle:
            self.shuffle_data()

    def read_image_list(self, image_list):
        """
            form lists of left, right & ground truth paths
        """
        with open(image_list) as f:

            lines = f.readlines()
            self.left_paths = []
            self.right_paths = []
            self.gt_paths = []

            for l in lines:
                sl = l.strip()
                self.left_paths.append(sl)
                self.right_paths.append(sl.replace(self.in_left_suffix, self.in_right_suffix))
                self.gt_paths.append(sl.replace(self.in_left_suffix, self.gt_suffix))
            
            # store total number of data
            self.data_size = len(self.left_paths)
            print "total image num in file {} is {}".format(image_list, self.data_size)

    def prefetch(self):
        """
            prefetch all images
            generally dataset for stereo matching contains small number of images
            so prefetch would not consume too much RAM
        """
        self.left_images = []
        self.right_images = []
        self.gt_images = []

        for _ in range(self.data_size):
            # NOTE: read image as grayscale as the origin paper suggested
            left_image = cv2.imread(self.left_paths[_], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            right_image = cv2.imread(self.right_paths[_], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        
            # preprocess images by subtracting the mean and dividing by the standard deviation
            # as the paper described
            left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
            right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))

            self.left_images.append(left_image)
            self.right_images.append(right_image)
            self.gt_images.append(readPfm(self.gt_paths[_]))

        print "prefetch done"

    def shuffle_data(self):
        """
            Random shuffle the images and labels
        """

        left_paths = copy.deepcopy(self.left_paths)
        right_paths = copy.deepcopy(self.right_paths)
        gt_paths = copy.deepcopy(self.gt_paths)
        left_images = copy.deepcopy(self.left_images)
        right_images = copy.deepcopy(self.right_images)
        gt_images = copy.deepcopy(self.gt_images)
        self.left_paths = []
        self.right_paths = []
        self.gt_paths = []
        self.left_images = []
        self.right_images = []
        self.gt_images = []
        
        # create list of permutated index and shuffle data accordingly
        idx = np.random.permutation(self.data_size)
        for i in idx:
            self.left_paths.append(left_paths[i])
            self.right_paths.append(right_paths[i])
            self.gt_paths.append(gt_paths[i])
            self.left_images.append(left_images[i])
            self.right_images.append(right_images[i])
            self.gt_images.append(gt_images[i])
                
    def reset_pointer(self):
        """
            reset pointer to beginning of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
            This function reads the next left, right and gt images, 
            and random pick batch_size patch pairs from these images to 
            construct the nect batch of training data

            NOTE: one batch consists of 1 left image patch, and 2 right image patches,
            which consists of 1 positive sample and 1 negative sample
        """
        # Get next batch of image (path) and labels
        left_path = self.left_paths[self.pointer]
        right_path = self.right_paths[self.pointer]
        gt_path = self.gt_paths[self.pointer]
        
        left_image = self.left_images[self.pointer]
        right_image = self.right_images[self.pointer]
        gt_image = self.gt_images[self.pointer]
        assert left_image.shape == right_image.shape
        assert left_image.shape[0:2] == gt_image.shape
        height, width = left_image.shape[0:2]

        # random choose pixels around which to pick image patchs
        rows = np.random.permutation(height)[0:batch_size]
        cols = np.random.permutation(width)[0:batch_size]

        # rule out those pixels with disparity inf and occlusion
        for _ in range(batch_size):
            while gt_image[rows[_], cols[_]] == float('inf') or \
                  int(gt_image[rows[_], cols[_]]) > cols[_]:
                # random pick another pixel
                rows[_] = random.randint(0, height-1)
                cols[_] = random.randint(0, width-1)

        # augment raw image with zero paddings 
        # this prevents potential indexing error occurring near boundaries
        auged_left_image = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 1], dtype=np.float32)
        auged_right_image = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 1], dtype=np.float32)

        # NOTE: patch size should always be odd
        rows_auged = (self.patch_size[0] - 1)/2
        cols_auged = (self.patch_size[1] - 1)/2
        auged_left_image[rows_auged: rows_auged+height, cols_auged: cols_auged+width, 0] = left_image
        auged_right_image[rows_auged: rows_auged+height, cols_auged: cols_auged+width, 0] = right_image

        # pick patches
        patches_left = np.ndarray([batch_size, self.patch_size[0], self.patch_size[1], 1], dtype=np.float32)
        patches_right_pos = np.ndarray([batch_size, self.patch_size[0], self.patch_size[1], 1], dtype=np.float32)
        patches_right_neg = np.ndarray([batch_size, self.patch_size[0], self.patch_size[1], 1], dtype=np.float32)

        for _ in range(batch_size):
            row = rows[_]
            col = cols[_]

            patches_left[_: _+1] = auged_left_image[row:row + self.patch_size[0], col:col+self.patch_size[1]]

            right_col = col - int(gt_image[row, col])

            # postive example
            # small random deviation added
            pos_col = -1
            while pos_col < 0 or pos_col >= width:
                pos_col = int(right_col + np.random.uniform(-1*self.dataset_pos, self.dataset_pos))
            patches_right_pos[_: _+1] = auged_right_image[row:row+self.patch_size[0], pos_col:pos_col+self.patch_size[1]]

            # negative example
            # large random deviation added
            neg_col = -1
            while neg_col < 0 or neg_col >= width:
                neg_dev = np.random.uniform(self.dataset_neg_low, self.dataset_neg_high)
                if np.random.randint(-1, 1) == -1:
                    neg_dev = -1 * neg_dev
                neg_col = int(right_col + neg_dev)
            patches_right_neg[_: _+1] = auged_right_image[row:row+self.patch_size[0], neg_col:neg_col+self.patch_size[1]]

        #update pointer
        self.pointer += 1
        return patches_left, patches_right_pos, patches_right_neg

    def next_pair(self):
        # Get next images 
        left_path = self.left_paths[self.pointer]
        right_path = self.right_paths[self.pointer]
        gt_path = self.gt_paths[self.pointer]
        
        # Read images
        left_image = self.left_images[self.pointer]
        right_image = self.right_images[self.pointer]
        gt_image = self.gt_images[self.pointer]
        assert left_image.shape == right_image.shape
        assert left_image.shape[0:2] == gt_image.shape

        #update pointer
        self.pointer += 1

        return left_image, right_image, gt_image
    
    def test_mk(self, path):
        if os.path.exists(path):
            return
        else:
            os.mkdir(path)

# just used for debug
if __name__ == "__main__" :
    dg = ImageDataGenerator("/scratch/xz/MC-CNN-python/data/list/train.txt")
    patches_left, patches_right_pos, patches_right_neg = dg.next_batch(128)
    print patches_left.shape
    print patches_right_pos.shape
    print patches_right_neg.shape

