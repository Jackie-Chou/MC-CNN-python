import os
import numpy as np
import cv2
import copy
from util import readPfm
import random
from tensorflow import expand_dims

class ImageDataGenerator:
    def __init__(self, left_image_list, shuffle=False, 
                 patch_size=(11, 11),
                 # NOTE: left and right can be exchanged
                 in_left_prefix='',
                 in_left_suffix='im0.png',
                 in_right_prefix='',
                 in_right_suffix='im1.png',
                 gt_prefix='',
                 gt_suffix='disp0GT.pfm'):
                
        # Init params
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.in_left_prefix = in_left_prefix
        self.in_left_suffix = in_left_suffix
        self.in_right_prefix = in_right_prefix
        self.in_right_suffix = in_right_suffix
        self.gt_prefix = gt_prefix
        self.gt_suffix = gt_suffix
        self.pointer = 0
        self.read_image_list(left_image_list)
        self.prefetch()
        if self.shuffle:
            self.shuffle_data()

    def read_image_list(self, image_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(image_list) as f:
            lines = f.readlines()
            self.left_paths = []
            self.right_paths = []
            self.gt_paths = []
            for l in lines:
                sl = l.strip()
                self.left_paths.append(sl)
                self.right_paths.append(sl.replace(self.in_left_prefix, self.in_right_prefix).\
                                           replace(self.in_left_suffix, self.in_right_suffix))
                self.gt_paths.append(sl.replace(self.in_left_prefix, self.gt_prefix).\
                                        replace(self.in_left_suffix, self.gt_suffix))
            
            #store total number of data
            self.data_size = len(self.left_paths)
            print "total image in file {} is {}".format(image_list, self.data_size)

    # prefetch data since reading pfm is time-consuming
    def prefetch(self):
        self.left_images = []
        self.right_images = []
        self.gt_images = []
        for _ in range(self.data_size):
            self.left_images.append(cv2.imread(self.left_paths[_]).astype(np.float32))
            self.right_images.append(cv2.imread(self.right_paths[_]).astype(np.float32))
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
        
        #create list of permutated index and shuffle data accoding to list
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
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch0(self, batch_size):
        """
        This function reads the next left, right and gt images, 
        and random pick batch_size patches from these images to 
        construct the nect batch of training data
        NOTE: one batch consists of one left image patch, and **10** right image patches,
        which comprises 5 positive samples and 5 negative samples
        """
        # Get next batch of image (path) and labels
        left_path = self.left_paths[self.pointer]
        right_path = self.right_paths[self.pointer]
        gt_path = self.gt_paths[self.pointer]
        
        # Read images
        # BGR
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
                  int(gt_image[rows[_], cols[_]]) > cols[_] - 2:
                # random pick another pixel
                rows[_] = random.randint(0, height-1)
                cols[_] = random.randint(0, width-1)

        # augment raw image with zero paddings 
        auged_left_image = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 3], dtype=np.float32)
        auged_right_image = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 3], dtype=np.float32)
        rows_auged = (self.patch_size[0] - 1)/2
        cols_auged = (self.patch_size[1] - 1)/2
        auged_left_image[rows_auged: rows_auged+height, cols_auged: cols_auged+width] = left_image
        auged_right_image[rows_auged: rows_auged+height, cols_auged: cols_auged+width] = right_image

        # pick patches
        patches_left = np.ndarray([batch_size*10, self.patch_size[0], self.patch_size[1], 3], dtype=np.float32)
        # NOTE: patches no overlap
        patches_right = np.ndarray([batch_size*10, self.patch_size[0], self.patch_size[1], 3], dtype=np.float32)
        labels = np.zeros([batch_size*10], dtype=np.float32)
        for _ in range(batch_size):
            # assume every pixel of gt images represents the disparity(only horizontally)
            # the positive disparity means the pixel in right image is to the left of that in the left image
            patches_left[_*10: (_+1)*10] = auged_left_image[rows[_]:rows[_]+self.patch_size[0], cols[_]:cols[_]+self.patch_size[1]]
            right_col = cols[_] - int(gt_image[rows[_], cols[_]])
            assert right_col >= 2
            patches_right[_*10] = auged_right_image[rows[_]:rows[_]+self.patch_size[0], right_col:right_col+self.patch_size[1]]
            patches_right[_*10+1] = auged_right_image[rows[_]:rows[_]+self.patch_size[0], right_col-1:right_col+self.patch_size[1]-1]
            patches_right[_*10+2] = auged_right_image[rows[_]:rows[_]+self.patch_size[0], right_col+1:right_col+self.patch_size[1]+1]
            patches_right[_*10+3] = auged_right_image[rows[_]:rows[_]+self.patch_size[0], right_col-2:right_col+self.patch_size[1]-2]
            patches_right[_*10+4] = auged_right_image[rows[_]:rows[_]+self.patch_size[0], right_col+2:right_col+self.patch_size[1]+2]
            labels[_*10] = 1.0
            labels[_*10+1] = 0.8
            labels[_*10+2] = 0.8
            labels[_*10+3] = 0.6
            labels[_*10+4] = 0.6

            # negative cols
            for i in range(5, 10):
                neg_col = random.randint(0, width-1)
                while abs(neg_col - right_col) <= 10:
                    neg_col = random.randint(0, width-1)
                patches_right[_*10+i] = auged_right_image[rows[_]:rows[_]+self.patch_size[0], neg_col:neg_col+self.patch_size[1]]

        #update pointer
        self.pointer += 1
        return patches_left, patches_right, labels

    def next_batch1(self, batch_size):
        """
        This function reads the next left, right and gt images, 
        and random pick batch_size patches from these images to 
        construct the nect batch of training data
        NOTE: one batch consists of one right image patch, and **10** left image patches,
        which comprises 5 positive samples and 5 negative samples
        """
        # Get next batch of image (path) and labels
        left_path = self.left_paths[self.pointer]
        right_path = self.right_paths[self.pointer]
        gt_path = self.gt_paths[self.pointer]
        
        # Read images
        # BGR
        left_image = self.left_images[self.pointer]
        right_image = self.right_images[self.pointer]
        gt_image = self.gt_images[self.pointer]
        assert left_image.shape == right_image.shape
        assert left_image.shape[0:2] == gt_image.shape
        height, width = left_image.shape[0:2]

        # random choose pixels around which to pick image patchs
        # NOTE: here rows and cols still refer to left images since 
        # gt is subject to left images
        rows = np.random.permutation(height)[0:batch_size]
        cols = np.random.permutation(width)[0:batch_size]
        # rule out those pixels with disparity inf and occlusion
        for _ in range(batch_size):
            while gt_image[rows[_], cols[_]] == float('inf') or \
                  cols[_] - int(gt_image[rows[_], cols[_]]) < 0 or \
                  cols[_] < 2 or \
                  cols[_] + 2 >= width:
                # random pick another pixel
                rows[_] = random.randint(0, height-1)
                cols[_] = random.randint(0, width-1)

        rrows = np.zeros([batch_size], dtype=np.int32)
        rcols = np.zeros([batch_size], dtype=np.int32)
        for _ in range(batch_size):
            rrows[_] = rows[_]
            rcols[_] = cols[_] - int(gt_image[rows[_], cols[_]])

        # augment raw image with zero paddings 
        auged_left_image = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 3], dtype=np.float32)
        auged_right_image = np.zeros([height+self.patch_size[0]-1, width+self.patch_size[1]-1, 3], dtype=np.float32)
        rows_auged = (self.patch_size[0] - 1)/2
        cols_auged = (self.patch_size[1] - 1)/2
        auged_left_image[rows_auged: rows_auged+height, cols_auged: cols_auged+width] = left_image
        auged_right_image[rows_auged: rows_auged+height, cols_auged: cols_auged+width] = right_image

        # pick patches
        patches_left = np.ndarray([batch_size*10, self.patch_size[0], self.patch_size[1], 3], dtype=np.float32)
        # NOTE: patches no overlap
        patches_right = np.ndarray([batch_size*10, self.patch_size[0], self.patch_size[1], 3], dtype=np.float32)
        labels = np.zeros([batch_size*10], dtype=np.float32)
        for _ in range(batch_size):
            # assume every pixel of gt images represents the disparity(only horizontally)
            # the positive disparity means the pixel in right image is to the left of that in the left image
            patches_right[_*10: (_+1)*10] = auged_right_image[rrows[_]:rrows[_]+self.patch_size[0], rcols[_]:rcols[_]+self.patch_size[1]]

            patches_left[_*10] = auged_left_image[rows[_]:rows[_]+self.patch_size[0], cols[_]:cols[_]+self.patch_size[1]]
            patches_left[_*10+1] = auged_left_image[rows[_]:rows[_]+self.patch_size[0], cols[_]-1:cols[_]+self.patch_size[1]-1]
            patches_left[_*10+2] = auged_left_image[rows[_]:rows[_]+self.patch_size[0], cols[_]+1:cols[_]+self.patch_size[1]+1]
            patches_left[_*10+3] = auged_left_image[rows[_]:rows[_]+self.patch_size[0], cols[_]-2:cols[_]+self.patch_size[1]-2]
            patches_left[_*10+4] = auged_left_image[rows[_]:rows[_]+self.patch_size[0], cols[_]+2:cols[_]+self.patch_size[1]+2]
            labels[_*10] = 1.0
            labels[_*10+1] = 0.8
            labels[_*10+2] = 0.8
            labels[_*10+3] = 0.6
            labels[_*10+4] = 0.6

            # negative cols
            for i in range(5, 10):
                neg_col = random.randint(0, width-1)
                while abs(neg_col - cols[_]) <= 10:
                    neg_col = random.randint(0, width-1)
                patches_left[_*10+i] = auged_left_image[rows[_]:rows[_]+self.patch_size[0], neg_col:neg_col+self.patch_size[1]]

        #update pointer
        self.pointer += 1
        return patches_left, patches_right, labels

    def next_pair(self):
        # Get next images 
        left_path = self.left_paths[self.pointer]
        right_path = self.right_paths[self.pointer]
        gt_path = self.gt_paths[self.pointer]
        
        # Read images
        # BGR
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

    def store_mask(self, masks):
        # mask: [batch, 224, 224, 1]
        self.last_images = self.last_images + self.mean
        self.masks = masks * 255.
        for i in range(len(self.last_paths)):
            items = self.last_paths[i].split('/')
            subDirName = items[-4]
            subsubDirName = items[-3]
            imageDirName = items[-2]

            subDirPath = os.path.join(self.mask_store_prefix, subDirName)
            subsubDirPath = os.path.join(subDirPath, subsubDirName)
            imageDirPath = os.path.join(subsubDirPath, imageDirName)
            
            self.test_mk(subDirPath)
            self.test_mk(subsubDirPath)
            self.test_mk(imageDirPath)
    
            mean_pixel = np.mean(self.masks[i])
            ret, thresh = cv2.threshold(self.masks[i], mean_pixel, 255., cv2.THRESH_BINARY)
            # green channel
            self.last_images[i, :, :, 1] = self.last_images[i, :, :, 1] + thresh
            ret, thresh = cv2.threshold(self.last_images[i, :, :, 1], 255., 225., cv2.THRESH_TRUNC)
            self.last_images[i, :, :, 1] = thresh

            cv2.imwrite(os.path.join(imageDirPath, items[-1]), self.last_images[i])
