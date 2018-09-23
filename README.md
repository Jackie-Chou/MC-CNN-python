## MC-CNN python implementation
simple implementation of MC-CNN origined from the paper[1] in python

### environment requirements
1. python 2.7
2. tensorflow >= 1.4
3. numpy >= 1.14.0
4. cv2 >= 3.3.0
5. tqdm >= 4.24.0

### file description
- model.py: MC-CNN model class, only the fast architecture in [1] is implemented but I suppose it's not hard to build the accurate one.
- datagenerator.py: training data generator class, used to generate data for model training.
- train.py: training of MC-CNN.
- util.py: some helper functions such as parsing calibration file.
- process_functional.py: processing functions used in stereo matching, such as cross-based cost aggregation.
- match.py: main program of stereo matching, call related procedures from process_functional.py by order and save the results.

### usage
1. train the MC-CNN model, this is pretty quick on my Nvidia 1080Ti GPU.
for details type
> python train.py -h

2. use trained model and do stereo matching, this is time-consuming.
for details type
> python match.py -h

### NOTE
- this code should only serve as a demo
- the running time of the whole program could be very long, python is obviously slow for such computationally intensive program, thus in their original paper [1], the authors used torch & cuda mainly. I comment out many intermediate procedures of the whole program in match.py, for those who wanna have a feeling of suffering or keep your cpu busy, please delete those comments manually :).
- I haved tested the code on [Middlebury stereo dataset(version 3)](http://vision.middlebury.edu/stereo/submit3/), using the half resolution data. It's supposed the code can be used seamlessly on any other dataset with some details taken care of.
..* in datagenerator.py, the default suffix arguments of __init__ method of ImageDataGenerator class is set with respect to Middlebury dataset, so it should be specified according to the dataset your are using.
..* in match.py, there are also some global suffix and filename variables set with respect to Middlebury, so the same changes should be made beforehand.
..* the hierarchy of the data directory should be consistent with mine. Please check the data directory for details. I only keep the directory hierarchy without any data since it's a bit large.

### License
MIT license.

[1] Jure Zbontar, Yann LeCuny. *Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches*
