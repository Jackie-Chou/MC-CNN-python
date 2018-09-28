## MC-CNN python implementation
simple implementation of MC-CNN origined from the paper[1] in python

### environment requirements
1. python 2.7
2. tensorflow >= 1.4
3. numpy >= 1.14.0
4. cv2 >= 3.3.0
5. tqdm >= 4.24.0

### file description
- model.py: MC-CNN model class. Only the fast architecture in [1] is implemented but I suppose it's not hard to build the accurate one.
- datagenerator.py: training data generator class. This is used to generate data for model training.
- train.py: training of MC-CNN.
- util.py: some helper functions such as parsing calibration file.
- process_functional.py: processing functions used in stereo matching such as cross-based cost aggregation.
- match.py: main program of stereo matching. Call relevant procedures from process_functional.py by order and save the final results as [Middlebury stereo dataset(version 3)](http://vision.middlebury.edu/stereo/submit3/) required.

### usage
1. train the MC-CNN model. This is pretty quick on my Nvidia 1080Ti GPU for 2000 epochs.
for details type
> python train.py -h

2. use trained model and do stereo matching. This is time-consuming.
for details type
> python match.py -h

### NOTE
- this code should only serve as a demo
- the running time of the whole program could be very long. Python is obviously slow for such computationally intensive program, thus in their original paper [1] the authors used torch & cuda mainly. I keep all the processing procedures mentioned in the paper, but feel free to skip some of them like semiglobal matching if you want--just comment out relevant snippets in match.py.
- I haved tested the code on [Middlebury stereo dataset(version 3)](http://vision.middlebury.edu/stereo/submit3/) using the half resolution data. It's supposed the code can be used seamlessly on any other dataset with some details taken care of, especially about data format.
- All the hyperparameters are set to the suggested value from the origin paper [1] and I do not do further finetuning.
- In my implementation, some processing details may be a little bit different from what the paper describes and I suppose it would not harm the performance too much. You can find those differences from the comments.
- The final result on Middlebury dataset, I have to admit, is not that impressive. This may be due to lack of further hyperparemeter finetuning or some unnoticed bugs. If you find any bug during using, please report it in an issue and I will try to fix it ASAP. 

### License
MIT license.

### Reference
[1] Jure Zbontar, Yann LeCuny. *Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches*
