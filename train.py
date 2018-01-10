import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from model import NET
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""
#######################
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

######################
file_prefix = '/home/zxz/stereo_matching'
# init_epoch + 1
FROM_SCRATCH = True
restore_epoch = 0
init_epoch = 0
init_step = 0
# equal to size of receptive field
patch_height = 11
patch_width = 11

######################
# Path to the textfiles for the trainings and validation set
train_file = os.path.join(file_prefix, 'data/trainlQ.in')
val_file = os.path.join(file_prefix, 'data/vallQ.in')

######################
# Learning params
learning_rate = 0.0001
num_epochs = 100000
batch_size = 3
beta1 = 0.9 #momentum for adam

######################
# Network params
train_layers = ['conv1', 'conv2', 'conv3', 'conv4', \
                'conv5']

######################
# How often we want to write the tf.summary data to disk
display_step = 3
save_epoch = 5000

######################
# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = os.path.join(file_prefix, "record/tfrecordQ2")
checkpoint_path = os.path.join(file_prefix, "record/tfrecordQ2")
old_checkpoint_path = os.path.join(file_prefix, "record/old_tfrecordQ2")

# Create parent path if it doesn't exist
if not os.path.isdir(filewriter_path): 
    os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path): 
    os.mkdir(checkpoint_path)
if not os.path.isdir(old_checkpoint_path): 
    os.mkdir(old_checkpoint_path)

# TF placeholder for graph input and output
leftx = tf.placeholder(tf.float32, shape=[batch_size*10, patch_height, patch_width, 3]) # [batch_size*10, sh, sw, 3]
rightx = tf.placeholder(tf.float32, shape=[batch_size*10, patch_height, patch_width, 3]) # [batch_size*10, sh, sw, 3]
y = tf.placeholder(tf.float32, shape=[batch_size*10])                                   # [batch_size*10, 1]

# Initialize model
# note padding differs
left_model = NET(leftx, batch_size=batch_size)
right_model = NET(rightx, batch_size=batch_size)

# Link variable to model output
featuresl = left_model.features
featuresr = right_model.features
assert featuresl.shape == (batch_size*10, 1, 1, 112)
assert featuresr.shape == (batch_size*10, 1, 1, 112)

# List of trainable variables of the layers we want to train
print "variables"
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
for var in var_list:
    print "{} shape: {}".format(var.name, var.shape)

# Op for calculating correlation
with tf.name_scope("correlation"):
    correlation = tf.squeeze(tf.reduce_sum(tf.multiply(featuresl, featuresr), axis=3), axis=[1,2])
    assert correlation.shape == (batch_size*10,)

# Op for calculating the loss
with tf.name_scope("l2_loss"):
    loss =  tf.reduce_mean(tf.squared_difference(correlation, y))

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

with tf.name_scope("training_metric"):
  training_summary = []
  # Add loss to summary
  training_summary.append(tf.summary.scalar('l2_loss', loss))

  # Merge all summaries together
  training_merged_summary = tf.summary.merge(training_summary)

# test loss and error_bias
with tf.name_scope("testing_metric"):
  testing_summary = []
  test_loss = tf.placeholder(tf.float32, [])

  # Add test loss and error_bias to summary
  testing_summary.append(tf.summary.scalar('test_l2_loss', test_loss))
  testing_merged_summary = tf.summary.merge(testing_summary)

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)
# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, shuffle = True)
val_generator = ImageDataGenerator(val_file, shuffle = False) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int32)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int32)

# Start Tensorflow session
with tf.Session(config=tf.ConfigProto(log_device_placement=False, \
        allow_soft_placement=True)) as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  if FROM_SCRATCH:
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
  else:
    saver.restore(sess, os.path.join(old_old_checkpoint_path, 'model_epoch%d.ckpt'%(restore_epoch)))

  print "training_batches_per_epoch: {}, val_batches_per_epoch: {}.".format(\
        train_batches_per_epoch, val_batches_per_epoch)
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  for epoch in range(init_epoch, num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        step = 0
        if epoch == init_epoch:
            step += init_step
        
        while step < train_batches_per_epoch:
            
            print('epoch number: {}. step number: {}'.format(epoch+1, step))

            # Get a batch of images and labels
            if step%2 == 0:
                batch_left, batch_right, batch_y = train_generator.next_batch0(batch_size)
            else:
                batch_left, batch_right, batch_y = train_generator.next_batch1(batch_size)
            
            # And run the training op
            sess.run(train_op, feed_dict={leftx: batch_left,
                                          rightx: batch_right,
                                          y: batch_y})
            
            # Generate summary with the current batch of data and write to file
            if (step+1)%display_step == 0:
                print('{} displaying...'.format(datetime.now()))
                s = sess.run(training_merged_summary, feed_dict={leftx: batch_left,
                                                                 rightx: batch_right,
                                                                 y: batch_y})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)

            step += 1
            
        if (epoch+1)%save_epoch == 0:
            print("{} Saving checkpoint of model...".format(datetime.now()))  
            #save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)  
            os.system("cp {}* {}/".format(checkpoint_name, old_checkpoint_path))
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_ls = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            if _%2==0:
                batch_left, batch_right, batch_y = val_generator.next_batch0(batch_size)
            else:
                batch_left, batch_right, batch_y = val_generator.next_batch1(batch_size)

            result = sess.run([loss], feed_dict={leftx: batch_left,
                                                 rightx: batch_right,
                                                 y: batch_y})
            test_ls += result[0]
            test_count += 1
        test_ls /= test_count
        
        print 'test_ls: {}'.format(test_ls)
        s = sess.run(testing_merged_summary, feed_dict={test_loss: np.float32(test_ls)})
        writer.add_summary(s, train_batches_per_epoch*(epoch + 1))
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        
