"""
    model training of MC-CNN
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from model import NET
from datagenerator import ImageDataGenerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="training of MC-CNN")
parser.add_argument("-g", "--gpu", type=str, default="0", help="gpu id to use, \
                    multiple ids should be separated by commons(e.g. 0,1,2,3)")
parser.add_argument("-ps", "--patch_size", type=int, default=11, help="length for height/width of square patch")
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="mini-batch size")
parser.add_argument("-mr", "--margin", type=float, default=0.2, help="margin in hinge loss")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.002, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument("-bt", "--beta", type=int, default=0.9, help="momentum")
parser.add_argument("--list_dir", type=str, required=True, help="path to dir containing training & validation \
                    left_image_list_file s, should be list_dir/train.txt(val.txt)")
parser.add_argument("--tensorboard_dir", type=str, required=True, help="path to tensorboard dir")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="path to checkpoint saving dir")
parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from. \
                    if None(default), model is initialized using default methods")
parser.add_argument("--start_epoch", type=int, default=0, help="start epoch for training(inclusive)")
parser.add_argument("--end_epoch", type=int, default=14, help="end epoch for training(exclusive)")
parser.add_argument("--print_freq", type=int, default=10, help="summary info(for tensorboard) writing frequency(of batches)")
parser.add_argument("--save_freq", type=int, default=1, help="checkpoint saving freqency(of epoches)")
parser.add_argument("--val_freq", type=int, default=1, help="model validation frequency(of epoches)")

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def main():
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    ######################
    # directory preparation
    filewriter_path = args.tensorboard_dir
    checkpoint_path = args.checkpoint_dir
    
    test_mkdir(filewriter_path)
    test_mkdir(checkpoint_path)

    ######################
    # data preparation
    train_file = os.path.join(args.list_dir, "train.txt")
    val_file = os.path.join(args.list_dir, "val.txt")

    train_generator = ImageDataGenerator(train_file, shuffle = True)
    val_generator = ImageDataGenerator(val_file, shuffle = False) 

    batch_size = args.batch_size
    train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int32)
    val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int32)

    ######################
    # model graph preparation
    patch_height = args.patch_size
    patch_width = args.patch_size
    batch_size = args.batch_size

    # TF placeholder for graph input
    leftx = tf.placeholder(tf.float32, shape=[batch_size, patch_height, patch_width, 3])
    rightx_pos = tf.placeholder(tf.float32, shape=[batch_size, patch_height, patch_width, 3])
    rightx_neg = tf.placeholder(tf.float32, shape=[batch_size, patch_height, patch_width, 3])

    # Initialize model
    left_model = NET(x, input_patch_size=patch_height, batch_size=batch_size)
    right_model_pos = NET(rightx_pos, input_patch_size=patch_height, batch_size=batch_size)
    right_model_neg = NET(rightx_neg, input_patch_size=patch_height, batch_size=batch_size)

    featuresl = left_model.features
    featuresr_pos = right_model_pos.features
    featuresr_neg = right_model_neg.features

    # Op for calculating cosine distance/dot product
    with tf.name_scope("correlation"):
        cosine_pos = tf.reduce_sum(tf.multiply(featuresl, featuresr_pos), axis=-1)
        cosine_neg = tf.reduce_sum(tf.multiply(featuresl, featuresr_neg), axis=-1)

    # Op for calculating the loss
    with tf.name_scope("hinge_loss"):
        margin = tf.ones(shape=[batch_size], dtype=tf.float32) * args.margin
        loss = tf.maximum(0.0, margin - cosine_pos + cosine_neg)
        loss = tf.reduce_mean(loss)

    # Train op
    with tf.name_scope("train"):
        var_list = tf.trainable_variables()
        for var in var_list:
            print "{}: {}".format(var.name, var.shape)
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))
  
        # Create optimizer and apply gradient descent with momentum to the trainable variables
        optimizer = tf.train.MomentumOptimizer(args.learning_rate, args.beta)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # summary Ops for tensorboard visualization
    with tf.name_scope("training_metric"):
        training_summary = []
        # Add loss to summary
        training_summary.append(tf.summary.scalar('hinge_loss', loss))

        # Merge all summaries together
        training_merged_summary = tf.summary.merge(training_summary)

    # validation loss
    with tf.name_scope("val_metric"):
        val_summary = []
        val_loss = tf.placeholder(tf.float32, [])

        # Add val loss to summary
        val_summary.append(tf.summary.scalar('val_hinge_loss', val_loss))
        val_merged_summary = tf.summary.merge(val_summary)

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)
    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    ######################
    # DO training 
    # Start Tensorflow session
    with tf.Session(config=tf.ConfigProto(
                        log_device_placement=False, \
                        allow_soft_placement=True, \
                        gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
     
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # resume from checkpoint or not
        if args.resume is None:
            # Add the model graph to TensorBoard before initial training
            writer.add_graph(sess.graph)
        else:
            saver.restore(sess, args.resume)

        print "training_batches_per_epoch: {}, val_batches_per_epoch: {}.".format(\
                train_batches_per_epoch, val_batches_per_epoch)
        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                        filewriter_path))
      
        # Loop training
        for epoch in range(args.start_epoch, args.end_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            for batch in tqdm(range(train_batches_per_epoch)):
                # Get a batch of data
                batch_left, batch_right_pos, batch_right_neg = train_generator.next_batch(batch_size)
                
                # And run the training op
                sess.run(train_op, feed_dict={leftx: batch_left,
                                              rightx_pos: batch_right_pos,
                                              rightx_neg: batch_right_neg})
                
                # Generate summary with the current batch of data and write to file
                if (batch+1) % args.print_freq == 0:
                    s = sess.run(training_merged_summary, feed_dict={leftx: batch_left,
                                                                     rightx_pos: batch_right_pos,
                                                                     rightx_neg: batch_right_neg})
                    writer.add_summary(s, epoch*train_batches_per_epoch + batch)

                
            if (epoch+1) % args.save_freq == 0:
                print("{} Saving checkpoint of model...".format(datetime.now()))  
                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)  

            if (epoch+1) % args.val_freq == 0:
                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                val_ls = 0.
                for _ in tqdm(range(val_batches_per_epoch)):
                    batch_left, batch_right_pos, batch_right_neg = val_generator.next_batch(batch_size)
                    result = sess.run(loss, feed_dict={leftx: batch_left,
                                                         rightx_pos: batch_right_pos,
                                                         rightx_neg: batch_right_neg})
                    val_ls += result

                val_ls = val_ls / (1. * val_batches_per_epoch)
                
                print 'validation loss: {}'.format(val_ls)
                s = sess.run(val_merged_summary, feed_dict={val_loss: np.float32(val_ls)})
                writer.add_summary(s, train_batches_per_epoch*(epoch + 1))

            # Reset the file pointer of the image data generator
            val_generator.reset_pointer()
            train_generator.reset_pointer()
        
if __name__ == "__main__":
    main()
        
