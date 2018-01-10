import tensorflow as tf
import numpy as np

class NET(object):
  
  def __init__(self, x,  
               weights_path = 'DEFAULT', batch_size = 1):
    
    # Parse input arguments into class variables
    self.X = x
    self.batch_size = batch_size
    
    if weights_path == 'DEFAULT':      
      self.WEIGHTS_PATH = 'pretrain.npy'
    else:
      self.WEIGHTS_PATH = weights_path
    
    # Call the create function to build the computational graph of AlexNet
    self.create()
    
  def create(self):
    #input size [batch_size, h, w, 3]
    self.conv1 = conv(self.X, 3, 3, 3, 112, 1, 1, padding = "VALID", name = 'conv1') # [~, h, w, 112]
    print "conv1: {}".format(self.conv1.shape)
    self.conv2 = conv(self.conv1, 3, 3, 112, 112, 1, 1, padding = "VALID", name = 'conv2') # [~, h, w, 112]
    print "conv2: {}".format(self.conv2.shape)
    self.conv3 = conv(self.conv2, 3, 3, 112, 112, 1, 1, padding = "VALID", name = 'conv3') # [~, h, w, 112]
    print "conv3: {}".format(self.conv3.shape)
    self.conv4 = conv(self.conv3, 3, 3, 112, 112, 1, 1, padding = "VALID", name = 'conv4') # [~, h, w, 112]
    print "conv4: {}".format(self.conv4.shape)
    self.conv5 = conv(self.conv4, 3, 3, 112, 112, 1, 1, padding = "VALID", name = 'conv5') # [~, h, w, 112]
    print "conv5: {}".format(self.conv5.shape)
    # receptive field size is 11 * 11
    self.features = tf.nn.l2_normalize(self.conv5, dim=-1)
    print "features: {}".format(self.features.shape)

  def load_initial_weights(self, session):
    all_vars = tf.trainable_variables()
    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
    for name in weights_dict:
      print "restoring var {}...".format(name)
      var = [var for var in all_vars if var.name == name][0]
      session.run(var.assign(weights_dict[name]))
     
  def save_weights(self, session, file_name='pretrain.npy'):
    save_vars = tf.trainable_variables()
    weights_dict = {}
    for var in save_vars:
      weights_dict[var.name] = session.run(var)
    np.save('pretrain.npy', weights_dict) 
    print "weights saved in file {}".format(file_name)
  
"""
Predefine all necessary layer for the AlexNet
""" 
def conv(x, filter_height, filter_width, input_channels, num_filters, stride_y, stride_x, name,
         padding='SAME', non_linear="RELU", groups=1):
  """
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    
    if groups == 1:
      conv = convolve(x, weights)
      
    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      
      # Concat the convolved output together again
      conv = tf.concat(axis = 3, values = output_groups)
      
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply non_linear function
    if non_linear == "RELU":
      non_lin = tf.nn.relu(bias, name = scope.name)
    elif non_linear == "NONE":
      non_lin = tf.identity(bias, name = scope.name)

    return non_lin
  
def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)
  
def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
  
def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)
  
    
