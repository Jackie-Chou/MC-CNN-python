"""
network architecture of MC-CNN by tensorflow
""" 
import tensorflow as tf
import numpy as np

# this is the fast architecture of MC-CNN

class NET(object):
  
    def __init__(self, x,  
               weights_path = 'DEFAULT',
               # tunable hyperparameters
               # use suggested values(on Middlebury dataset) of the origin paper as default
               input_patch_size=11, num_conv_layers=5, num_conv_feature_maps=64, 
               conv_kernel_size=3, batch_size = 128):

        self.X = x
        self.batch_size = batch_size
        self.input_patch_size = input_patch_size
        self.num_conv_layers = num_conv_layers
        self.num_conv_feature_maps = num_conv_feature_maps
        self.conv_kernel_size = conv_kernel_size

        if weights_path == 'DEFAULT':      
            self.WEIGHTS_PATH = 'pretrain.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph
        self.create()
    
    def create(self):

        # input size/size of x:
        # [batch_size, h, w, 3] for RGB image
        # [batch_size, h, w, 1] for greyscale image

        # input channels: 3 for RGB while 1 for greyscale
        ic = 3 
        bs = self.batch_size
        k = self.conv_kernel_size
        nf = self.num_conv_feature_maps
        # num of conv layers: at least 2
        nl = self.num_conv_layers

        # use "VALID" padding here(i.e. no zero padding) since the patch size is small(e.g. 11*11) itself, 
        # padded zero may dominant the result
        # in the origin MC-CNN, there's no detail about this(maybe I ignored it), but I strongly recommend using "VALID"

        self.conv1 = conv(self.X, k, k, ic, nf, 1, 1, padding = "VALID", non_linear = "RELU", name = 'conv1')
        print "conv1: {}".format(self.conv1.shape)

        for _ in range(2, nl):
            setattr(self, "conv{}".format(_), conv(getattr(self, "conv{}".format(_-1)), k, k, nf, nf, 1, 1, \
                    padding = "VALID", non_linear = "RELU", name = 'conv{}'.format(_)))
            print "conv{}: {}".format(_, getattr(self, "conv{}".format(_)).shape)

        # last conv without RELU
        setattr(self, "conv{}".format(nl), conv(getattr(self, "conv{}".format(nl-1)), k, k, nf, nf, 1, 1, \
                padding = "VALID", non_linear = "NONE", name = 'conv{}'.format(nl)))
        print "conv{}: {}".format(nl, getattr(self, "conv{}".format(nl)).shape)

        self.flattened = tf.reshape(getattr(self, "conv{}".format(nl)), [bs, -1])
        self.features = tf.nn.l2_normalize(self.flattened, dim=-1, name = "normalize")
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
Predefine all necessary layers
""" 
def conv(x, filter_height, filter_width, input_channels, num_filters, stride_y, stride_x, name,
         padding='SAME', non_linear="RELU", groups=1):

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

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [128, 11, 11, 3])
    net = NET(x)
