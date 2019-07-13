import tensorflow as tf  # Needs 64 bit python
import numpy as np

class TensorflowData:
    def __init__(self, Training_Data):
        self.tf_X = tf.placeholder("float", name="X_temp_log")  # tensorflow placeoholders get updated as we go down the gradient
        self.tf_Y = tf.placeholder("float", name="Y_Goodness_log")
        # Define the tensorflow variables we update during training, remember these are logs  
        # We initialize them to some random values based on the normal distribution.
        self.tf_m = tf.Variable(np.random.randn(), name="monkey_magic")
        self.tf_f = tf.Variable(np.random.randn(), name="monkey_factor_log")
        #Y = mX + f 
        self.tf_Y_pred = tf.add(tf.multiply(self.tf_m, self.tf_X), self.tf_f)  

        #Define the Loss Function (how much error) - Mean squared error
        self.tf_cost = tf.reduce_sum(tf.pow(self.tf_Y_pred-self.tf_Y, 2))/(2*Training_Data.number_of_training_points)
        # Optimizer learning rate.  The size of the steps down the gradient
        self.learning_rate = 0.02
        # use Gradient descent optimizer that will minimize the loss 
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.tf_cost)