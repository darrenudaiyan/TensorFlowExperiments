import tensorflow as tf  # Needs 64 bit python
from data.trained_model import TrainedModel

def run_tensorflow_optimization(TrainingData, TensorflowData):
    init = tf.global_variables_initializer() #initialise global variables
    sess = tf.Session()
    sess.run(init)

    num_training_steps = 50
    for iteration in range(num_training_steps):
        for (x, y) in zip(TrainingData.train_X_norm, TrainingData.train_Y_norm):
            sess.run(TensorflowData.optimizer, feed_dict={TensorflowData.tf_X: x, TensorflowData.tf_Y: y})
            current_loss = sess.run(TensorflowData.tf_cost, feed_dict={TensorflowData.tf_X: TrainingData.train_X_norm, TensorflowData.tf_Y: TrainingData.train_Y_norm})
            print("Current loss=", current_loss, "monkey_magic=", sess.run(TensorflowData.tf_m), "monkey_factor_log=", sess.run(TensorflowData.tf_f), '\n')

    print("Finished Optimising")
    training_loss = sess.run(TensorflowData.tf_cost, feed_dict={TensorflowData.tf_X:TrainingData.train_X_norm, TensorflowData.tf_Y:TrainingData.train_Y_norm})
    monkey_magic = sess.run(TensorflowData.tf_m)
    monkey_factor_log = sess.run(TensorflowData.tf_f)
    print("Trained cost=", training_loss, "monkey_magic=", monkey_magic, "monkey_scale_log=", monkey_factor_log, '\n')

    sess.close()

    trained_model = TrainedModel(monkey_magic, monkey_factor_log)
    return trained_model