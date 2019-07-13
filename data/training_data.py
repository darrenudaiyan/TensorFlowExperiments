import numpy as np
import math
from global_functions.normalise_plots import normalize

class TrainingData:
    def __init__(self, Data_Model):
        self.number_of_training_points = math.floor(Data_Model.number_of_data_points * 0.7)
        self.train_distillation_temp = np.asanyarray(Data_Model.distillation_temp[:self.number_of_training_points:])
        self.train_monkey_juice_goodness = np.asanyarray(Data_Model.monkey_juice_goodness[:self.number_of_training_points:])
        self.train_X = np.log(self.train_distillation_temp)
        self.train_Y = np.log(self.train_monkey_juice_goodness)
        self.train_X_norm = normalize(self.train_X)
        self.train_Y_norm = normalize(self.train_Y)