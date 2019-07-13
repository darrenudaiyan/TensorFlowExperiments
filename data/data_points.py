import numpy as np

class DataPoints:
    number_of_data_points = 150
    np.random.seed(25)
    distillation_temp = np.random.uniform(low=1, high=100,size=(number_of_data_points,))
    monkey_magic = np.random.uniform(low=0.28, high=0.32, size=(number_of_data_points,))
    monkey_factor = np.random.uniform(low=2.5, high=3.5, size=(number_of_data_points,))
    monkey_juice_goodness = monkey_factor*np.power(distillation_temp,monkey_magic)
    Y = np.log(monkey_juice_goodness)
    X = np.log(distillation_temp)