import numpy as np

class TrainedModel:
     def __init__(self, monkey_magic,monkey_factor_log):
         self.monkey_magic = monkey_magic
         self.monkey_factor = np.exp(monkey_factor_log)