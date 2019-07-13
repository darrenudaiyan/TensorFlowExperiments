import matplotlib.pyplot as plt
import numpy as np

# Normalize values to prevent under/overflows. Make sure x and y on similar scales
def normalize(array):
    return (array - array.mean()) / array.std()

def denormalize(array):
    return (array * array.std()) + array.mean()
   
def plot_data(Data_Model):
    plt.plot(Data_Model.distillation_temp,Data_Model.monkey_juice_goodness,"bx")
    plt.ylabel = "Monkey Goodness"
    plt.xlabel = "Distillation temperature"
    plt.show()

def plot_log_data(Data_Model):
    plt.plot(Data_Model.X,Data_Model.Y,"bx")
    plt.ylabel = "Log Monkey Goodness"
    plt.xlabel = "Log Distillation temperature"
    plt.show()

def plot_data_and_model(Data_Model, TrainedModel):
    plt.plot(Data_Model.distillation_temp,Data_Model.monkey_juice_goodness,"bx")
    temp = np.arange(0, 100, 10)
    plt.plot(temp,TrainedModel.monkey_factor*(np.power(temp,TrainedModel.monkey_magic)))
   
    plt.ylabel = "Monkey Goodness"
    plt.xlabel = "Distillation temperature"

    plt.show()