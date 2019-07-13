from data.data_points import DataPoints
from global_functions.normalise_plots import plot_data, plot_log_data, plot_data_and_model
from data.training_data import TrainingData
from data.tensorflow_data import TensorflowData
from optimization.run_tensorflow_optimization import run_tensorflow_optimization
# What are we trying to model:
# monkey_juice_goodness = monkey_factor*(distillation_temp^monkey_magic)
# G = F*t^M
# linearise take Log of both sides to give Y = mX + f  
# where Y = Log(monkey_juice_goodness), X = Log(distillation_temp), 
# m = monkey_magic, f = log(monkey_factor)

def main():
    data = DataPoints()
    plot_data(data)
    plot_log_data(data)
    training_data = TrainingData(data) 
    tensorflow_data = TensorflowData(training_data)
    trained_model = run_tensorflow_optimization(training_data, tensorflow_data)
    plot_data_and_model(data, trained_model)
		
main()