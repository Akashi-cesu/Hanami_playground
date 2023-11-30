import dataLoader
import torch
from models.nn import network_torch
from models.nn import network_matrix


if __name__ == '__main__':

    training_data, validation_data, test_data = dataLoader.load_data_wrapper()
    # 加载训练样本、测试样本、评估样本

    net = network_torch.NN([784, 100, 10])
    # net = network_matrix.NN([784, 100, 10])
    # 神经网络

    net.SGD(training_data, 30, 10, 0.3, test_data=test_data)
    # 梯度下降学习

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # 清除缓存

'''
to do 
rewrite this neural network in python3-standard 
cuda devices instead cpu
'''
