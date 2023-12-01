import time

import dataLoader
import torch
from models.nn import network_torch
from models.nn import network_matrix


if __name__ == '__main__':

    training_data, validation_data, test_data = dataLoader.load_data_wrapper()
    # 加载训练样本、测试样本、评估样本

    time_start = time.time()
    net = network_torch.NN([784, 100, 10])
    # 神经网络
    net.sgd(training_data, 30, 10, 0.3, test_data=test_data)
    # 梯度下降学习
    end_time = time.time()
    print(f"cost time is {end_time - time_start:.2f}")

    time_start = time.time()
    net_m = network_matrix.NN([784, 100, 10])
    net_m.sgd(training_data, 30, 10, 0.3, test_data=test_data)
    end_time = time.time()
    print(f"cost time is {end_time - time_start:.2f}")

    time_start = time.time()
    net_m = network_matrix.NN([784, 100, 10])
    net_m.sgd(training_data, 100, 10, 0.3, test_data=test_data)
    end_time = time.time()
    print(f"cost time is {end_time - time_start:.2f}")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # 清除缓存
