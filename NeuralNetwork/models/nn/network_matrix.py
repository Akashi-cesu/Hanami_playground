import torch
import random


class NN(object):

    def __init__(self, sizes: list):
        self.num_layers = len(sizes)  # num of layers
        self.sizes = sizes  # num of neon in layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # setting cuda as device

        layer_input_sizes = self.sizes[:-1]
        layer_output_sizes = self.sizes[1:]

        self.biases = [torch.randn(y, 1, device=self.device, dtype=torch.float64)
                       for y in layer_output_sizes]  # biases dimensions as same as output size
        self.weights = [torch.randn(y, x, device=self.device, dtype=torch.float64)
                        for (y, x) in list(zip(layer_output_sizes, layer_input_sizes))]
        # weights dimensions as same as (y, x) for matrix multi: input(1, n) (n, x) -> output(1, x)

    def sgd(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        n_test = 0
        if test_data:
            n_test = len(test_data)
            # print(f"test len = {n_test}")
        n = len(training_data)
        # print(f"training len is {n}")

        for epoch in range(0, epochs):
            random.shuffle(training_data)  # 打乱训练数据的顺序
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]  # divide training data -> [[mini batch] [mini batch] ...](length==len(training data)/batch size)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # calculate mini batch derivative

            if test_data:
                print(f"Epoch {epoch}: having accuracy as {self.evaluate(test_data)/n_test} "
                      + f"in total {n_test} examples")
            else:
                print(f"{epoch} training completed , total {n_test}")

    def feedforward(self, a):
        a = torch.tensor(a, device=self.device, dtype=torch.float64)
        for (b, w) in list(zip(self.biases, self.weights)):
            z = torch.add(torch.mm(w, a), b)
            a = torch.sigmoid(z)
        return a

    def update_mini_batch(self, mini_batch, eta):
        X = torch.cat(
            [torch.tensor(x, device=self.device, dtype=torch.float64).view(-1, 1)
             for (x, y) in mini_batch], dim=1
        )
        Y = torch.cat(
            [torch.tensor(y, device=self.device, dtype=torch.float64).view(-1, 1)
             for (x, y) in mini_batch], dim=1
        )
        # print(f"X shape is {X.shape}, Y shape is {Y.shape}")

        delta_nabla_b, delta_nabla_w = self.backprop(X, Y)  # calculate derivative of Cost/weight, Cost/bias
        # for b, w in list(zip(delta_nabla_b, delta_nabla_w)):
        #     print(f"this layer nw is {w.shape}, nb is {b.shape}")
        # for b, w in list(zip(self.biases, self.weights)):
        #     print(f"this layer w is {w.shape}, b is {b.shape}")
        # print(f"\n")

        self.weights = [
            torch.sub(w, torch.mul(eta / X.shape[1], nw))
            for w, nw in list(zip(self.weights, delta_nabla_w))
        ]  # w -> w - eta*(avg( delta * a  =  derivative Cost / derivative weight))

        self.biases = [
            torch.sub(b, torch.mul(eta / X.shape[1], nb))
            for b, nb in list(zip(self.biases, delta_nabla_b))
        ]  # b -> b - eta*(avg( delta  =  derivative Cost / derivative biases))

    def backprop(self, x: torch.tensor, y: torch.tensor):
        X = x.clone().detach()  # align X in tensor
        Y = y.clone().detach()  # align Y in tensor
        activations = [X]
        zs = []

        for (b, w) in list(zip(self.biases, self.weights)):
            z = torch.add(torch.mm(w, activations[-1]), b)
            zs.append(z)
            activation = torch.sigmoid(z)
            activations.append(activation)
        # feed forward: a^L = sigmoid(z^L), z^L = w^L dot a^(L-1) + b^L

        # for activation, z in list(zip(activations, zs)):
            # print(f"activation is {activation.shape}, z is {z.shape}")

        delta = torch.mul(
            cost_derivative(activations[-1], Y), sigmoid_prime(zs[-1])
        )  # delta = derivative Cost * sigmoid prime = (a^L - Y) * sigmoid prime
        # final output delta

        nabla_b = [torch.sum(delta, dim=1, keepdim=True) / X.shape[1]]
        # avg(delta) the last biases matrix comment value
        nabla_w = [torch.mm(delta, activations[-2].t()) / X.shape[1]]
        # avg( a^(L-1) * delta ) the last weight matrix comment value

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = torch.mul(
                torch.mm(self.weights[-layer + 1].t(), delta), sp
            )
            nabla_b.insert(0, torch.sum(delta, dim=1, keepdim=True) / X.shape[1])
            nabla_w.insert(0, torch.mm(delta, activations[-layer - 1].t()) / X.shape[1])
        # back prop with BP2\3\4

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [
            (torch.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid_prime(z: torch.tensor):
    prime = torch.mul(
        torch.sigmoid(z),
        torch.sub(1, torch.sigmoid(z))
    )
    return prime


def cost_derivative(output_activations: torch.tensor, y: torch.tensor):
    return torch.sub(output_activations, y)
