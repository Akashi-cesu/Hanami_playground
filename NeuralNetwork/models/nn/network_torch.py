import torch
import random


class NN(object):

    def __init__(self, sizes: list):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_list = self.sizes[:-1]
        y_list = self.sizes[1:]

        self.biases = [torch.randn(y, 1, device=self.device, dtype=torch.float64)
                       for y in y_list]
        self.weights = [torch.randn(y, x, device=self.device, dtype=torch.float64)
                       for (y, x) in list(zip(y_list, x_list))]

        # for x in self.biases:
        #     print(f" biases shape is {x.shape}")
        # for x in self.weights:
        #     print(f"weights shape is {x.shape}")

    def feedforward(self, a):
        a = torch.tensor(a, device=self.device, dtype=torch.float64)
        for (b, w) in list(zip(self.biases, self.weights)):
            z = torch.add(torch.mm(w, a), b)
            a = torch.sigmoid(z)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

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
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(f"Epoch {epoch}: having accuracy as {self.evaluate(test_data)/n_test} "
                      + f"in total {n_test} examples")
            else:
                print(f"{epoch} training completed , total {n_test}")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [torch.zeros(b.shape, device=self.device, dtype=torch.float64)
                   for b in self.biases]
        nabla_w = [torch.zeros(w.shape, device=self.device, dtype=torch.float64)
                   for w in self.weights]

        for x, y in mini_batch:
            # print(f"in mini batch x is {x}")
            # print(f"in mini batch y is {y}")
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [torch.add(nb, dnb) for nb, dnb in list(zip(nabla_b, delta_nabla_b))]
            nabla_w = [torch.add(nw, dnw) for nw, dnw in list(zip(nabla_w, delta_nabla_w))]

        self.weights = [torch.sub(
            w, torch.mul(
                eta/len(mini_batch), nw
            )
        ) for w, nw in list(zip(self.weights, nabla_w))]
        self.biases = [torch.sub(
            b, torch.mul(
                eta/len(mini_batch), nb
            )
        ) for b, nb in list(zip(self.biases, nabla_b))]

    def backprop(self, x: torch.tensor, y: torch.tensor):

        nabla_b = [torch.zeros(b.shape, device=self.device, dtype=torch.float64)
                   for b in self.biases]
        nabla_w = [torch.zeros(w.shape, device=self.device, dtype=torch.float64)
                   for w in self.weights]

        activation = torch.tensor(x, device=self.device, dtype=torch.float64)
        # print(f" activation shape is {activation.shape}")
        activations = [activation]
        zs = []

        for (b, w) in list(zip(self.biases, self.weights)):
            # print(f"bias shape is {b.shape}")
            # print(f"weight shape is {w.shape}")
            z = torch.add(torch.mm(w, activation), b)
            zs.append(z)
            activation = torch.sigmoid(z)
            activations.append(activation)

        delta = torch.mul(
            self.cost_derivative(activations[-1],
                                 torch.tensor(y, device=self.device, dtype=torch.float64)),
            sigmoid_prime(zs[-1])
        )
        nabla_b[-1] = delta
        nabla_w[-1] = torch.mm(
            delta, activations[-2].t()
        )

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = torch.mul(torch.mm(
                self.weights[-layer + 1].t(),
                delta
            ), sp)
            nabla_b[-layer] = delta
            nabla_w[-layer] = torch.mm(
                delta, activations[-layer - 1].t()
            )

        return nabla_b, nabla_w

    def cost_derivative(self, output_activations: torch.tensor, y:torch.tensor):
        return torch.sub(output_activations, y)

    def evaluate(self, test_data):
        test_results = [
            (torch.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]
        return sum(int (x == y) for (x, y) in test_results)


def sigmoid_prime(z: torch.tensor):
    prime = torch.mul(
        torch.sigmoid(z),
        torch.sub(1, torch.sigmoid(z))
    )
    return prime
