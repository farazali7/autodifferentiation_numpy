from torch import flatten, nn, no_grad
import torch.nn
from torch.nn import ReLU, CrossEntropyLoss, Linear
from torch.optim import Adam, SGD
from itertools import chain
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self, weights):
        super(Network, self).__init__()
        self.hidden1 = Linear(2, 10)
        self.hidden2 = Linear(10, 10)
        self.output = Linear(10, 1)

        for idx, param in enumerate(self.parameters()):
            p = torch.Tensor(weights[idx])
            # if idx == 3:
            #     p = torch.zeros_like(p)
            param.data = nn.parameter.Parameter(p)

        print('')

    def forward(self, x):
        # x = flatten(x, 1)
        x = torch.Tensor(x)
        x = self.hidden1(x)
        print(x)
        x = ReLU()(x)
        x = self.hidden2(x)
        x = ReLU()(x)
        return self.output(x)

    def one_forward(self, x, y):
        optimizer = SGD(self.parameters(), lr=1./100.)

        optimizer.zero_grad()

        t = [p.data.clone() for p in self.parameters()]

        x = torch.Tensor(x)

        x1 = self.hidden1(x)
        # retain grad
        x1.retain_grad()

        x1L = ReLU()(x1)
        # retain grad
        x1L.retain_grad()

        x2 = self.hidden2(x1L)
        # retain grad
        x2.retain_grad()

        x2L = ReLU()(x2)
        # retain grad
        x2L.retain_grad()

        o = self.output(x2L)
        o.retain_grad()

        # (preds - torch.Tensor(actual)) ** 2 / 2
        loss_sub = o - torch.Tensor(y)
        loss_sub.retain_grad()

        loss_pow = loss_sub ** 2
        loss_pow.retain_grad()

        loss_div = loss_pow / 2
        loss_div.retain_grad()

        # loss = self.compute_loss(o, y)
        # loss.retain_grad()

        mean_loss = torch.mean(loss_div)
        #retain grad
        mean_loss.retain_grad()

        mean_loss.backward()  # Compute and store gradients

        grads = [p.grad for p in self.parameters()]
        p_this = [p.data.clone() for p in self.parameters()]

        optimizer.step()

        all_non_leaf_grads = {'mean_loss': mean_loss.grad,
                              'loss_div': loss_div.grad,
                              'loss_pow': loss_pow.grad,
                              'loss_sub': loss_sub.grad,
                              'output': o.grad,
                              'x2L': x2L.grad,
                              'x2': x2.grad,
                              'x1L': x1L.grad,
                              'x1': x1.grad}

        return grads, all_non_leaf_grads, p_this

    def train(self, train_x, train_y, epochs=1, log_interval=50, optimizer_fn=SGD, learning_rate=0.001):
        optimizer = optimizer_fn(self.parameters(), lr=learning_rate)

        losses = []
        for epoch in range(epochs):
            # Reset gradients, don't accumulate
            optimizer.zero_grad()

            # Forward data through network
            network_output = self.forward(train_x)

            # Calculate loss
            loss = self.compute_loss(network_output, train_y)
            loss = torch.mean(loss)
            losses.append(loss)
            loss.backward()  # Compute and store gradients
            optimizer.step()  # Apply gradients; gradient descent

        # Plot avg loss over epochs
        x_d = range(5 + 1)
        plt.plot(x_d, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.show()


    def compute_loss(self, preds, actual):
        # Regression loss if no. of output neurons, k = 1 in network
        return (preds - torch.Tensor(actual)) ** 2 / 2

    def test(self, test_loader):
        test_loss = 0
        correct = 0
        with no_grad():
            for test_x, test_y in test_loader:
                output = self.forward(test_x)
                test_loss += CrossEntropyLoss()(output, test_y).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(test_y.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)

        print(f"Test set: Avg. loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)}, "
              f"({100.*correct/len(test_loader.dataset)})")
