import torch
import torch.nn as nn
import torch.optim as optim
class DenseModel(nn.Module):
    def chooseActivation(self, act_str):
        if act_str == 'relu':
            return nn.ReLU()
        elif act_str == 'sigmoid':
            return nn.Sigmoid()
        elif act_str == 'tanh':
            return nn.Tanh()
        elif act_str == 'linear':
            return nn.Identity()
        else:
            return nn.ReLU()

    def __init__(self, hidden_layers=1, neurons_per_layer=1, activation_hidden='relu', activation_output='linear'):
        print(f"DenseModel: hidden_layers:{hidden_layers}, neurons_per_layer:{neurons_per_layer}, "
              f"activation_hidden:{activation_hidden}, activation_output:{activation_output}")
        super().__init__()  # Constructor of parent class
        self.n_hidden_layers = hidden_layers  # 2, 20
        self.input_layer = nn.Linear(1, neurons_per_layer)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(neurons_per_layer, neurons_per_layer) for x in range(self.n_hidden_layers)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(neurons_per_layer) for x in range(self.n_hidden_layers)])
        self.act_hidden = self.chooseActivation(activation_hidden)
        self.act_output = self.chooseActivation(activation_output)
        self.output_layer = nn.Linear(neurons_per_layer, 1)

    # On the forward function we indicate how to make one 'pass' of the model
    def forward(self, x):
        l1 = self.input_layer(x)  # With simple non-linear function
        for i in range(self.n_hidden_layers):
            l1 = self.act_hidden(self.hidden_layers[i](l1))  # With batch normalization
            l1 = self.bns[i](l1)
        l2 = self.act_output(self.output_layer(l1))
        return l2

