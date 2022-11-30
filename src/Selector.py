import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from src.ConcreteNeuron import concrete_dropout_neuron
import torch.optim as optim
from torch.autograd import Variable

def sigmoid(x):
    return 1/(1+np.exp(-x))

class SelectorNetwork:
    def __init__(self, mask_batch_size, data_shape, init_dropout_rate):
        self.batch_size = mask_batch_size
        self.mask_batch_size = mask_batch_size
        self.tr_loss_history = []
        self.epoch_counter = 0
        self.data_masks = None
        self.data_targets = None
        self.sample_weights = None

        self.data_shape = data_shape
        init_arg = np.log(init_dropout_rate) - np.log(1. - init_dropout_rate)
        self.logit_p = Parameter(torch.Tensor(self.data_shape[0]).cuda())
        self.logit_p.requires_grad = True
        self.logit_p.data.fill_(init_arg)

        self.predict_loss = None
        self.regur_loss = None

        self.lr = 0.001

    def create_dense_model(self, input_shape, dense_arch):

        self.model = nn.ModuleList([nn.Linear(input_shape[0], dense_arch[0])])
        self.model.append(nn.ReLU())
        self.model.append(nn.BatchNorm1d(dense_arch[0]))
        self.model.append(nn.Dropout(0.5))

        for i in range(len(dense_arch) - 2):
            self.model.append(nn.Linear(dense_arch[i], dense_arch[i+1]))
            self.model.append(nn.ReLU())
            self.model.append(nn.BatchNorm1d(dense_arch[i+1]))
            self.model.append(nn.Dropout(0.5))

        self.model.append(nn.Linear(dense_arch[-2], dense_arch[-1]))

        self.model.cuda()

    def sampled_from_logit_p(self, num_samples):
        expanded_logit_p = self.logit_p.unsqueeze(0).expand(num_samples, *self.logit_p.size())
        dropout_p = torch.sigmoid(expanded_logit_p)
        bern_val = concrete_dropout_neuron(dropout_p)
        return bern_val

    def forward(self, x):

        ber_val = self.sampled_from_logit_p(x.shape[0])
        x = x * ber_val

        for layer in self.model:
            x = layer(x)
        return x

    def loss(self, outputs, targets, apply_weights):

        # outputs: output of the selector
        # targets: the averaged learning performance from the operator

        if apply_weights == False:
            predict_loss = torch.mean(torch.abs(outputs - targets.reshape(-1, 1)))
        else:
            predict_loss = torch.mean(torch.abs(outputs - targets.reshape(-1, 1)) * torch.tensor(self.sample_weights).cuda())

        dropout_p = torch.sigmoid(self.logit_p)
        regur_loss = torch.mean(dropout_p)

        loss = predict_loss / regur_loss

        self.predict_loss = predict_loss
        self.regur_loss = regur_loss

        return loss

    def my_parameters(self):
        return [{'params': self.logit_p},
                {'params': self.model.parameters()}]

    def initialize(self):
        self.optimizer = optim.Adam(self.my_parameters(), lr=self.lr)

    def train_one(self, epoch_number, apply_weights):

        inputs = Variable(self.data_masks)
        targets = Variable(self.data_targets)
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        curr_loss = self.loss(outputs, targets, apply_weights)
        curr_loss.backward()
        self.optimizer.step()

        if epoch_number % 500 == 1:
            print('Epoch:[{}]: selector training loss={:.5f}, (predict loss={:.5f}, regularization loss={:.5f})'.format(epoch_number, curr_loss.data.cpu().numpy(), self.predict_loss, self.regur_loss))

        self.tr_loss_history.append(curr_loss)
        self.epoch_counter = epoch_number
        self.data_masks = None
        self.data_targets = None

    def append_data(self, x, y):
        if self.data_masks is None:
            self.data_masks = x
            self.data_targets = y
        else:
            self.data_masks = torch.concat((self.data_masks, x), axis=0)
            self.data_targets = torch.concat((self.data_targets, y), axis=0)