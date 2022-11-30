import numpy as np
import torch
from .MaskOptimizer import MaskOptimizer
from .Operator import OperatorNetwork
from .Selector import SelectorNetwork

def progressbar(it, prefix="", size=60):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print("\r%s[%s%s] %i/%i" % (prefix, "#" * x, "." * (size - x), j, count), end=" ")

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print()

class FeatureSelector():
    def __init__(self, data_shape, data_batch_size, mask_batch_size,
                 epoch_on_which_selector_trained=8):
        self.data_shape = data_shape
        self.data_size = np.zeros(data_shape).size
        self.data_batch_size = data_batch_size
        self.mask_batch_size = mask_batch_size
        self.x_batch_size = mask_batch_size * data_batch_size
        self.epoch_on_which_selector_trained = epoch_on_which_selector_trained

    def create_dense_operator(self, arch):
        self.operator = OperatorNetwork(self.data_batch_size, self.mask_batch_size)
        self.operator.create_dense_model(self.data_shape, arch)

    def create_dense_selector(self, arch, init_dropout_rate):
        self.selector = SelectorNetwork(self.mask_batch_size, self.data_shape, init_dropout_rate)
        self.selector.create_dense_model(self.data_shape, arch)

    def create_mask_optimizer(self, epoch_condition=5000):
        self.mopt = MaskOptimizer(self.mask_batch_size, self.data_shape, epoch_condition=epoch_condition)
        self.selector.sample_weights = self.mopt.get_mask_weights(self.epoch_on_which_selector_trained)

    def train_networks_on_data(self, x_tr, y_tr, number_of_batches):

        self.operator.initialize()
        self.selector.initialize()

        x_tr = torch.tensor(x_tr).cuda()
        y_tr = torch.tensor(y_tr).cuda()

        for i in progressbar(range(number_of_batches), "Batch:", 50):
            mopt_condition = self.mopt.check_condiditon()

            random_indices = np.random.randint(0, len(x_tr), self.data_batch_size)
            x = x_tr[random_indices, :]
            y = y_tr[random_indices]
            selector_train_condition = ((self.operator.epoch_counter % self.epoch_on_which_selector_trained) == 0)
            m = self.mopt.get_new_mask_batch(self.selector, gen_new_opt_mask=selector_train_condition)

            self.operator.train_one(x, m, y)
            losses = self.operator.get_per_mask_loss()
            self.selector.append_data(m, losses)
            if (selector_train_condition):
                self.selector.train_one(self.operator.epoch_counter, mopt_condition)

    def get_dropout_logit_p(self):
        return -self.selector.logit_p.data.cpu().numpy()
