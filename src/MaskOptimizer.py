import numpy as np
import torch
from src.ConcreteNeuron import concrete_dropout_neuron

class MaskOptimizer:
    def __init__(self, mask_batch_size, data_shape, epoch_condition=1000):
        self.data_shape = data_shape
        self.data_size = np.zeros(data_shape).size
        self.epoch_counter = 0
        self.mask_batch_size = mask_batch_size
        self.epoch_condition = epoch_condition
        self.max_optimization_iters = 5

    def get_opt_mask(self, selector):

        def sampled_from_logit_p(logit_p, mask_batch_size):
            expanded_logit_p = torch.tile(logit_p, (mask_batch_size, 1))
            dropout_p = torch.sigmoid(expanded_logit_p)
            bern_val = concrete_dropout_neuron(dropout_p)
            return bern_val

        m_opt = sampled_from_logit_p(selector.logit_p, self.mask_batch_size)

        return m_opt

    def check_condiditon(self):
        if (self.epoch_counter >= self.epoch_condition):
            return True
        else:
            return False

    def get_new_mask_batch(self, selector,  gen_new_opt_mask):
        self.epoch_counter += 1
        if (gen_new_opt_mask):
            self.mask_opt = self.get_opt_mask(selector)
        random_masks = self.mask_opt
        return random_masks

    def get_mask_weights(self, tiling):
        w = np.ones(shape=self.mask_batch_size)
        return np.tile(w, tiling)
