import torch
from torch.autograd import Variable
import numpy as np
EPSILON = np.finfo(float).eps

def concrete_dropout_neuron(dropout_p, temp=1.0 / 10.0, **kwargs):
    '''
    Use concrete distribution to approximate binary output. Here input is logit(dropout_prob).
    '''
    # Note that p is the dropout probability here
    unif_noise = Variable(dropout_p.data.new().resize_as_(dropout_p.data).uniform_())

    approx = (
        torch.log(dropout_p + EPSILON)
        - torch.log(1. - dropout_p + EPSILON)
        + torch.log(unif_noise + EPSILON)
        - torch.log(1. - unif_noise + EPSILON)
    )
    approx_output = torch.sigmoid(approx / temp)
    return 1 - approx_output

