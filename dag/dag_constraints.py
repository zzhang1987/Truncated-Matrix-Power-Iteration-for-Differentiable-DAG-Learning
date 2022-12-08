import numpy as np
import scipy.linalg as slin
import torch

from .other_constraints import h_binomial, h_exponential, h_binomial_torch, h_exponential_torch
from .tmpi import h_fast_tmpi, HTorchTmpi



def h_torch(B, h_type='fast_tmpi', eps=1e-6):
    if h_type == 'fast_tmpi':
        _h_impl = lambda x: HTorchTmpi.apply(B, eps)
    elif h_type == 'exponential':
        _h_impl = h_exponential_torch
    elif h_type == 'binomial':
        _h_impl = h_binomial_torch
    else:
        raise ValueError('unknown h type')
    return _h_impl(B)


def h_numpy(B, h_type='fast_tmpi', eps=1e-6):
    if h_type == 'exponential':
        _h_impl = h_exponential
    elif h_type == 'binomial':
        _h_impl = h_binomial
    elif h_type == 'fast_tmpi':
        _h_impl = lambda x: h_fast_tmpi(x, eps=eps)
    else:
        raise ValueError('unknown h type')
    return _h_impl(B)

