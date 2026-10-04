import torch
from typing import Callable
from spikingjelly.activation_based import neuron, surrogate




def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)



class LIFNode(neuron.BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1., decay_input: bool = False,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.decay_input = decay_input

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            self.v = self.v + (x - self.v) / self.tau
        else:
            self.v = self.v * (1. - 1. / self.tau) + x

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)



class DLIFNode(neuron.BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1., decay_input: bool = False, D_threshold: float = -1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(), store_D_seq: bool = False,
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.decay_input = decay_input
        self.register_memory('D', 1.)
        if store_D_seq:
            self.register_memory('D_seq', None)
        self.D = 1.
        self.D_threshold = D_threshold
        self.store_D_seq = store_D_seq

    def D_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.D, float):
            D_init = self.D
            self.D = torch.full_like(x.data, D_init)

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            self.v = self.v + (x - self.v) / self.tau
        else:
            self.v = self.v * (1. - 1. / self.tau) + x

    def neuronal_fire(self):
        return self.D * self.surrogate_function(self.v - self.v_threshold)
    
    def D_change(self):
        self.D = self.D * self.surrogate_function(self.v - self.D_threshold)

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.D_float_to_tensor(x)
        self.D_change()
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        if self.store_D_seq:
            D_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
            if self.store_D_seq:
                D_seq.append(self.D)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_D_seq:
            self.D_seq = torch.stack(D_seq)

        return torch.stack(y_seq)