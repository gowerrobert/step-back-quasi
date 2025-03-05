"""
Implements the MoMo algorithm.

Authors: Fabian Schaipp, Ruben Ohana, Michael Eickenberg, Aaron Defazio, Robert Gower
"""
import torch
import warnings
from math import sqrt

from ..types import Params, LossClosure, OptFloat

class SAG(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-1,
                 weight_decay: float=0) -> None:
        """
        SAG (Stochastic Average Gradient) optimizer

        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate, by default 1e-1.
        weight_decay : float, optional
            Weight decay parameter, by default 0.
        """
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))

        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SAG, self).__init__(params, defaults)
        # Initialization
        self._number_steps = 0
        self.state['step_size_list'] = list() # for storing the adaptive step size term
        
        return
        
    def prestep(self, out, targets, ind, loss_name):
        """
        out : model output (before loss)
        targets: for computing the loss (e.g. image classes)
        ind: indices of the batch members (from 1,...,n_train_samples)
        loss_name: a string that indicates which loss is used. See `step-back/metrics.py`.
        """
        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
  
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data           
                state = self.state[p]
                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradients
                    state['grad_table'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of squared gradient values
                    state['grad_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of inner product <grad, weight>
                    state['grad_dot_w'] = torch.tensor(0.).to(p.device)
                

                state['step'] += 1 
        return
    
    def step(self, closure: LossClosure=None, loss: torch.Tensor=None) -> OptFloat:
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss, by default None.
        
        loss : torch.tensor, optional
            The loss tensor. Use this when the backward step has already been performed. By default None.
        

        Returns
        -------
        (Stochastic) Loss function value.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")

        self._number_steps += 1
        

        
        ############################################################
        # Notation
        # d_k: p.grad_avg, gamma_k: _gamma, \bar f_k: self.loss_avg
        for group in self.param_groups:
            for p in group['params']:
                
                grad = p.grad.data.detach()
                state = self.state[p]

                # Initialize EMA
                if self._number_steps == 1:
                    if self.bias_correction:
                        state['grad_hist'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                        state['grad_dot_w'] = torch.zeros(1).to(p.device)
                    else:
                        # Exponential moving average of gradients
                        state['grad_avg'] = grad.clone()
                        # Exponential moving average of inner product <grad, weight>
                        state['grad_dot_w'] = torch.sum(torch.mul(p.data, grad))
                        
                grad_avg, grad_dot_w = state['grad_avg'], state['grad_dot_w']

        #################   
        # Update
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            
            if self.use_fstar:
                cap = ((1+lr*lmbda)*self.loss_avg + _dot - (1+lr*lmbda)*_gamma).item()
                # Reset
                if cap < (1+lr*lmbda)*rho*self.lb:
                    self.lb = cap/(2*(1+lr*lmbda)*rho) 
                    self.lb = max(self.lb, self._initial_lb) # safeguard

            ### Compute adaptive step size
            if lmbda > 0:
                nom = (1+lr*lmbda)*(self.loss_avg - rho*self.lb) + _dot - (1+lr*lmbda)*_gamma
                t1 = max(nom, 0.)/_norm
            else:
                t1 = max(self.loss_avg - rho*self.lb + _dot - _gamma, 0.)/_norm
            
            t1 = t1.item() # make scalar
            
            tau = min(lr/rho, t1) # step size

            ### Update lb estimator
            if self.use_fstar:
                h = (self.loss_avg  + _dot -  _gamma).item()
                self.lb = ((h - (1/2)*tau*_norm)/rho).item() 
                self.lb = max(self.lb, self._initial_lb) # safeguard

            ### Update params
            for p in group['params']:   
                state = self.state[p]
                grad_avg = state['grad_avg']          
                p.data.add_(other=grad_avg, alpha=-tau)
                
                if lmbda > 0:
                    p.data.div_(1+lr*lmbda)
                    
        ############################################################
        if self.use_fstar:
            self.state['h'] = h
            self.state['fstar'] = self.lb
            
        self.state['step_size_list'].append(t1)
        
        return loss
    
