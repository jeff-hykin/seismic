from abc import abstractmethod
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.functional import Tensor
from tqdm import tqdm

from neuroaiengines.utils.signals import create_decoding_fn
import pandas as pd
#pylint: disable=not-callable

from functools import partial
import warnings

class NanValueError(ValueError):
    pass

class EarlyTermination(Exception):
    pass

class TBTTModule(nn.Module):
    @abstractmethod
    def update_parameterizations(self):
        """
        Since TBTT only updates the optimizer params at certain points, some models may increase speed from updating paramerizations of those parameters only when the optimization happens.
        """
        pass

class TBPTT():
    """
    Truncated backpropagation through time optimization
    This class is mostly from https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/4
    
    
    """
    def __init__(self, 
                 one_step_module :TBTTModule, 
                 loss_module:nn.Module,
                 k1:int, 
                 k2:int, 
                 optimizer:torch.optim.Optimizer,
                 lr_schedulers:Optional[Iterable[nn.Module]]=None,
                 backprop_k1_states:bool=False,
                 epoch_callback:Optional[Callable]=None,
                 cumulative_loss:bool=False
                ):
        """
        params:
        -------
        one_step_module: 
            Model to be optimized. Has the call signature 
            # output, new_state = one_step_module(input, state)
        loss_module: 
            Loss function. Has the call signature
            # loss = loss_module(output, target)
        k1,k2:
            every k1 steps, backprop loss k2 states.
            good visualization below.
        optim:
            Optimizer for the model. 
            Should be initialized already with one_step_module.parameters()
        lr_schedulers:
            A learning rate scheduler or list of learning rate schedulers for the optimizer. They will be applied in the order given.
        backprop_k1_states:
            Uses the loss from the last k1 outputs/targets. Otherwise, the loss is computed at the most recent target/output and backpropped through the last k1 states.
        epoch_callback:
            after a training epoch, call this function with the epoch result dict
        cumulative_loss:
            Accumulate loss instead of using last value
        """
        self.one_step_module = one_step_module
        self.model = one_step_module
        self.loss_module = loss_module
        self.k1 = k1
        self.k2 = k2
        # TODO unsure about this line -- for my code, having this true breaks it. 
        # I'm still unclear on pytorch and retain graph.
        self.retain_graph = k1 < k2
        self.retain_graph = True
        self.mk1k2 = max(k1,k2)
#         self.retain_graph = False
        # You can also remove all the optimizer code here, and the
        # train function will just accumulate all the gradients in
        # one_step_module parameters
        self.optimizer = optimizer
        self.requires_closure = isinstance(optimizer, torch.optim.LBFGS)
        self.losses = []
        self.parameters = None
        self.states = None
        self.backprop_k1_states = backprop_k1_states
        self.epoch_callback = epoch_callback
        self.cumulative_loss = cumulative_loss
        # Assume LR schedulers initialized
        if lr_schedulers is None:
            self.lr_schedulers = []
        else:
            try:
                iter(lr_schedulers)
                self.lr_schedulers = lr_schedulers
            except TypeError:
                # Not iterable
                self.lr_schedulers = [lr_schedulers]
        
    def train(self, input_sequence: Iterable[Tuple[torch.Tensor,torch.Tensor]], init_state: torch.Tensor):
        """Trains on a single input sequence

        Args:
            input_sequence (Iterable[Tuple[torch.Tensor,torch.Tensor]]): Input sequence 
            init_state (torch.Tensor): initial state

        Raises:
            NanValueError: If state has a nan value in it
        Returns:
            dict: information about the training, including state, mean_loss, and final_loss
        """
        states = [(None, init_state) ] # (prev_state, curr_state)
        targets = []
        outputs = []
        inputs = []
        stage_loss = None
        total_loss = 0.
        cum_loss = tensor(0.,requires_grad=True)
        for i, (inp, target) in enumerate(input_sequence):
            
            # Get the "current" state from the last timestep
            state = states[-1][1].detach()
            if torch.any(torch.isnan(state)):
                raise NanValueError(f'State contains nan values : {state}')
            state.requires_grad=True
            output, new_state = self.one_step_module(inp, state)
            states.append((state, new_state))
            while len(states) > self.mk1k2:
                # Delete stuff that is too old
                del states[0]
            if self.backprop_k1_states or self.requires_closure:
                targets.append(target)
                outputs.append(output)
                inputs.append(inp)
                while len(outputs) > self.mk1k2:
                    # Delete stuff that is too old
                    del outputs[0]
                    del targets[0]
                    del inputs[0]
            # Calculate loss to track
            stage_loss = self.loss_module(output, target)
            if self.cumulative_loss:
                cum_loss = cum_loss + stage_loss
            total_loss += stage_loss.item()
            # k1 steps have gone, time to backprop

            if (i+1)%self.k1 == 0:
                
                self.optimizer.zero_grad()
                
                if (not self.backprop_k1_states) and (not self.cumulative_loss):
                    
                    # backprop last module (keep graph only if they ever overlap)
                    stage_loss.backward(retain_graph=self.retain_graph)
                elif self.cumulative_loss:
                    cum_loss.backward(retain_graph=self.retain_graph)
                # Go back k2 states
                for j in range(self.k2-1):
                    # if we get all the way back to the "init_state", stop
                    if states[-j-2][0] is None:
                        break
                    # If backpropping states, do it here
                    if ((j < self.k1) and self.backprop_k1_states) and (not self.cumulative_loss):
                        loss = self.loss_module(outputs[-j-1], targets[-j-1])
                        
                        loss.backward(retain_graph=True)
                    
                    curr_grad = states[-j-1][0].grad
                    states[-j-2][1].backward(curr_grad, retain_graph=self.retain_graph)
                   
#                     curr_grad = states[-j-1][0].grad
#                     states[-j-2][1].backward(curr_grad, retain_graph=self.retain_graph)
                if self.requires_closure:
                    self.optimizer.step(partial(self.closure, states, targets, inputs))
                    self.one_step_module.update_parameterizations()
                else:
                    self.optimizer.step()
                    self.one_step_module.update_parameterizations()
                # Reset cumulative loss
                if self.cumulative_loss:
                    cum_loss = tensor(0., requires_grad=True)
        return {'mean_loss' : total_loss/(i+1),
                'final_loss' : stage_loss,
                'states': np.array([s[1].data.numpy() for s in states])}
    def closure(self, states, targets, inputs):
        # self.optimizer.zero_grad()
        # state = states[0][1].detach() # state to start from --  we are going forward!
        
        
        # if self.backprop_k1_states:
        #     for i,(inp,target) in enumerate(zip(inputs, targets)):
                
        #         output, state = self.one_step_module(inp,state)
        #         loss = self.loss_module(output, target)
        #         loss.backward(retain_graph=self.retain_graph)
        
        # else:
        #     output,_ = self.one_step_module(inputs[-1],states[-1][1].detach())

        #     loss = self.loss_module(output, targets[-1])
        #     loss.backward(retain_graph=self.retain_graph)
        self.optimizer.zero_grad()
        outputs = []
        state = states[0][1]
        new_states = [(None, state)]
        for inp in inputs:
            state = new_states[-1][1].detach()
            state.requires_grad=True
            output,new_state = self.one_step_module(inp,state)
            new_states.append((state,new_state))
            outputs.append(output)
        if not self.backprop_k1_states:
            loss = self.loss_module(outputs[-1], targets[-1])
            
            # backprop last module (keep graph only if they ever overlap)
            loss.backward(retain_graph=self.retain_graph)
        # Go back k2 states
        loss = tensor(0., requires_grad=True)
        for j in range(self.k2-1):
            # if we get all the way back to the "init_state", stop
            if new_states[-j-2][0] is None:
                break
            # If backpropping states, do it here
            if j < self.k1 and self.backprop_k1_states:
                loss_ = self.loss_module(outputs[-j-1], targets[-j-1])
                if self.cumulative_loss:
                    loss = loss + loss_
                else:
                    loss.backward(retain_graph=True)

            # curr_grad = new_states[-j-1][0].grad
            # print(new_states[-j-2][1], new_states[-j-1][0])
            # new_states[-j-2][1].backward(curr_grad, retain_graph=self.retain_graph)
        if self.cumulative_loss:
            loss.backward(retain_graph=True)
        return loss
    def batch_train(self,
                    input_sequencer: Iterable[Iterable[Tuple[torch.Tensor, torch.Tensor]]], 
                    initial_conditioner: Union[Callable, Iterable],
                    epoch_callback: Optional[Callable]=None,
                    progress_bar=True):
        """
        Trains of a bunch of sequences
        params:
        -------
        input_sequencer: 
            Iterable of iterables that contain inputs and targets
        initial_conditioner:
            function that returns an initial state OR an initial state that remains constant
            callable must have signature f(epoch_number, (inputs, targets))
        epoch_callback:
            a callback that is called with form f(train_return_dict, epoch_number)
            train_return_dict includes mean_loss, final_loss, and states from training. See train().
        """
        # self.mean_losses = np.ones(len(input_sequencer))*np.nan
        # self.final_losses = np.ones(len(input_sequencer))*np.nan

        
        # self.states = None
        if self.parameters is None:
            self.parameters = pd.DataFrame(columns=[n for n,v in self.one_step_module.named_parameters()])
        self.parameters.loc[0,:] = {n:v.detach().clone().numpy() for n,v in self.one_step_module.named_parameters()}
        nancount = 0
        if not callable(initial_conditioner):
            initial_conditioner = lambda i,d: initial_conditioner
        for i,d in tqdm(enumerate(input_sequencer),"Epochs",total=len(input_sequencer), disable=not progress_bar):
            init_state = initial_conditioner(i,d)
            try:
                r = self.train(zip(*d), init_state)
                nancount=0
            except NanValueError:
                warnings.warn(f'Received nan values in training epoch {i}, continuing...')
                nancount += 1
                if nancount > 10:
                    warnings.warn('Received 10 nan values in a row, returning...')
                    break
                continue
            # self.final_losses[i] = r['final_loss']
            # self.mean_losses[i] = r['mean_loss']
            
            
                
            # if self.states is None:
            #     self.states = np.ones((len(input_sequencer), *r['states'].shape))*np.nan
            
            # self.states[i,:,:] = r['states']
            self.parameters.loc[i+1,:] = {n:v.detach().clone().numpy() for n,v in self.one_step_module.named_parameters()}
            try:
                for lr_scheduler in self.lr_schedulers:
                    # lr_scheduler.step(r['mean_loss'])
                    lr_scheduler.step()
                    # Check if all optimizer param groups LR are close to zero -- no need to continue training!
                    lrs = [group['lr'] for group in self.optimizer.param_groups]
                    if np.all(np.array(lrs) <= 1e-8):
                        raise EarlyTermination
                if epoch_callback is not None:
                    epoch_callback(r,i)
            except EarlyTermination:
                print(f'Learning rates reached {lrs}. Exiting early!')
                break
            
            
    def test(self, 
             input_sequence : Iterable[Tuple[torch.Tensor, torch.Tensor]], 
             initial_conditioner : Callable,
            ):
        """
        Tests on an input sequence
        params:
        -------
        input_sequence: 
            The input sequence composed of inputs, targets
        init_state:
            initial state
        """
        init_state = initial_conditioner(0,input_sequence)
        states = [(None, init_state)]
        total_loss = 0
        with torch.no_grad():
            for j, (inp, target) in enumerate(input_sequence):

                state = states[-1][1].detach()
                output, new_state = self.one_step_module(inp, state)
                states.append((state, new_state))
                loss = self.loss_module(output, target).item()
                total_loss += loss
            
        return {'mean_loss' : total_loss/(j+1),
                'final_loss' : loss,
               'states': np.array([s[1].data.numpy() for s in states])}

def cosine_similarity(x1,x2):
    """
    Cosine similarity (CS)
    """
    return 1 - torch.matmul(x1,x2)/(torch.linalg.norm(x1)*torch.linalg.norm(x2))

def mult_similarity(x1,x2):
    """
    CS*MSE
    """
    return cosine_similarity(x1,x2)*nn.MSELoss()(x1,x2)

def MSELoss(x1,x2):
    """
    MSE
    """
    return nn.MSELoss()(x1,x2)

def add_similarity(x1,x2,a=1,b=1):
    """
    a*CS+b*MSE
    """
    return a*cosine_similarity(x1,x2)+b*nn.MSELoss()(x1,x2)

def max_similarity(x1,x2):
    """
    CS+DiffMax
    """
    return cosine_similarity(x1,x2)+torch.abs(torch.max(x1)-torch.max(x2))

from neuroaiengines.utils.signals import create_pref_dirs
from torch import tensor
def create_hilbert_loss_fn(sz):
    
    epg_pref_dirs = create_pref_dirs(sz,centered=True)
    epg_pref_dirs = np.tile(epg_pref_dirs, (2,1))
    epg_pref_dirs_inv = tensor(np.linalg.pinv(epg_pref_dirs))
    def decode_epgs(act):
        act = act*5 - 1
        sc = torch.matmul(epg_pref_dirs_inv,act)
        return sc
    def hilbert(x1,x2):
        s1,c1 = decode_epgs(x1)
        s2,c2 = decode_epgs(x2)
        n1,n2 = torch.linalg.norm(x1), torch.linalg.norm(x2)
        w = torch.abs(n1-n2)/n1
        cc = c1*c2 + s1*s2 + (s1*c2 - c1*s2)
        return torch.abs(w*cc)
    return hilbert

def create_angluar_cosine_difference(sz):
    decode = create_decoding_fn(sz, sincos=True, backend=torch)
    
    
    def cosine_difference(x1,x2):
        sc1 = decode(x1)
        sc2 = decode(x2)
        return cosine_similarity(sc1,sc2)
    return cosine_difference

def create_angular_cosine_diff_mse(sz, cs_weight=1, mse_weight=1):
    cs_fn = create_angluar_cosine_difference(sz)
    mse = nn.MSELoss()
    def combined(x1,x2): 
        return cs_fn(x1,x2)*cs_weight + mse(x1,x2)*mse_weight
    return combined

def create_angular_cosine_diff_norm(sz, cs_weight=1, norm_weight=1):
    cs_fn = create_angluar_cosine_difference(sz)
    norm = torch.linalg.norm
    def combined(x1,x2): 
        return cs_fn(x1,x2)*cs_weight + torch.dist(norm(x1),norm(x2))*norm_weight
    return combined
def create_angular_cosine_diff_variance(sz, cs_weight=1, var_weight=1):
    cs_fn = create_angluar_cosine_difference(sz)
    var = torch.var
    def combined(x1,x2): 
        return cs_fn(x1,x2)*cs_weight + torch.square(torch.dist(var(x1),var(x2)))*var_weight
    return combined
def create_decoded_mse(sz: int, **kwargs) -> Tensor:
    """
    Erik's magical loss function.

    returns mse_loss(decode(x1), decode(x2))

    Args:
        sz : Size of the expect input vector 

    Returns:
        Tensor: Loss
    """
    decode = create_decoding_fn(sz, sincos=True, backend=torch, **kwargs)
    mse = nn.MSELoss()
    def decoded_mse(x1,x2):
        sc1 = decode(x1)
        sc2 = decode(x2)
        return mse(sc1,sc2)
    return decoded_mse
def create_decoded_mse_var(sz, mse_lambda=1, var_lambda=1, **kwargs):
    mse = create_decoded_mse(sz, **kwargs)
    var = torch.var
    def combined(x1,x2): 
        return mse(x1,x2)*mse_lambda + torch.square(torch.dist(var(x1),var(x2)))*var_lambda
    return combined
def create_angular_cosine_diff_minmax(sz, cs_weight=1, min_weight=1, max_weight=1):
    cs_fn = create_angluar_cosine_difference(sz)
    min = torch.min
    max = torch.max
    def combined(x1,x2): 
        return cs_fn(x1,x2)*cs_weight + torch.dist(min(x1),min(x2))*min_weight + torch.dist(max(x1),max(x2))*max_weight
    return combined
def create_angular_cosine_diff_singular_variance(sz, cs_weight=1, var_weight=1):
    cs_fn = create_angluar_cosine_difference(sz)
    var = torch.var
    def combined(x1,x2): 
        return cs_fn(x1,x2)*cs_weight +  (1 - var(x1))*var_weight
    return combined
def create_angular_cosine_diff_minmean(sz, cs_weight=1, min_weight=1, mean_weight=1):
    cs_fn = create_angluar_cosine_difference(sz)
    min = torch.min
    mean = torch.mean
    def combined(x1,x2): 
        return cs_fn(x1,x2)*cs_weight + torch.dist(min(x1),min(x2))*min_weight + torch.dist(mean(x1),mean(x2))*mean_weight
    return combined
def create_angular_cosine_diff_varmean(sz, cs_weight=1, var_weight=1, mean_weight=1):
    cs_fn = create_angluar_cosine_difference(sz)
    var = torch.var
    mean = torch.mean
    def combined(x1,x2): 
        return cs_fn(x1,x2)*cs_weight + torch.dist(var(x1),var(x2))*var_weight + torch.dist(mean(x1),mean(x2))*mean_weight
    return combined
def create_combined_loss(*loss_fns):
    """
    Creates a linear combination of loss functions
    """
    all_loss_fns = []
    all_weights = []
    for loss_fn in loss_fns:
        if isinstance(loss_fn, tuple):
            loss_fn,weight = loss_fn
        else:
            weight = 1
        all_loss_fns.append(loss_fn)
        all_weights.append(weight)
    def combined_loss(x1,x2):
        loss = tensor(0)
        for loss_fn, weight in zip(all_loss_fns, all_weights):
            loss += weight*loss_fn(x1,x2)
        return loss
    return combined_loss
