from argparse import ArgumentParser
from operator import itemgetter
from tabnanny import verbose
from typing import Dict
from importlib_metadata import metadata
import numpy as np
import os
import pandas as pd
from neuroaiengines.networks import create_symmetric_hemi_ra_matrix
from neuroaiengines.networks.ring_attractor import RingAttractorPytorch
from neuroaiengines.optimization.torch import TBPTT, create_decoded_mse
from torch import Tensor, is_tensor, nn, tensor
from torch.optim import lr_scheduler
from torch.optim import Adam
import torch
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from make_data import N_EPG, bump_fn

def main():
    summary_writer = SummaryWriter()
    get_data_path = lambda *p: os.path.join(summary_writer.get_logdir(),'data',*p)
    os.makedirs(get_data_path(), exist_ok=True)
    # Load data
    
    training_data_inputs = torch.from_numpy(np.load(os.path.join('data','training_data_inputs.npy')))
    training_data_targets = torch.from_numpy(np.load(os.path.join('data','training_data_targets.npy')))

    # Generate the weights and slices
    # Slices are a dict of population names to slice objects, which defines which slices of the weight matrix correspond to populations.
    # E.g w[slcs['epg'],slcs['pen']] gives the weights where EPG is presynaptic and PEN is postsynaptic (a block of weights)
    w, slcs = create_symmetric_hemi_ra_matrix(N_EPG,gaussian=True, both=True, null_offset=True)
    # Define the gains/biases for each neuron population. Passing these to the network trains them as parameters!
    # If left out, they are left to 1 or 0 respectively and NOT trained.
    gains = {
        'epg' : 1.,
        'pen' : 1.,
        'peg' : 1.,
        'd7' : 1.
    }
    biases = {
        'epg' : 0.,
        'pen' : 0.,
        'peg' : 0.,
        'd7' : 0.
    }
    
    
    ring_attractor = RingAttractorPytorch(
        # Multiplied by a small initial scaling factor
        tensor(w*0.0025),
        slcs,
        initial_angle=0,
        gain=gains,
        bias=biases,
        # Specifies the neuron *activation* nonlinearity
        nonlinearity=nn.ReLU,
        # Split all ipsi/contralateral populations, EXCEPT this population(s)
        no_ipsi_contra_split=['d7'], 
        # Don't use landmark input
        use_landmarks=False,
        
    )

    # Fixing one weight does NOT change the problem, it just puts all the parameters in terms of this one (chosen arbitrarily)
    fixed_weights = ['d7_peg']
    loss_module = create_decoded_mse(N_EPG,hemisphere_offset=True, )
    optimize_params = [v for k,v in ring_attractor.named_parameters() if k not in fixed_weights]
    optimizer = Adam(optimize_params, lr=0.01, weight_decay=0)
    lr_scheduler_module = lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0,10000,1000)))
    initial_condition = np.ones(ring_attractor.state_size)
    initial_condition[ring_attractor.slcs['epg']] = bump_fn(0)
    initial_condition = tensor(initial_condition)
    initial_conditioner = lambda x,y: initial_condition
    trainer = TBPTT(ring_attractor,optimizer=optimizer, loss_module=loss_module,lr_schedulers=[lr_scheduler_module], k1=499,k2=499,cumulative_loss=True)
    dataset = TensorDataset(training_data_inputs, training_data_targets)
    metadata_list = []
    def epoch_callback(r:Dict,i:int,metadata_list=metadata_list):
        r.pop('states')
        for k,v in r.items():
            if isinstance(v,Tensor):
                r[k] = v.cpu().detach().numpy()
            
        summary_writer.add_scalars('train',r,i)
        metadata_list.append((i,r))

    try:
        trainer.batch_train(dataset, initial_conditioner=initial_conditioner, epoch_callback=epoch_callback)
    except KeyboardInterrupt:
        pass
    # Write out model
    torch.save(ring_attractor.state_dict(), get_data_path('model.pt'))
    m_idx,m_dat = list(zip(*metadata_list))
    pd.DataFrame(m_dat,m_idx).to_pickle(get_data_path('train_metadata.pkl'))
if __name__ == "__main__":
    main()