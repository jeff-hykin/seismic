from base64 import decode
from pickletools import optimize
from tabnanny import verbose
import numpy as np
import os
import pickle
from neuroaiengines.networks import create_symmetric_hemi_ra_matrix
from neuroaiengines.networks.ring_attractor import RingAttractorPytorch
from neuroaiengines.optimization.torch import TBPTT, create_decoded_mse
from torch import initial_seed, nn, tensor
from torch.optim import lr_scheduler
from torch.optim import Adam
import torch
from torch.utils.data import TensorDataset

from make_data import N_EPG, bump_fn
from neuroaiengines.utils.signals import create_decoding_fn
def main():
    get_data_path = lambda *p: os.path.join(os.path.dirname(__file__),'data',*p)
    # Load model parameters

    state_dict = torch.load(get_data_path('model.pt'))
    
    testing_data_inputs = np.load(get_data_path('testing_data_inputs.npy'),'r')
    testing_data_targets = np.load(get_data_path('testing_data_targets.npy'),'r')
    
    # Generate the weights and slices
    # Slices are a dict of population names to slice objects, which defines which slices of the weight matrix correspond to populations.
    # E.g w[slcs['epg'],slcs['pen']] gives the weights where EPG is presynaptic and PEN is postsynaptic (a block of weights)
    w, slcs = create_symmetric_hemi_ra_matrix(N_EPG,gaussian=True, both=True, null_offset=False)
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
        tensor(w)*0.0025,
        slcs,
        initial_angle=0,
        gain=gains,
        bias=biases,
        # Specifies the neuron *activation* nonlinearity
        nonlinearity=nn.ReLU,
        # Split all ipsi/contralateral populations, EXCEPT this population(s)
        no_ipsi_contra_split=['d7'], 
        # Don't use landmark input
        use_landmarks=False
    )
    ring_attractor.load_state_dict(state_dict)
    # Fixing one weight does NOT change the problem, it just puts all the parameters in terms of this one (chosen arbitrarily)
    loss_module = create_decoded_mse(N_EPG,hemisphere_offset=True)
    decoder = create_decoding_fn(N_EPG,hemisphere_offset=True)
    initial_condition = torch.ones(ring_attractor.state_size, dtype=float)
    initial_condition[ring_attractor.slcs['epg']] = tensor(bump_fn(0))
    dataset = TensorDataset(torch.from_numpy(testing_data_inputs), torch.from_numpy(testing_data_targets))
    testing_dataframe = []
    with torch.no_grad():
        for test_dataset in dataset:
            states = [(None, initial_condition)]
            decoded_angles = [0]
            ground_truths = [0]
            total_loss = 0
            for inp, target in zip(*test_dataset):

                state = states[-1][1].detach()
                output, new_state = ring_attractor(inp, state)
                states.append((state, new_state))
                loss = loss_module(output, target).item()
                decoded_angles.append(decoder(output.detach().numpy()))
                ground_truths.append(decoder(target.detach().numpy()))
                total_loss += loss
            states = np.array([s[1].numpy() for s in states])
            testing_dataframe.append({
                'mean_loss':total_loss/len(dataset),
                'final_loss':loss,
                'states':states,
                'decoded_angle':decoded_angles,
                'ground_truth':ground_truths
                })
    with open(get_data_path('testing_outputs.pkl'),'wb') as fp:
        pickle.dump(testing_dataframe,fp)
    
if __name__ == "__main__":
    main()