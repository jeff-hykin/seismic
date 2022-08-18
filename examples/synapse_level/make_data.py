import os
import numpy as np
from neuroaiengines.optimization.datasets import RingAttractorDatasetGenerator
import pickle

from neuroaiengines.utils.signals import create_epg_bump_fn
N_EPG = 8
bump_fn = create_epg_bump_fn(N_EPG, fwhm=np.pi/3,scaling_factor=1.,hemisphere_offset=True)
def main():
    n_epochs = 6000
    # Time of samples
    trange = np.arange(0,0.5,0.001)
    # Velocity samples
    vrange = np.ones((n_epochs,len(trange)))*np.pi*4
    training_dataset_generator = RingAttractorDatasetGenerator(
        target_act_fn=bump_fn,
        vrange=vrange,
        trange=trange,
        initial_angles=0
    )
    training_input, training_targets, training_metadata = training_dataset_generator.get_data()

    # Make data dirs if they don't exist already
    datadir = 'data'
    os.makedirs(datadir, exist_ok=True)
    # Save files
    with open(os.path.join(datadir, "training_data_inputs.npy"),'wb') as fp:
        np.save(fp, training_input)
        del training_input
    with open(os.path.join(datadir, "training_data_targets.npy"),'wb') as fp:
        np.save(fp, training_targets)
        del training_targets
    with open(os.path.join(datadir, "training_data_metadata.pkl"),'wb') as fp:
        pickle.dump(training_metadata, fp)
        del training_metadata
    del training_dataset_generator

    vrange = np.ones((1,len(trange)))*np.pi*4
    testing_dataset_generator = RingAttractorDatasetGenerator(
        target_act_fn=bump_fn,
        vrange=vrange,
        trange=trange,
        initial_angles=0
    )
    testing_input, testing_targets, testing_metadata = testing_dataset_generator.get_data()

   
    # Save files
    with open(os.path.join(datadir, "testing_data_inputs.npy"),'wb') as fp:
        np.save(fp, testing_input)
        del testing_input
    with open(os.path.join(datadir, "testing_data_targets.npy"),'wb') as fp:
        np.save(fp, testing_targets)
        del testing_targets
    with open(os.path.join(datadir, "testing_data_metadata.pkl"),'wb') as fp:
        pickle.dump(testing_metadata, fp)
        del testing_metadata
if __name__ == "__main__":
    main()


    


    
