import re
import warnings
from functools import wraps
from inspect import signature

import neuroaiengines.networks as networks
import numpy as np
from neuroaiengines.networks.ring_attractor import RingAttractorPytorch
from neuroaiengines.optimization.datasets import (CSVDatasetGenerator,
                                                  RingAttractorDatasetGenerator,
                                                  SimulatorDatasetGenerator)
from neuroaiengines.optimization.torch import *
from neuroaiengines.utils.signals import (create_epg_bump_fn,
                                          get_epg_activation)

import torch
from torch import nn, tensor

rev_epg_act = lambda x: (get_epg_activation(x)+1.)/5.
def fix_config(config, training_fn, testing_fn):
    """
    Modifies the config so that it has the defaults from the training function and testing function. Used for repeatability.
    """
    train_sig = signature(training_fn).bind(**config['training_kwargs'])
    train_sig.apply_defaults()
    config['training_kwargs'] = train_sig.arguments

    test_sig = signature(testing_fn).bind(**config['testing_kwargs'])
    test_sig.apply_defaults()
    config['testing_kwargs'] = test_sig.arguments
    return config
# pylint: disable=not-callable
# Defaults for the functions, used here so they are accessible after calling
DEFAULTS = dict(
    run_optimization=dict(
        network_type='kakaria_bivort', 
        network_kwargs={},
        nonlinearity=nn.ReLU,
        tstart=0, 
        tend=0.5, 
        dt=0.001, 
        n_epochs=50,
        vrange=0,
        angle=np.pi/2,
        initial_alpha=0.0025,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr':1e-2},
        propagate_input=0,
        bump_fn=None,
        bump_fn_kwargs={},
        k1=10,
        k2=10,
        reporter=None,
        progressbar=False,
        bias=0,
        gain=0,
        initial_state=1,
        fixed_weights=[],
        noise=None,
        state_dict=None,
    ),
    test_optimization=dict(
        tstart=0, 
        tend=1, 
        dt=0.001, 
        vrange=0, 
        angle=0, 
        bump_fn=None,
        bump_fn_kwargs=None,
    )
)
class InitialConditioner():
    pass
class EPGInitialConditioner(InitialConditioner):
    def __init__(self, model, bump_fn,dataset, propagate_input=0, dt=0.001, initial_state=1):
        self.model=model
        self.bump_fn = bump_fn
        self.propagate_input=propagate_input
        self.dt= dt
        self.initial_state=initial_state
        self.dataset = dataset
    def __call__(self, i, sequence):
        angle = self.dataset.gt[i][0][0]
        return self.make_initial_condition(angle)
    def make_initial_condition(self, angle):
        """
        Makes the initial condition.

        arguments
        ---------
        angle:
            initial angle
        
        """
    
     
        u = torch.ones(self.model.state_size, dtype=float)*self.initial_state
        u[self.model.slcs['epg']] = tensor(self.bump_fn(angle))
        # Run the initial state through the model a few times to propagate the state to the other neurons
        with torch.no_grad():
            for _ in range(self.propagate_input):
                _,u = self.model([self.dt,0],u)
                u[self.model.slcs['epg']] = self.bump_fn(angle)
        return u
# TODO this is unnecessary, replace with inspect.Signature calls. 
# Defaults should be set in the function as normal.
def find_default_args(fn):
    """
    Wrapper to put in default arguments based on function name
    """
    key = re.findall(r'_default:\s*(.*)\s*',fn.__doc__)[0]
    def_kwargs = DEFAULTS[key]
    @wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs = {**def_kwargs, **kwargs}
        return fn(*args, **kwargs)
    return wrapper
def _get_network_from_name(network_type,initial_alpha=0.0025,**network_kwargs):

    symmetric_match = re.findall(r"symmetric(\d+)",network_type)
    hemibrain_match = re.findall(r"hemibrain(\d+)",network_type)
    if network_type == 'kakaria_bivort':
        # Using the kakaria bivort
        w,slcs = networks.get_kakaria_bivort()
        wd=9
        
        
    elif symmetric_match:
        # Using the generalized model
        wd = int(symmetric_match[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, slcs = networks.create_symmetric_ra_matrix(wd)
    elif hemibrain_match:
        wd = int(hemibrain_match[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, slcs = networks.create_symmetric_hemi_ra_matrix(wd, **network_kwargs)
    else:
        assert False, network_type + " is not supported"
    w = tensor(w)*initial_alpha
    return w, slcs
def run_csv_optimization(progressbar=False, **kwargs):
    """
    Runs optimization on the ring attractor. See init_trainer for kwargs.

    :param progressbar: 
    """
    trainer,rad,initcond = init_trainer(**kwargs)
    trainer.batch_train(rad,initial_conditioner=initcond ,progress_bar=progressbar)
    return trainer
def initialize(cls, *args, **kwargs):
    # Generalized initialize method with error forwarding
    try:
        return cls(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Received error intializing {cls}")
def init_trainer(
    fixed_weights=None,
    state_dict=None,

    network_cls=_get_network_from_name,
    network_kwargs=None,
    
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs=None,

    lr_scheduler_clss=None,
    lr_scheduler_kwargss=None,

    dataset_cls=RingAttractorDatasetGenerator,    
    dataset_kwargs=None,

    bump_fn_cls=create_epg_bump_fn,
    bump_fn_kwargs=None,

    model_cls=RingAttractorPytorch,
    model_kwargs=None,

    trainer_cls=TBPTT,
    trainer_kwargs=None,

    loss_module_cls=nn.MSELoss,
    loss_module_kwargs=None,

    initial_condition_cls=EPGInitialConditioner,
    initial_condition_kwargs=None,
    **kwargs
    ):
    """
    Initializes a trainer
    
    
    """
    model_kwargs = model_kwargs or {}
    dataset_kwargs = dataset_kwargs or {}
    bump_fn_kwargs = bump_fn_kwargs or {}
    network_kwargs = network_kwargs or {}
    
    optimizer_kwargs = optimizer_kwargs or {'lr':1e-2}
    trainer_kwargs = trainer_kwargs or {'k1':5,'k2':5}
    loss_module_kwargs = loss_module_kwargs or {}
    fixed_weights = fixed_weights or []
    initial_condition_kwargs = initial_condition_kwargs or {}
    
            
    w,slcs = initialize(network_cls,**network_kwargs)
    
    output_len = len(w[slcs['epg'] ])

    
    bump_fn = initialize(bump_fn_cls, int(output_len/2),**bump_fn_kwargs)
    
    rad = initialize(dataset_cls,
        output_len=output_len, 
        dudt_len=len(w),
        target_act_fn=bump_fn,
        **dataset_kwargs,
    )
    angle = rad.gt[0][0][0]
    # Create the dataset and model
    model = initialize(model_cls,
        w,
        slcs,
        initial_angle=angle,
        **model_kwargs,)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    

    optimize_params = [v for k,v in model.named_parameters() if k not in fixed_weights]
    optimizer = initialize(optimizer_cls,optimize_params, **optimizer_kwargs)
    loss_module = initialize(loss_module_cls,**loss_module_kwargs)
    initcond = initialize(initial_condition_cls,model, bump_fn, rad, **initial_condition_kwargs)
    lr_schedulers = []
    if lr_scheduler_clss is not None:
        if not isinstance(lr_scheduler_clss, list):
            lr_scheduler_clss = [lr_scheduler_clss]
        if not isinstance(lr_scheduler_kwargss,list):
            lr_scheduler_kwargss = [lr_scheduler_kwargss]
        for lr_scheduler_cls, lr_scheduler_kwargs in zip(lr_scheduler_clss, lr_scheduler_kwargss):
            lr_schedulers.append(initialize(lr_scheduler_cls,**lr_scheduler_kwargs,optimizer=optimizer))
    # Trainer
    trainer = initialize(trainer_cls,one_step_module=model, optimizer=optimizer, loss_module=loss_module,lr_schedulers=lr_schedulers,**trainer_kwargs)
    
    return trainer,rad,initcond
def test_csv_optimization(
    trainer, 
    initial_condition_cls=EPGInitialConditioner,
    initial_condition_kwargs=None,
    bump_fn_cls=create_epg_bump_fn, 
    bump_fn_kwargs=None,
    dataset_cls=RingAttractorDatasetGenerator,
    dataset_kwargs=None,
    ):
    """
    tests the optimization
    arguments:
    ----------
    trainer: 
        TBTT object

    dt:
        timestep
    vrange:
        range of velocities to test
    angle:
        initial angle
    bump_fn, bump_fn_kwargs:
        Defines desired EPG activation given angle
    

    """
    initial_condition_kwargs = initial_condition_kwargs or {}
    dataset_kwargs = dataset_kwargs or {}
    bump_fn_kwargs = bump_fn_kwargs or {}
    bump_fn = initialize(bump_fn_cls,int(trainer.model.output_size/2), **bump_fn_kwargs)
    rad_test = initialize(dataset_cls,
        output_len=trainer.model.output_size,
        dudt_len=trainer.model.state_size, 
        target_act_fn=bump_fn,
        **dataset_kwargs
    )
    initcond = initialize(initial_condition_cls,trainer.model, bump_fn, rad_test, **initial_condition_kwargs)

    
    rets = []
    for d in rad_test:
        ret = trainer.test(zip(*d),initcond)
        rets.append(ret)
    
    return rets,rad_test

