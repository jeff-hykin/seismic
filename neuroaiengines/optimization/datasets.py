"""
Contains datasets to be used with torch optimization
"""
# pylint: disable=no-name-in-module
from copy import copy
from typing import Callable, Iterable, Union, Mapping
import warnings

import numpy as np
from numpy.core.shape_base import atleast_2d
try:
    from sklearn.linear_model import Lasso
except ModuleNotFoundError:
    Lasso = None

import torch
from torch import tensor


from neuroaiengines.optimization.simulator import (ObservationRecorderMetric,
                                           StateRecorderMetric)
from neuroaiengines.optimization.simulator import BasicSimulator
import functools
from tqdm import tqdm
import pandas as pd
# pylint: disable=not-callable
class NeuronActivationDatasetGenerator():
    """
    Base Dataset class for neuron activation over time
    """
    def __init__(self, trange, *args, output_len=None,dudt_len=None):
        """
        params:
        -------
        output_len:
            length of the output
        trange:
            Range of times in the dataset. dt will be calcuated from this.
            ex.
            trange=np.linspace(0,1,100)
        """
        self.dudt_len = dudt_len
        self.output_len = output_len
        self.trange = np.array(trange)
    
        
    def __len__(self):
        return len(self.trange)
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def get_data(self):
        """
        Returns data from the dataset in terms of 

        
        Returns:
            (tensor, tensor, dict) : (input, target, metadata)
            input: TxN
            target: TxM
            metadata: dict with properties defined by the dataset, such as dropout indices etc
            
        """
        raise NotImplementedError
def broadcast_first_dim(a,b):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    ashp = a.shape
    bshp = b.shape
    if bshp[0] == 1 and ashp[0] == 1:
        return a,b
    assert not bshp[0] == 1 or not ashp[0] == 1, 'One array must have the length of 1 in the first dim'
    if ashp[0] < bshp[0]:
        a = np.array(np.broadcast_to(a,(bshp[0],*ashp[1:])))
    if ashp[0] > bshp[0]:
        b = np.array(np.broadcast_to(b,(ashp[0],*bshp[1:])))
    return a,b


class RingAttractorDatasetGenerator(NeuronActivationDatasetGenerator):
    """
    A dataset specifically for the Drosophila ring attractor
    """
    def __init__(self, 
                 target_act_fn : Callable, 
                 vrange : Union[Iterable[float],Iterable[Iterable[float]]], 
                 *args,
                 initial_angles: Union[float, Iterable[float]] = 0., 
                 **kwargs
                ):
        """
        params:
        -------
        target_act_fn: 
            Turns angle into the desired bump.
        vrange:
            Range of velocities in the dataset
        initial_angle:
            initial angle
        """
        super().__init__(*args, **kwargs)
        
        
        vrange = np.atleast_1d(vrange)
        initial_angles = np.atleast_1d(initial_angles)
        vrange,initial_angles = broadcast_first_dim(vrange,initial_angles)

        if len(vrange.shape) == 1:
            # Vrange is of the form (n_epochs,)

            vrange = np.tile(vrange,(len(self.trange),1)).T
        elif len(vrange.shape )== 2:
            # Vrange is of the form (n_epochs, time)
            pass
        else:
            raise ValueError('Unsupported vrange type')
        
        self.a_fn = target_act_fn
        self.output_len = len(target_act_fn(0.))
        self._generate_data(initial_angles, vrange)
    def _generate_data(self, 
                      initial_angles :float, 
                      vrange: Iterable[float]
                     ):
        '''
        generates timeseries data for the ring attractor
        
        
        
        params:
        -------
        initial_angle:
            initial angle
        vrange : 
            range of angular velocities to gen data for
        
        
        '''
        trange = self.trange
        epochs = vrange.shape[0]
        # Input for given velocity is t,v over time
        inp = np.zeros((epochs, len(trange), 2))
        # Output for given velocity is output activation over time
        out = np.zeros((epochs, len(trange), self.output_len))
        # Output for given velocity is output activation over time
        gt = np.zeros((epochs, len(trange),1))
        # Create the dts
        dts = trange[1:] - trange[:-1]
        # Iterate over desired velocities
        for i in range(epochs):
            inp[i,:-1,0] = dts
            inp[i,:,1] = vrange[i,:]
            a = initial_angles[i]
            
            # First angle should be initial
            out[i,0,:] = self.a_fn(a)
            gt[i,0,0] = copy(a)
            # Iterate over desired times
            for j, dt in enumerate(dts,1):
                
                out[i,j,:] = self.a_fn(a)
                # Increase angle using euler integration
                a += vrange[i,j]*dt
                gt[i,j,0] = copy(a)
        self.gt = gt
        self.out = tensor(out)
        self.inp = tensor(inp)
    def __len__(self):
        return len(self.inp)
    def __getitem__(self, idx):
        # Indexing desired velocities
        return self.inp[idx,:,:], self.out[idx,:,:]
    def get_data(self):
        return self.inp.numpy(), self.out.numpy(), {'ground_truth_angle':self.gt}
    
class SimulatorDatasetGenerator(NeuronActivationDatasetGenerator):
    """
    Dataset that uses a simulator from the sim-env-policy loop. 
    Observations from the simulator are passed through an activation function to produce
    input currents that can be passed into a neural population. A mapping can be learned 
    and used between the ground truth and the observation.
    """
    _passthrough = lambda x: x
    def __init__(self, 
            *args,
            sim: BasicSimulator,
            in_act_fn : Callable=_passthrough, 
            in_target_act_fn: Union[Callable, None]=None,
            target_act_fn: Callable=_passthrough,
            learn_mapping: bool=False,
            seeds: Union[Iterable[int], int]=None,
            epochs: int=None,
            state_slc: Union[np.ndarray,slice,int]=2,
            observation_slc: Union[np.ndarray,slice,int]=2,
            observation_passthrough_slc: Union[np.ndarray,slice,int]=0,
            share_learning:bool=False,
            alpha=0.0001, 
            max_iter=1000000,
            fit_intercept=False,
            dropout:Mapping[float, Iterable[bool]]=None,
            progressbar=False,
            generate_as_needed=False,
            **kwargs):
        """
        params:
        -------
        sim: simulator object to use to generate data
        in_act_fn, in_target_act_fn: If learn_mapping is true, then a positive linear mapping is 
            learned between in_act_fn and in_target_act_fn. Else, in_act_fn is used alone to determine input currents from the observation. 
            in_act_target_fn acts on the state. If None, uses the target_act_fn instead.
        target_act_fn: Used to generate ground truth (i.e target) from state using state_slc
        learn_mapping: whether to learn a mapping for the output
        seeds: seeds for the simulator. One per epoch.
        epochs: number of epochs. Seeds/epochs will be cast to the largest size.
        state_slc: Slice of the state to pass into the in_act_fn. Defaults to theta (2) for the KinematicPoint environment
        observation_slc: slice of the observation to pass into the target_act_fn. Defaults to theta (2) for the KinematicPoint environment.
        observation_passthrough_slc: slice of the observation to not modify using the in_act_fn. Gets prepended to the model input. Defaults to vtheta in the KinematicPoint environment.
        share_learning: whether the mapping should be learned per epoch (share_learning=False) or shared across all epochs.
        alpha, max_iter, fit_intercept: Parameters for Lasso solver
        dropout: a mapping of times to lists of booleans, which "dropout" observations. 
        """
        super().__init__(*args,**kwargs)
        self.sim = sim
        if seeds is None and epochs is None:
            raise ValueError('Either seeds or epochs must be set')
        n_epochs = epochs if epochs is not None else 1
        if seeds is None or isinstance(seeds,(int,str)):

            seeds = [seeds]*n_epochs
            self.cache_results = True
        else:
            n_epochs = len(seeds)
            self.cache_results = False
        self.seeds = seeds
        self.n_epochs = n_epochs
        self.in_act_fn = in_act_fn
        self.out_act_fn = target_act_fn
        self.in_target_act_fn = in_target_act_fn or target_act_fn
        self.state_slc = state_slc
        self.observation_slc = observation_slc
        self.observation_passthrough_slc = observation_passthrough_slc
        self.out = None
        self.inp = None
        self.gt = None
        self.target = None
        self.dropout_inp = None
        self.dropout = dropout
        self._passthrough_length = None
        assert (not share_learning or not generate_as_needed), "Cannot generate as needed with shared learning on"
        self.generate_as_needed = generate_as_needed
        self.share_learning = share_learning
        self.solver_kwargs = {'alpha':alpha,'max_iter':max_iter,'fit_intercept':fit_intercept}
        self.learn_mapping = learn_mapping
        self.transform_model = np.empty(self.n_epochs,dtype=object)
        if not generate_as_needed:
            self._generate_data(progressbar=progressbar)
        else:
            self[0]
    def _set_epoch_data(self, i, inp, out, target, gt, dts,dinp):
        # if self.out is None:
        #     self.out = np.zeros((self.n_epochs,*out.shape))
        # self.out[i,:,:] = out
        # if self.inp is None:
        #     inps = inp.shape
        #     self.inp = np.zeros((self.n_epochs,inps[0], inps[1]+1))
        # self.inp[i,:-1,0] = dts[1:] - dts[:-1]
        # self.inp[i,:,1:] = inp
        # if self.target is None:
        #     self.target = np.zeros((self.n_epochs, *target.shape))
        # self.target[i,:,:] = target
        # if self.dropout is not None:
        #     if self.dropout_inp is None:
        #         self.dropout_inp = np.zeros((self.n_epochs, *dinp.shape))
        #     self.dropout_inp[i,:,:] = dinp 
        # if self.gt is None:
        #     self.gt = np.zeros((self.n_epochs,*gt.shape))
        # self.gt[i,:,:] = gt
        if self.inp is None:
            self.inp = []
        if self.out is None:
            self.out = []
        if self.gt is None:
            self.gt = []
        if self.target is None:
            self.target = []
        if self.dropout_inp is None:
            self.dropout_inp = []
        _inp = np.zeros((inp.shape[0], inp.shape[1]+1))
        _inp[:-1,0] = dts[1:] - dts[:-1]
        _inp[:,1:] = inp
        self.inp.append(_inp)
        self.out.append(out)
        self.gt.append(gt)
        self.target.append(target)
        if self.dropout is not None:
            self.dropout_inp.append(dinp)
        
        
    def _make_dropout_matrix(self,obs,trange):
        
        oblen = obs.shape[1]
        mat = np.ones_like(obs,dtype=np.float32)
        if self.dropout is not None:
            ds = sorted(list(self.dropout.keys()))
            for d in ds:
                tslc = trange >= d
                if self.dropout[d] is False:
                    v = np.zeros(oblen)
                else:
                    v = np.array(self.dropout[d],dtype=np.float32)
                mat[tslc,:] = v
            mat[mat==0] = np.nan
       
        return mat
    def _get_sim_data(self,i):
        if self.cache_results:
            # Seed is already i
            seed = i
        else:
            seed = self.seeds[i]
        self.sim.reset(seed)
        
        dat = self.sim.run(self.trange, metrics=[ObservationRecorderMetric(),StateRecorderMetric()])
        # First column is time
        obs = dat[0][:,1:]
        states = dat[1][:,1:]
        dts = dat[0][:,0]
        n = obs.shape[0]
        obs = np.atleast_2d(obs[:,self.observation_slc]).reshape(n,-1)
        states = np.atleast_2d(states[:,self.state_slc]).reshape(n,-1)
        obsp = np.atleast_2d(obs[:, self.observation_passthrough_slc]).reshape(n,-1)
        print(obs.shape, states.shape, obsp.shape)
        return obs, obsp, states, dts
    @functools.lru_cache(maxsize=None)
    def _get_epoch_data(self,i):
        
            
        obs,obsp,states,dts = self._get_sim_data(i)
        
        # Make input from observation
        trange = range(obs.shape[0])
        l = len(trange)
        inp = None
        out = None
        target = None
        gt = None
        dinp = None
        passthrough_inp = None
        
        mat = self._make_dropout_matrix(obs,self.trange)
        for j in trange:
            # Get time sliced data
            ob = obs[j,:]
            
            state = states[j, :]
            observation_passthrough = obsp[j,:]

            # Make dropout input 
            if self.dropout is not None:
                dob = ob*mat[j,:]
                dinpob = self.in_act_fn(dob)
                if dinp is None:
                    try:
                        lendob = len(dinpob)
                    except TypeError:
                        lendob = 1
                    dinp = np.zeros((l,lendob))
                dinp[j,:] = dinpob


            # Make input
            inpob = self.in_act_fn(ob)
            
            if inp is None:
                try:
                    lenob = len(inpob)
                except TypeError:
                    lenob = 1
               
            
                inp = np.zeros((l,lenob))
            inp[j,:] = inpob

            # Make passthrough input
            if passthrough_inp is None:
                try:
                    lenobp = len(observation_passthrough)
                except TypeError:
                    lenobp = 1
                self._passthrough_length = lenobp
                passthrough_inp = np.zeros((l,lenobp))
            passthrough_inp[j,:] = observation_passthrough
            
            # Make target
            inptarget = self.in_target_act_fn(state)
            if target is None:
                try:
                    lent = len(inptarget)
                except TypeError:
                    lent = 1
                target = np.zeros((l,lent))
            target[j,:] = inptarget
            

            # Make output(target) from state
            fnout = self.out_act_fn(state)
            if out is None:
                try:
                    leno = len(fnout)
                except TypeError:
                    leno = 1
                out = np.zeros((l,leno))
            out[j,:] = fnout

            # Keep gt
            if gt is None:
                try:
                    lengt = len(state)
                except TypeError:
                    lengt = 1
                gt = np.zeros((l,lengt))
            gt[j,:] = state
        model = None
        if self.learn_mapping and not self.share_learning:
            inp,model = self._fit_model(dinp, inp, target)
            
        elif not self.learn_mapping and self.dropout is not None:
            inp = dinp
        return inp, dinp, out, target,gt, passthrough_inp,dts,model
  
    def _fit_model(self, dinp, inp, target):
        try:
            model = Lasso(positive=True, **self.solver_kwargs)
        except TypeError:
            raise ModuleNotFoundError('scikit-learn needs to be installed to optimize using the SimulatorDatasetGenerator')
        model.fit(y=target, X=inp)
        
        if dinp is not None:
            inp = model.predict(dinp)
        else:
            inp = model.predict(inp)
        return inp,model
    def _generate_and_set_epoch(self,i):
        if self.cache_results:
        
            inp, dinp, out, target,gt, passthrough_inp,dts,model = self._get_epoch_data(self.seeds[i])
        else:
            inp, dinp, out, target,gt, passthrough_inp,dts,model = self._get_epoch_data(i)
        if model is not None:
            self.transform_model[i] = model
        inp = np.concatenate((passthrough_inp, inp), axis=1)
        # No need to concat dinp since it was either used above or will be used below
        self._set_epoch_data(i, inp, out, target,gt, dts, dinp)
        return inp, dinp, out, target,gt, passthrough_inp,dts,model
    def _generate_data(self,progressbar):
        """
        Generates inp/out data and learns
        """
        learning=self.learn_mapping
        share_learning=self.share_learning
      
     
        
        
        for i in tqdm(range(self.n_epochs),desc='Dataset epochs generated', disable=not progressbar,total=self.n_epochs):
            inp, dinp, out, target, gt, passthrough_inp,dts,model = self._generate_and_set_epoch(i)
        if learning and share_learning:
            model = Lasso(positive=True, **self.solver_kwargs)
            tars = target.shape
            target = target.reshape((tars[0]*tars[1], tars[2]), order='F')
            inp = self.inp[:,:,1+self._passthrough_length:]
            inpp = self.inp[:,:,1:self._passthrough_length+1]
            dts = self.inp[:,:,0]
            inps = inp.shape
            inp = inp.reshape((inps[0]*inps[1], inps[2]), order='F')
            model.fit(y=target, X=inp)
            self.transform_model = model
            if self.dropout_inp is not None:
                self.dropout_inp = np.array(self.dropout_inp)
                dinp = self.dropout_inp.reshape((inps[0]*inps[1], inps[2]), order='F')
                inp = model.predict(dinp)
            else:
                inp = model.predict(inp)
            inp = inp.reshape((tars[0],tars[1], tars[2]), order='F')
            self.inp = np.concatenate((dts[:,:,None],inpp,inp),axis=2)
        


    def __len__(self):
        return self.n_epochs
    def __getitem__(self, idx):
        # Indexing desired velocities
        # return self.inp[idx], self.out[idx]
        if self.generate_as_needed:
            self._generate_and_set_epoch(idx)
        return tensor(self.inp[idx]), tensor(self.out[idx])
    def get_data(self):
        return self.inp, self.out, {'ground_truth_angle':self.gt, 'dropout_input':self.dropout_inp,}
class RaggedView():
    def __init__(self,dat):
        self.dat = dat
    def __getitem__(self,i):
        if len(i) == 2:
            return self.dat[i[0]][i[1]]
        return self.dat[i[0]][i[1:]]
    def __len__(self):
        return len(self.dat)
class CSVDatasetGenerator(SimulatorDatasetGenerator):
    """
    Dataset that uses a CSV with timestamped data to generate a dataset. 
    
    The CSV is expected to be formatted with a timestamp as its first column, and then data in the rest of the columns. All columns should be named. 
    
    The CSV is interpolated to a constant timestep (defined by dt), based on its starting and ending time.
    
    Like the SimulatorDataset, it uses activation functions to convert the data it receives into currents and neuron activation targets. 

    The in_act_fn operates on the columns defined by observation_columns, while out_act_fn operates on the columns defined by state_columns. If `learning_mapping` is true, activation is generated by `in_target_act_fn` (operating on the state) and a linear mapping is learned between `in_act_fn` and `in_target_act_fn`. This mapping is then used to generate input.
    
    observation_passthrough_columns get prepended to the input without any modification.
    """
    _passthrough = lambda x: x
    def __init__(self, 
            *args,
            csvs: Union[Iterable,str],
            rate_limit:Mapping[str, float]=None,
            dt: float=0.001,
            
            time_column: str='t',
            state_columns: Union[str, Iterable[str], None]='yaw',
            observation_columns: Union[str, Iterable[str],None]=None,
            observation_passthrough_columns: Union[str, Iterable[str],None]='ang_vel',
            time_slcs: Union[slice, Iterable[slice]]=None,
            use_nan_dropout:bool=True,
            **kwargs):
        """
        params:
        -------
        csvs: A list of filenames or a filename of a CSV to load. If a single
        filename, it is used for `epochs` times. Else, `epochs` is ignored and
        the length of the list of csvs is used as the number of epochs.

        rate_limit : A mapping of CSV column names to a rate limit. Any gap in
        data larger than this limit (in hz) will be present in the data. By
        default, any gap will be filled in all columns. 
        
        dt: The timestep to cast to. If none, no casting will be done and data will be fed raw.

        in_act_fn, in_target_act_fn: If learn_mapping is true, then a positive
        linear mapping is learned between `in_act_fn` and `in_target_act_fn`.
        Else, `in_act_fn` is used alone to determine input currents from the
        observation. `in_act_target_fn` acts on the state. If None, uses the
        `target_act_fn` instead.

        target_act_fn: Used to generate ground truth (i.e target) from state using
        `state_columns`

        learn_mapping: whether to learn a mapping for the output

        epochs: number of epochs. Ignored if `csvs` is a list or if None.

        state_columns: Columns in the CSV to use as state, and pass into the
        `out_act_fn` and `in_act_target_fn` to make a target.

        observation_columns: Columns in the CSV to use as the observation, and
        pass into the `in_act_fn`.

        observation_passthrough_slc: slice of the observation to not modify
        using the `in_act_fn`. Gets prepended to the model input.

        share_learning: whether the mapping should be learned per epoch
        (`share_learning=False`) or shared across all epochs.

        alpha, max_iter, fit_intercept: Parameters for Lasso solver, which is
        used when `learn_mapping=True`

        dropout: a mapping of times to lists of booleans, which "dropout"
        observations. The length of each list should be equivalent to the length
        of `observation_columns`

        time_slcs: 

        """
        self.tcol = time_column
        self.rate_limit = rate_limit or {}
        self.dt = dt
        self.use_nan_dropout = use_nan_dropout
        state_columns = state_columns or []
        observation_passthrough_columns = observation_passthrough_columns or []
        observation_columns = observation_columns or []
        self.time_slc = time_slcs or slice(0,None)
        
        try:
            if isinstance(time_slcs[0], (slice, list, np.ndarray)):
                # it is a list of slices
                n_epochs = len(time_slcs)
                if csvs is None or isinstance(csvs,(str)):

                    csvs = [csvs]*len(time_slcs)
                    self.cache_results = True
                else:
                    
                    if len(csvs) != n_epochs:
                        warnings.warn('Length of csvs does not match length of time slices')
                    
                    self.cache_results = False
            else:
                raise ValueError(f'Incorrect type for time slice iterable {type(time_slcs[0])} ')
        except:
            # Single time slice
            pass
        for unused_kwarg in ['sim','seeds','trange','state_slc','observation_slc','observation_passthrough_slc']:
            if kwargs.pop(unused_kwarg,None) is not None: 
                warnings.warn(f'{unused_kwarg} in kwargs, but unused in CSVDataset')
        super().__init__(
            trange=[],
            seeds=csvs, 
            sim=None, 
            state_slc=state_columns, 
            observation_slc=observation_columns,
            observation_passthrough_slc=observation_passthrough_columns, 
            **kwargs
        )
        
       
        
        
        
    def _make_dropout_matrix(self, obs,trange):
        # import pdb; pdb.set_trace()
        # Mat is nan where dropout, mat is now true where dropout happens
        mat = np.isnan(super()._make_dropout_matrix(obs,trange))
        # Mat is true where dropout happens in either observation or mat
        mat = np.bitwise_or(mat,np.isnan(obs))
        # Mat is 0 when dropout happens, 1 otherwise
        mat = (~mat).astype(np.float32)
        # Mat is nan when dropout
        mat[mat == 0] = np.nan

        return mat

        
        
    
    def _interp_data(self, dat):
        for coltype in [self.state_slc, self.observation_passthrough_slc, self.tcol]:
            if isinstance(coltype,list):
                dat = dat.dropna(subset=coltype)
            else:
                dat = dat.dropna(subset=[coltype])
        
        if self.dt is None:
            return dat
        dat[self.tcol] = (dat[self.tcol].round(3)/self.dt).astype(int)
        dat = dat.groupby(self.tcol).agg(np.nanmean).reset_index()
        dat = dat.set_index(self.tcol)
        
        bt = list(dat.index)[0]
        et = list(dat.index)[-1]
        interp_index = np.linspace(bt,et,int((et-bt))).astype(int)
        dat = dat.reindex(interp_index)
        dat = dat.set_index(dat.index*self.dt)
        dat = dat.sort_index()
        dat = dat.reset_index()

        for k,v in self.rate_limit.items():
            v = int((1./v)//self.dt)
            dat.loc[:,k] = dat[k].fillna(method='ffill', limit=v)
        for col in dat.columns:
            if col not in self.rate_limit:
                dat.loc[:,col] = dat[col].fillna(method='ffill')
        
        return dat
    def _get_sim_data(self, i):
        if self.cache_results:
            # i is seed already
            csv = i
            ti = self.seeds.index(i)
        else:
            csv = self.seeds[i]
            ti = i
        if isinstance(self.time_slc, slice):
            time_slc = self.time_slc
        else:
            time_slc = self.time_slc[ti]
            # Don't cache future results
            self.cache_results = False
        
        dat = pd.read_csv(csv)
        dat = dat.loc[time_slc,:]
        
        dat = self._interp_data(dat)
        # First column is time
        
        dts = np.array(dat[self.tcol])
        self.trange = dts-dts[0]
        n  = len(dts)
        obs = np.atleast_2d(dat.loc[:,self.observation_slc].to_numpy()).reshape((n,-1))
        obps = np.atleast_2d(dat.loc[:,self.observation_passthrough_slc].to_numpy()).reshape((n,-1))
        states = np.atleast_2d(dat.loc[:,self.state_slc].to_numpy()).reshape((n,-1))

        return obs, obps,states,dts 
    