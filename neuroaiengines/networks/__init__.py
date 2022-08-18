import numpy as np
import os
from typing import Iterable, Tuple
import pandas as pd
import re
from neuroaiengines.networks.offset_utils import gen_matrix_gauss, gen_matrix, radian_offset
from abc import abstractclassmethod, abstractmethod
import warnings
def parse_network_name(name):
        
    match = re.findall(r"(symmetric|hemibrain)(\d+)",name)[0]
    if match:
        return match[0],int(match[1])
    else:
        raise ValueError(f'{name} is not a valid network type')

class NetworkCreator():
    @classmethod
    @abstractmethod
    def create_network(cls, wd, **kwargs):
        """
        Creates a network with the desired number of wedges
        """
        pass
    @classmethod
    def create_slcs(cls,wd, order, granularity='combined', granularity_exclude=None):
        """
        Creates slices based on wedge number and order
        
        
        """
        if granularity_exclude is None:
            granularity_exclude = ['d7']
        try:
            npop = len(wd)
        except TypeError:
            wd = [wd]*len(order)
            npop = len(order)
        if npop !=len(order):
            raise ValueError(f'Invalid list of wedges {wd}')
        slcs = {}
        n = 0
        if granularity == 'combined':
            # Contra/ipilateral have combined connections i.e no differentiation between hemispheres
            for pop,_wd in zip(order,wd):
                slcs[pop] = slice(n,n+_wd)
                n = n+_wd
        elif granularity == 'separate':
            # Contra/ipsilateral connections all seperate
            for pop,_wd in zip(order,wd):
                if pop not in granularity_exclude:
                    slcs[pop + '_L'] = slice(n,n+_wd//2)
                    n += _wd//2
                    slcs[pop + '_R'] = slice(n,n+_wd//2)
                    n += _wd//2
                else:
                    slcs[pop] = slice(n,n+_wd)
                    n = n+_wd
        else:
            raise ValueError(f'Granularity {granularity} not supported')
        return slcs

class KakariaBivortCreator(NetworkCreator):
    
    @classmethod
    def create_network(cls,**kwargs):
        
        _w_fp = os.path.join(os.path.dirname(__file__), 'PBa.csv')
        w = np.loadtxt(_w_fp, delimiter=",")
        slcs = cls.create_slcs([16,16,18,10],['pen','peg','epg','d7'], **kwargs)
        
        return w, slcs
    
class SymmetricKBCreator(NetworkCreator):
    @classmethod
    def create_network(cls, wd : int,**kwargs) -> np.ndarray:
        """
        Creates a symmetric weight matrix for the ring attractor

        :param wedges: Number of wedges. Must be even.

        :return:
            np.ndarray[wedges*3.5, wedges*3.5]
            weight matrix sorted by P-ENs, P-EGs, E-PGs, Pintrs.

        """
        N = wd*2 + wd*2 + wd*2 + wd
        wn = np.zeros((N,N))
        pens = slice(0, wd*2)
        pegs = slice(wd*2, wd*4)
        epgs = slice(wd*4, wd*6)
        pintrs = slice(wd*6, wd*7)
        wpe = np.zeros((wd*2,wd*2))
    #     wpe[:wd,:wd] = np.roll(np.eye(wd),-1)
    #     wpe[wd:,wd:] = np.roll(np.eye(wd),1)
        wpe[:wd,:wd] = np.eye(wd)
        wpe[wd:,wd:] = np.eye(wd)
        wn[(pens, epgs)] = wpe*20

        wn[(pens, pintrs)] = np.tile(np.eye(wd),(2,1))*-15
        
        wn[(pegs, epgs)] = np.eye(wd*2)*20
        wn[(pegs, pintrs)] = np.tile(np.eye(wd),(2,1))*-15
        wep = np.eye(wd*2)
        wep[:wd,wd:] = np.eye(wd)
        wep[wd:,:wd] = np.eye(wd)
        wn[(epgs, pegs)] = wep*20
        wep = np.eye(wd*2)
        wep[:wd,wd:] = np.eye(wd)
        wep[wd:,:wd] = np.eye(wd)
        wep[:,wd:] = np.roll(wep[:,wd:],1,axis=1)
        wep[:,:wd] = np.roll(wep[:,:wd],-1,axis=1)
        wn[(epgs, pens)] = wep*20
        
        wn[(pintrs, epgs)] = np.ones((wd,wd*2))*20
        wn[(pintrs,pintrs)] = -20*(np.ones((wd,wd))-np.eye(wd))
        
        slcs = cls.create_slcs([wd*2, wd*2, wd*2, wd],['pen','peg','epg','d7'],**kwargs)
        return wn.T,slcs
class HemibrainCreator(NetworkCreator):
    @classmethod
    def create_network(cls,wd:int, threshold:float=2, gaussian:bool=False, symmetric:bool=True, both:bool=False, filename:str=None, angular:bool=False,null_offset:bool=False, clear_autapses:bool=True,**kwargs)->Tuple[np.ndarray, Iterable[slice]]:
        if filename is None:
            try:
                fn = {
                    True: 'hemibrain_conn_df_both.csv',
                    'a': 'hemibrain_conn_df_a.csv',
                    'b': 'hemibrain_conn_df_b.csv',
                    'both' : 'hemibrain_conn_df_both.csv',
                    False: 'hemibrain_conn_df_a.csv'
                }[both]
            except KeyError:
                fn = 'hemibrain_conn_df_a.csv',
        else:
            fn = filename
        fp = os.path.join(os.path.dirname(__file__),fn)
        try:
            conn_df = pd.read_csv(fp)
        except FileNotFoundError:
            print('Please download additional data with pull_data.py --pen-a --pen-b')
            raise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if gaussian:
                if angular:
                    wn = radian_offset(conn_df, wrap = True, both=both, synthetic_pintr=-7, threshold=threshold, symmetric=symmetric, resolution_per_wedge=wd*2)
                else:
                    if both in [True, 'both']:
                        gboth = True
                    else:
                        gboth = False
                    wn = gen_matrix_gauss(conn_df, wd*2, wrap=True, both=gboth, synthetic_pintr=-7,threshold=0,symmetric=symmetric,null_offset=null_offset)
            else:
                if angular:
                    raise ValueError('Cannot use angular matrix while gaussian is not set.')
                wn = gen_matrix(conn_df, wd*2, wrap=True, weighted=True, both=both, synthetic_pintr=-7,threshold=threshold,symmetric=symmetric)
        wn = np.nan_to_num(wn, nan=0)
        if both is True:
            
            
            slcs = cls.create_slcs([wd*2, wd*2, wd*2, wd*2, wd],['epg','pen','pen-b','peg','d7'],**kwargs)
        else:

            slcs = cls.create_slcs([wd*2, wd*2, wd*2, wd],['epg','pen','peg','d7'],**kwargs)
        if clear_autapses:

            wn = wn*(1-np.eye(len(wn)))
        len_slices = sum([slcs[x].stop - slcs[x].start for x in slcs.keys()])
        if len_slices != len(wn):
            raise ValueError("Slices and weight matrix don't match")
        return wn, slcs

get_kakaria_bivort = KakariaBivortCreator.create_network
create_symmetric_ra_matrix = SymmetricKBCreator.create_network
create_symmetric_hemi_ra_matrix = HemibrainCreator.create_network
