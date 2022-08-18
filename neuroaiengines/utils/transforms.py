import numpy as np
from numpy import pi
import math
from scipy.stats import norm
from neuroaiengines.utils.angles import *
from typing import Iterable, Tuple, Dict
from itertools import product
def diag_block_mat_boolindex(a, sz):
    """
    Tiles the given array a number of times into the diagonals of a larger matrix

    parameters:
    -----------
    a: np.array(n,m)
        array to tile
    sz: int 
        the sz of the output matrix. output size will be (n*sz, m*sz)

    returns:
    ---------
    out: np.array(n*sz, m*sz)
        output matrix

    example:
    a = [
        [1,0], 
        [1,1]
        ]
    sz = 3
    out = diag_block_mat_boolindex(a, sz)
    >> out = [
        [1,0,0,0,0,0],
        [1,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,1,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,1,1]
        ]

    """
    shp = a.shape
    mask = np.kron(np.eye(sz), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*sz)
    out[mask] = np.concatenate(np.tile(a,(sz,1))).ravel()
    return out
def generate_middle_slice(osz,sz, end=True):
    """
    generates a continuous slice of size sz that is in the middle of osz.
    if osz and sz are differnt parity, then the slice will be either offset at the end (if end==True) or the beginning (if end==False)
    
    parameters:
    -----------
    osz: int
        original size
    sz: int
        desired size
    
    returns:
    --------
    ret: slice
        output slice of size sz
    
    example:
    --------
    a = [0,1,2,3,4]
    slc = generate_middle_slice(len(a), sz=3, end=True)
    a[slc]
    >>> [1,2,3]


    """
    diff = int((osz - sz)/2)
    # Check if differing parity
    if ((osz % 2) ^ (sz % 2)):
        if end:
            ret = slice(diff, osz-diff-1)
        else:
            ret = slice(diff+1, osz-diff)
    else:
        ret = slice(diff, osz-diff)
    return ret




def reorder_matrix(wn:np.array, slcs:Dict[str, slice], order:Iterable[str]) -> Tuple[np.array, Dict[str, slice]]:
    """
    Reorders a matrix based on slices and desired order

    arguments:
    ----------
    wn: matrix to reorder. Assumes the row and column order are the same
    slcs: mapping from string to index slices
    order: desired order of the mapping keys

    returns:
    -------
    new_wn: reordered matrix
    new_slcs: new slices with the same names
    """
    new_wn = np.zeros_like(wn)
    new_slcs = {}
    i = 0
    for k in order:
        l = len(wn[slcs[k], slcs[k]])
        new_slcs[k] = slice(i,i+l)
        i = i+l
    for (pre_k,pre_s),(post_k, post_s) in product(new_slcs.items(), new_slcs.items()):
        blk = wn[slcs[pre_k], slcs[post_k]]
        new_wn[pre_s, post_s] = blk
    return new_wn, new_slcs
