
import numpy as np
from numpy import SHIFT_UNDERFLOW, pi
from neuroaiengines.utils.angles import wrap_pi
import math
import functools
# pylint: disable=not-callable
from scipy.optimize import minimize

def _root_k(k, fwhm):
    c = np.log(np.cosh(k))/k

    return np.square(np.cos(fwhm/2) - c)

def _simp_vonmises(a,loc,k):
    x = np.exp(k*np.cos(a-loc))
    return x
def create_activation_fn(num_neurons=27, encoding_range=[-0.75*pi, 0.75*pi], fwhm=pi/18, scaling_factor=1,**kwargs):
    """
    Creates an activation function that returns the activation of num_neurons given a landmark angle.

    parameters:
    ----------
    :param num_neurons : the number of ring neurons
    :param encoding_range: the range of the encoding, [min, max]
    :param fwhm: full width, half max of the bump centered around each ring neuron's receptive field. In radians.
    :param scaling_factor: How much the final current is scaled.

    :returns activation_fn: function(ang, slc)
        an activation function that returns activations given landmark angles
        parameters:
        ----------
        angs: float
            angle
        slc: slice
            a slice to truncate the output to a certain number

    """
    r_min, r_max = encoding_range
    rng = r_max - r_min
    centers = create_preferred_angles(num_neurons,centered=True, rng=rng, **kwargs)
    # standard devation calculation from FWHM
    std = fwhm/(2*np.sqrt(2*np.log(2)))
    k = 1/np.sqrt(std)
    k = minimize(_root_k,x0=k,args=fwhm, bounds=[(0.1,np.inf)]).x
    maxx = np.exp(np.abs(k))
    minx = np.exp(-np.abs(k))
    rescale = lambda x: (x - minx)/(maxx-minx)
    # Just compute using k
    def activation_fn(ang, slc=slice(0,None)):
        
        ret = rescale(_simp_vonmises(centers,ang,k))
        
        
        return ret*scaling_factor
    activation_fn.k = k
    return activation_fn
def create_epg_bump_fn(num_neurons, scaling_factor=2,fwhm=pi/2, hemisphere_offset=False):
    if hemisphere_offset:
        
        fn = create_activation_fn(num_neurons*2, encoding_range=[-pi,pi], fwhm=fwhm, scaling_factor=scaling_factor, hemisphere_offset=hemisphere_offset)
    else:
        fn = create_activation_fn(num_neurons, encoding_range=[-pi,pi], fwhm=fwhm, scaling_factor=scaling_factor)
    def hemfn(ang):
        # We assume ang is single dimensional
        
        val = fn(ang)
        if not hemisphere_offset:
            return np.hstack((val,val))
        else:
            return val
    hemfn.k = fn.k
    return functools.cache(hemfn)

def create_tiled_bump_fn(num_neurons, bins_per_neuron=10,tile=2, **kwargs):
  
    fn = create_activation_fn(num_neurons, **kwargs)

    def hemfn(ang):
        val = fn(ang)
        return np.tile(val, (1,tile))
    hemfn.k = fn.k
    return hemfn

def create_pref_dirs(*args, **kwargs):
    """
    Create circular preferred directions in two dimenions (cos, sin)

    sz: int
        number of neurons to create the directions for. Output shape will be (sz,2)
    centered: bool
        If centered, the preferred directions will be centered on 0, such that the preferred direction of the sz/2 neuron is 0.
    rng: float
        The total range that the preferred directions cover.

    returns:
    --------
    pref_dirs: np.array((sz,2))
        Preferred directions of the neurons. Feed this into the ensemble's encoders
    """
    pref_angles = create_preferred_angles(*args, **kwargs)
    
    pref_dirs = np.array(
        [
            [np.cos(a), np.sin(a)]
            for a in pref_angles
        ]
    )
    return pref_dirs

def create_preferred_angles(sz, centered=False, rng=2*pi, hemisphere_offset=False):
    """Generates preferred angles for a ring of neurons

    Args:
        sz (int): number of neurons in the ring
        centered (bool, optional): If the preferred angles should be centered on 0. If true, preferred_angles[sz//2] ~= 0. Otherwise, preferred_angles[0] ~-0. Defaults to False.
        rng (float, optional): Range of encoding. Defaults to 2*pi. 
        hemisphere_offset (bool, optional): If true, indicates that the number of neurons includes two hemispheres and that the encoding should return in L/R order. 
            For example, if hemisphere_offset==False, the order of preferred_angles would be [0L,0R,1L,1R...], but if hemisphere_offset==True, the order would be [0L,1L,2L..0R,1R,2R...]. 
            Defaults to False.

    Returns:
        [type]: [description]
    """
    if centered:
        min_a = math.floor(sz/2)
        max_a = math.ceil(sz/2)
    else:
        min_a = 0
        max_a = sz
    pref_angs = np.array(
        [
            rng * (t + 1) / sz
            for t in np.arange(-min_a, max_a)
        ]
    )
    if hemisphere_offset:

        return unshuffle_ring(pref_angs)
    return pref_angs




def create_sine_epg_activation(sz, offset=1., scale=0.2):
    """
    Creates a sine-based EPG bump function based off preferred angles.
    
    parameters:
    ----------
    sz: size of the output
    offset,scale: offset and scale of the output, such that the bump will be between (offset-1)*scale and (offset+1)*scale.
    """
    epg_pref_dirs = create_pref_dirs(sz,centered=True)
    epg_pref_dirs = np.tile(epg_pref_dirs, (2,1))
    def fn(angle):
        """
        Gets the epg activation given an angle.
        parameters:
        ----------
        angle: float
            Angle in radians

        returns:
        -------
        activation: np.array(18)
            activation of the EPGs. Units are arbitrary.
        """

        
        # Double it for each hemisphere
        
        sc = np.array([np.cos(angle), np.sin(angle)])
        activation = np.dot(epg_pref_dirs, sc.T)
      
        return (activation+offset)*scale
    return fn

# For legacy code
get_epg_activation = create_sine_epg_activation(9, 0, 1)

def create_decoding_fn(sz, sincos=False, backend=np, hemisphere_offset=False):
    """

    Creates a function to return the angle given bumps of EPG activity
    """
    if not hemisphere_offset:
        epg_pref_dirs = create_pref_dirs(sz,centered=True)
        try:
            epg_pref_dirs = backend.tensor(epg_pref_dirs)
        except:
            pass  
        epg_pref_dirs = backend.tile(epg_pref_dirs, (2,1)).T
    else:
        epg_pref_dirs = create_pref_dirs(sz*2,centered=True, hemisphere_offset=hemisphere_offset).T
        try:
            epg_pref_dirs = backend.tensor(epg_pref_dirs)
        except:
            pass  
    def decode(act):
        sc = backend.matmul(epg_pref_dirs,act)
        if sincos:
            return sc
        return backend.arctan2(sc[1],sc[0])
    return decode
def shuffle_ring(a):
    """
    Shuffles the order of ring with two hemispheres from [0L,1L,2L...NL,0R,1R,2R...NR] to [0L,0R,1L,1R,2L,2R...NL,NR]

    Args:
        a (np.array): Array to be shuffled

    Returns:
        np.array: Shuffled array
    """

    return np.array(list(zip(a[:len(a)//2],a[len(a)//2:]))).ravel()
def unshuffle_ring(a):
    """
    Unshuffles the order of ring with two hemispheres from [0L,0R,1L,1R,2L,2R...NL,NR] to [0L,1L,2L...NL,0R,1R,2R...NR]

    Args:
        a (np.array): Array to be unshuffled

    Returns:
        np.array: Unshuffled array
    """

    return np.concatenate((a[::2],a[1::2]))
