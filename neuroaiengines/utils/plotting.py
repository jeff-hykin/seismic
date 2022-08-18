import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os
import pickle
import time
import namegenerator
from warnings import warn
import cloudpickle
from copy import copy
from ipywidgets import interact
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from operator import itemgetter
from pprint import pprint
from neuroaiengines.utils.signals import *
from matplotlib.cm import get_cmap
import itertools
from neuroaiengines.optimization.torch import TBPTT
def absolute_error(gt_angle,est_angle):
    diff = gt_angle-est_angle
    diff=np.mod(diff,2*np.pi)
    diff[diff>np.pi] = -(2*np.pi-diff[diff>np.pi])
    return diff


def unwrap(gt,output):
    o = np.zeros(gt.shape)
    o = gt + absolute_error(gt, output)

    return o
def filter_spikes(data, sz=4):
    data = data.copy().T
    out = np.zeros(data.shape)
    for i, row in enumerate(data):
        out[i,:] = gaussian_filter(row, sz)

    return out
def plot_matrix(w,slcs,ax=None):
    indices = np.arange(len(w))
    labellocs = [(k,np.mean(indices[slc])) for (k,slc) in slcs.items()]
    if ax is None:
        _,ax = plt.subplots()
    ax.imshow(w, aspect='auto')
    labels,locs = zip(*labellocs)
    ax.set_xticks(locs)
    ax.set_yticks(locs)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    return ax

def relabel_nodes(G,slcs,delim='-',zerod=False):
    """
    Relabels nodes in a graph  based on a slice dictionary.

    :param G: the graph
    :param slcs: mapping between population name and index slice

    G = Graph()
    G.add_from(range(5))
    slcs = {'pen':slice(0,3),'epg':slice(3,5)}
    relabel_nodes(G,slcs)
    >> {0: "pen-0", 1:"pen-1", 2:"pen-2", 3:"epg-1", 4:"epg-2"}    
    """
    nodes = list(G.nodes())
    remap = {}
    start = 0 if zerod else 1
    for k,slc in slcs.items():
        for i,n in enumerate(start,nodes[slc]):
            k[n] = str(k) + delim + str(i)
    return remap

# For backwards compat
create_epg_activation1 = create_sine_epg_activation
import subprocess   
from typing import Iterable, Tuple
def grab_results(host : str, folder : str, ray_results:str ='~/ray_results', dest:str ='.') -> str:
    """
    Gets results from an external host that ran a ray tune
    params:
    -------
    host: the external host
    folder: folder within the ray_results directory
    ray_results: where all ray_results are stored
    dest: local destination to rsync to
    returns:
    -------
    path of the folder w.r.t dest.
    """
    pth = os.path.join(ray_results, folder)
    hoststr = f"{host}:{pth}"
    subprocess.run(['rsync', '-r', hoststr, dest], check=True)
    return os.path.join(dest, folder)
def resolve_label(ll: Iterable[Tuple[str, str]]) -> str:
    """
    Given a list of labels of pre/post synaptic populations, i.e [('EPG', 'PEG'), ('EPG', 'PEN')], returns a combined label,
    e.g 'EPG->PEG+PEN'.
    params:
    -------
    ll: list of tuple of labels
    returns:
    --------
    combined label
    """

    
    pre,post = list(zip(*[l.split('_') for l in ll]))
    l = ""
    to_remove = []
    lmb = lambda x: str(x).upper()
    for pr,po in itertools.groupby(zip(pre,post),itemgetter(0)):
        po = [p for p in list(zip(*po))[1]]
        
        if len(po) > 1:
            if len(l) != 0:
                l += " + "
            l += pr.upper()
            l += '->'
            
            l+= f"({', '.join(map(lmb, po))})"
            for p in po:
                
                to_remove.append(p)
    pre = list(pre)
    post = list(post)
    for i in to_remove:
        ii = post.index(i)
        pre.pop(ii)
        post.remove(i)
    
    for pr,po in itertools.groupby(zip(post,pre),itemgetter(0)):
        po = [p for p in list(zip(*po))[1]]
        
        if len(l) != 0:
            l += " + "
        
        if len(po) > 1:
            l+= f"({', '.join(map(lmb, po))})"
        else:
            l += lmb(po[0])
        l += '->'
        l += pr.upper()
        
    return l
def get_label_color(label:str, palette:str='tab20') -> np.ndarray:
    """
    Gets consistent colors for ring attractor populations.
    params:
    -------
    label: pre_post label for a synapse type, e.g pen_d7
    palette: one of the matplotlib palettes.
    returns:
    --------
    r,g,b of the color
    """
    labels = ["_".join(l) for l in itertools.product(['d7','epg','pen','peg'],repeat=2)]
    lspace = np.linspace(0,1,len(labels))
    d = dict(zip(labels,lspace))
    return get_cmap(palette)(d[label])
def plot_losses(ax,trainer:TBPTT):
    """
    Plots the losses of a trainer over epochs
    params:
    -------
    ax: matplotlib axis to plot on
    trainer: TBPTT object
    """
    ax.plot(trainer.mean_losses, color='tab:orange')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch #')
def plot_weights(ax, trainer, drop=['bias','v_scaling']):
    plotparams = trainer.parameters.copy()
    try:
        for d in drop:
            plotparams = plotparams.drop(d,axis=1)
    except:
        pass
    cols = {str(c) for c in plotparams.columns}
    while cols:
        c = cols.pop()
        label = [str(c)]
        ccols = copy(cols)
        while ccols:
            cc = ccols.pop()
            cd = plotparams[c].to_numpy(dtype=float)
            ccd = plotparams[cc].to_numpy(dtype=float)
            mn = np.mean(np.isclose(cd,ccd,rtol=0.1).astype(float))
            if mn > 0.9:
                cols.remove(cc)
                label.append(str(cc))
        label = resolve_label(label)
        try:
            lcolor = get_label_color(c)
        except KeyError:
            lcolor = 'k'
        ax.plot(plotparams[c], label=label,color=lcolor)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Epoch #')
    if 'bias' in trainer.parameters:
        
        ax.plot(trainer.parameters['bias'], label='bias', color='k')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='xx-small')
def break_hemi_slc(slc,total_len):
    """Breaks a slice in two"""
    l = len(np.arange(total_len)[slc])//2
    return (slice(slc.start, slc.start + l),slice(slc.start + l, slc.stop))
def _get_epg_slc(trainer, which):
    total_n = len(trainer.model.w)
    epg = trainer.model.slcs['epg']
    epg_L,epg_R = break_hemi_slc(epg, total_n)
    slc = {
        'both' : epg,
        'l' : epg_L,
        'r' : epg_R,
    }[which.lower()]
    
    return slc
def plot_states_time(ax,trainer,ret,imshow=True, which='both'):
    slc = _get_epg_slc(trainer, which)
    ax.set_xlabel('Timsteps')
    if imshow:
        im = ax.imshow(ret['states'][:,slc].T, aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation')
        ax.set_ylabel('EPG #')
    else:
        ax.plot(ret['states'][:,slc])
        ax.set_ylabel('Activation')
def plot_single_bump(ax,trainer,ret,bump_fn,rad_test, which='both',disp_degree=True):
    
    
    slc = _get_epg_slc(trainer,which)
    n = len(trainer.model.w[slc,slc])
    total_n = len(trainer.model.w)
    epg = trainer.model.slcs['epg']
    epg_L,epg_R = break_hemi_slc(epg, total_n)

    x = create_preferred_angles(n, centered=True)
    gt = rad_test.out[-1,-1,:]
    ax.set_xlabel('Encoding angle')
    ax.set_xticks([-pi,-pi/2, 0, pi/2,pi])
    ax.set_xlim([-pi-0.1,pi+0.1])
    if disp_degree:
        ax.set_xticklabels(["-$180\degree$","$-90\degree$","$0\degree$","$90\degree$","$180\degree$"])
    else:
        ax.set_xticklabels(["-$\pi$","-$\pi$/2","0","$\pi/2$","$\pi$"])
    
    ln_max = ret['states'].shape[0]-1
    ax.set_ylabel('Activation')
    tloss = np.round(ret['final_loss'],4)
    ax.set_title(f'Final loss: {tloss}', fontdict=dict(fontsize='small'))
#     ax2 = axs[3].twinx()
    which = which.lower()
    l1,l2 = None,None
    if which == 'both':
        l1 = ax.plot(x,ret['states'][-1,epg_L])[0]
        l2 = ax.plot(x,ret['states'][-1,epg_R],ls='--')[0]
        
    elif which=='l':
        l1 = ax.plot(x,ret['states'][-1,epg_L])[0]
    elif which=='r':
        l1 = ax.plot(x,ret['states'][-1,epg_R])[0]
    gt_l = ax.plot(x,gt[:n], ls='--', color='k')

    ax.set_ylabel('Activation')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    return l1,l2,gt_l

def plot_data(trainer,ret,rad_test, args,k, inset=False, interact=False):
#     fig, axs = plt.subplots(2,2, figsize=(12,6),dpi=300)
    pprint(args)
    fig, axs = plt.subplots(2,2)
    axs = axs.ravel()

    bump_fn = args['test_kwargs']['bump_fn']

    if bump_fn is None:
        bump_fn_kwargs = args['test_kwargs'].get('bump_fn_kwargs')
        bump_fn = create_epg_bump_fn(trainer.model, **bump_fn_kwargs)
    else:
        try:
            nwedge = int(k.split('-')[1])
        except:
            nwedge = 9
            
            
        bump_fn = create_epg_activation1(nwedge)
    
    
    params = trainer.parameters.iloc[-1].to_dict()

    plot_losses(axs[0],trainer)
    
    plot_weights(axs[1], trainer)
    
    plot_states_time(axs[2],trainer,ret)
    
    
    
    
    
    
    
    if inset:
        iax = axs[2].inset_axes(bounds=[0.3,0.6,0.4,0.3])
        iax.plot(np.arange(990,1000),ret['states'][-10:,trainer.model.slcs['epg']])
        iax.set_xlim(990,1000)
        iax.set_xticks([990,1000])
        iax.set_yticks([])
        axs[2].indicate_inset_zoom(iax, edgecolor="black")
    else:
        ts = ret['states'][-1,trainer.model.slcs['epg']]
        minn, maxn = np.min(ts),np.max(ts)
        rect,lines = axs[2].indicate_inset(bounds=[1000,minn,0.01,maxn], inset_ax=axs[3], edgecolor="black", )
    
    for con in lines:
        con.set_in_layout(False)
    epg_L = slice(trainer.model.slcs['epg'].start, trainer.model.slcs['epg'].start+nwedge)
    
    fig.suptitle(k,x=0.5, y=1.05)
#     rmse = 0
#     loss = 0
#     n_steps = 500
#     for i in range(-n_steps,0):
#         state = tensor(ret['states'][i,v.model.slcs['epg']])
#         tgt = tensor(gt)
#         rmse += nn.MSELoss()(tgt,state)
#         loss += args['train_kwargs']['loss_fn'](tgt,state)
#     rmse_bars[k] = rmse/n_steps
#     loss_bars[k] = loss/n_steps
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_ylabel('Loss')
    plt.tight_layout()
    # if interact:
    #     def update_act(i):
    #         i = int(i)
    #         l_L.set_ydata(ret['states'][i,epg_L])
    #         l_R.set_ydata(ret['states'][i,epg_R])
    #         l2_L.set_ydata(ret['states'][i,pen_L])
    #         l2_R.set_ydata(ret['states'][i,pen_R])
    #         j = i if i < 1000 else 999
    #         gt_l.set_ydata(rad_test.out[-1,j,:nwedge])
    #         rect.set_x(i)
    #         ts = ret['states'][i,v.model.slcs['epg']]
    # #         minn, maxn = np.min(ts),np.max(ts)
    #         minn, maxn = -0.5,nwedge
    #         rect.set_y(minn)
    #         rect.set_height(maxn-minn)
    #         for i,li in enumerate(lines):
    #             if li.get_visible():
    #                 print(i)
    #                 if i == 0:
    #                     li.xy2 = rect.xy
    #                 elif i == 2:
    #                     li.xy2 = (rect.xy[0] + rect.get_width()-0.5, rect.xy[1] + rect.get_height()-0.5)
    #     interact(update_act, i=widgets.FloatSlider(value=ln_max,min=0,max=ln_max,step=1))
    #     interact_manual(_inp_ot, t='EPG')
    return fig