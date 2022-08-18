import numpy as np
import pandas as pd
from scipy.stats import norm, mode
from scipy import interpolate
import networkx as nx

def draw_hists(df, pre_type, post_type, indx, hemi_pre="L", fix=True, diffs=False, wrap = True, weighted=True, ct = 1, sub_ind = 8, fs =(15,5)):
    """
    Function to draw histograms of {pre_type}_{hemisphere} to {post_type} (both hemispheres)
    Arguments:
        df (Pandas DataFrame): Formatted dataframe output from process_hemibrain function
        pre_type (string): "EPG" "PEN" or "PEG" presynaptic type
        post_type (string): "EPG" "PEN" or "PEG" postsynaptic type
        indx (string): Dataframe index to be used for plotting. If using rewrapped labels, use
            "index_fix_post", otherwise use "index_post"
        hemi_pre (string): "L" or "R"
        diffs (boolean): True to plot difference between pre_index and post_index, False to plot raw values
        wrap (boolean): True to wrap a difference of +/- 5 or greater to other side
        weighted (boolean): Whether to weight histogram by synaptic weight 
        ct (int): index to start at
        sub_ind(int): integer to consider the "wrapping maximum" ie: 
            if wrap==True and diff_ij = -5 -> wrapped_value = sub_ind + diff_ij
    Returns:
        None: Plots histograms in matplotlib
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(2,4,figsize = fs)

    df[indx] = df[indx].astype(int)
    
    # By default creates a 2x4 subplot - 1-4 is ct + 0 -> ct + 3, 5-8 is ct + 4 -> ct + 7
    for i in range(2):
        for j in range(4):
            
            df["intidx"] = df[indx].astype(int)
            # Creating filtered DataFrame
            fdf = df[df.type_pre == pre_type]

            fdf=fdf[fdf["hemisphere_pre"]==hemi_pre]
            fdf = fdf[fdf["type_post"]==post_type]
            if fix:
                fdf = fdf[fdf["index_fix_pre"]==ct]
            else:
                fdf = fdf[fdf["index_pre"]==ct]

            fdf = fdf[fdf.hemisphere_pre == hemi_pre]
            # Creating difference values for the difference between pre_index and post_index
            # This subtracts the current wedge index from each post_index
            vals = np.array(fdf["intidx"]) - ct
            print(np.mean(np.array(fdf["intidx"])), ct)
            # Wrapping code using masking
            if wrap:
                vals[vals < -4] = sub_ind + vals[vals < -4]
                vals[vals > 4] = -sub_ind + vals[vals > 4]
                real_vals = vals
            # If no wrap, only include values within middle range that doesn't overflow
            else:
                fdf["mask"] = vals
                vals = vals[vals > -5]
                real_vals = vals[vals < 5]
                fdf = fdf[(fdf["mask"] > -5) & (fdf["mask"] < 5)]

            fdf["diffs"] = real_vals
            fdf = fdf.sort_values(by="intidx")
            # Filtered dfs to plot
            hemi_r = fdf[fdf["hemisphere_post"]=="R"]
            hemi_l = fdf[fdf["hemisphere_post"]=="L"]
            
            if diffs:
                to_plot = "diffs"
                bins=[-5,5]
            else:
                to_plot = "intidx"
                bins=[0,10]

            if weighted:
                sns.histplot(hemi_l, x=to_plot, color='b',ax=ax[i,j],label="L",binwidth=1, binrange=bins, weights="weight")
                sns.histplot(hemi_r, x=to_plot, color='g',ax=ax[i,j],label="R",binwidth=1, binrange=bins, weights="weight")
            else:
                sns.histplot(hemi_l, x=to_plot, color='b',ax=ax[i,j],label="L",binwidth=1, binrange=bins)
                sns.histplot(hemi_r, x=to_plot, color='g',ax=ax[i,j],label="R",binwidth=1, binrange=bins)
            ax[i,j].legend()
            ax[i,j].set_title(f"{pre_type} {ct}")

            ct += 1
        plt.suptitle(f"{pre_type} {hemi_pre} to {post_type}")
    fig.tight_layout()

def hists_collapsed(df, pre_type, post_type, indx, hemi_pre="L", diffs=False, wrap = True, ct = 1, sub_ind = 8):
    """
    Function to draw histograms of {pre_type}_{hemisphere} to {post_type} (both hemispheres)
    Arguments:
        df (Pandas DataFrame): Formatted dataframe output from process_hemibrain function
        pre_type (string): "EPG" "PEN" or "PEG" presynaptic type
        post_type (string): "EPG" "PEN" or "PEG" postsynaptic type
        indx (string): Dataframe index to be used for plotting. If using rewrapped labels, use
            "index_fix_post", otherwise use "index_post"
        hemi_pre (string): "L" or "R"
        diffs (boolean): True to plot difference between pre_index and post_index, False to plot raw values
        wrap (boolean): True to wrap a difference of +/- 5 or greater to other side
        weighted (boolean): Whether to weight histogram by synaptic weight 
        ct (int): index to start at
        sub_ind(int): integer to consider the "wrapping maximum" ie: 
            if wrap==True and diff_ij = -5 -> wrapped_value = sub_ind + diff_ij
    Returns:
        None: Plots histograms in matplotlib
    """

    df[indx] = df[indx].astype(int)
    df_r = pd.DataFrame()
    df_l = pd.DataFrame()
    
    # By default creates a 2x4 subplot - 1-4 is ct + 0 -> ct + 3, 5-8 is ct + 4 -> ct + 7
    for i in range(2):
        for j in range(4):
            
            df["intidx"] = df[indx].astype(int)
            # Creating filtered DataFrame
            fdf = df[df.type_pre == pre_type]
            fdf=fdf[fdf["hemisphere_pre"]==hemi_pre]
            fdf = fdf[fdf["type_post"]==post_type]
            fdf = fdf[fdf["index_fix_pre"]==ct]
            fdf = fdf[fdf.hemisphere_pre == hemi_pre]
            
            # Creating difference values for the difference between pre_index and post_index
            # This subtracts the current wedge index from each post_index
            vals = np.array(fdf["intidx"]) - ct
            
            # Wrapping code using masking
            if wrap:
                vals[vals < -4] = sub_ind + vals[vals < -4]
                vals[vals > 4] = -sub_ind + vals[vals > 4]
                real_vals = vals
            # If no wrap, only include values within middle range that doesn't overflow
            else:
                fdf["mask"] = vals
                vals = vals[vals > -5]
                real_vals = vals[vals < 5]
                fdf = fdf[(fdf["mask"] > -5) & (fdf["mask"] < 5)]

            fdf["diffs"] = real_vals
            fdf = fdf.sort_values(by="intidx")
            # Filtered dfs to plot
            hemi_r = fdf[fdf["hemisphere_post"]=="R"]
            hemi_l = fdf[fdf["hemisphere_post"]=="L"]

            df_r = pd.concat((df_r,hemi_r))
            df_l = pd.concat((df_l,hemi_l))

            ct += 1
    return df_r, df_l

def parse_instance(r, i=0):
    """
    Filtering function written by Raph to separate pandas df items into left and right
    Arguments:
        r (Pandas DataFrame): The df to process
    Returns:
        None: used as function for use in df.apply()
    """
    import re

    m = re.findall('(R|L)(\d)', r.instance)
    
    if m:
        try:
            return m[0][i]
        except:
            return np.nan
    return None

def process_hemibrain(token, both_pen = "both", si1 = 9, si2 = 9, si3 = 9, which_hemi="L", epg_special=False, pen_rewrap=True, peg_rewrap=False, only_EBPB=False, connection_threshold=0):
    """
    Function to pull adjacency data for EPG, PEN, PEG and filter and wrap it according to the wedge structure
    Much of the original pulling code is adapted from Raph

    Arguments:
        token (string): User's neuprint token
        both_pen (string): "a" to pull PEN_a, "b" to pull PEN_b, "both" to pull both
        si1, si2, si3 (ints): The subtraction indices for wrapping- sets the upper and lower bound for 
            the offset differences by type. Variables for EPG, PEN, PEG respectively
        pen_rewrap, peg_rewrap (Booleans): Flags to indicate whether or not to subtract 1 from PEN and PEG  indices

    Returns:
        None: Plots histograms in matplotlib
    """
    from neuprint import fetch_adjacencies,merge_neuron_properties, NeuronCriteria as NC, SynapseCriteria as SC
    import neuprint

    client = neuprint.Client('https://neuprint.janelia.org', token=token, dataset='hemibrain:v1.2.1')
    
    # PEN selection
    if both_pen == "both" or both_pen == "add":
        order = ['PEN_a(PEN1)', 'PEN_b(PEN2)', 'PEG', 'EPG', 'Delta7']
    elif both_pen == "b":
        order = ['PEN_b(PEN2)', 'PEG', 'EPG', 'Delta7']
    elif both_pen == "a":
        order = ['PEN_a(PEN1)', 'PEG', 'EPG', 'Delta7']

    # Make a neuron criteria object

    neuron_criteria = NC(
        status='Traced', # Only traced neurons
        type=order,  
        cropped=False, # No cropped neurons
        inputRois=['EB','PB'], # Defines input regions of interest 
        outputRois=['EB','PB'], # Defines output regions of interest 
        roi_req='any' # Neurons that begin or end in either ROI are selected 

    )
    # Get adjacency dataframes for neurons that match the criteria (with themselves)
    if only_EBPB:
        neuron_df, conn_df = fetch_adjacencies(neuron_criteria, neuron_criteria, min_total_weight=connection_threshold, rois=["EB","PB"])
    else:
        neuron_df, conn_df = fetch_adjacencies(neuron_criteria, neuron_criteria, min_total_weight=connection_threshold)

    # Preprocessing the connection df
    # Merge PEN types
    if both_pen == "both":
        c1 = neuron_df.type.isin(['PEN_a(PEN1)'])
        c2 = neuron_df.type.isin(['PEN_b(PEN2)'])
        neuron_df.loc[c1, 'type'] = 'PENa'
        neuron_df.loc[c2, 'type'] = 'PENb'
    elif both_pen == "b":
        c1 = neuron_df.type.isin(['PEN_b(PEN2)'])
        neuron_df.loc[c1, 'type'] = 'PEN'   
    elif both_pen == "a":
        c1 = neuron_df.type.isin(['PEN_a(PEN1)'])
        neuron_df.loc[c1, 'type'] = 'PEN'
    else:
        c1 = neuron_df.type.isin(['PEN_a(PEN1)', 'PEN_b(PEN2)'])
        neuron_df.loc[c1, 'type'] = 'PEN' 

    # Merge Pintr types
    c1 = neuron_df.type.isin(['P6-8P9', 'Delta7'])
    neuron_df.loc[c1, 'type'] = 'Pintr' 
    # Order types to be same as Kakaria/Bivort
    if both_pen == 'both':
        norder = ['PENa', 'PENb', 'PEG', 'EPG', 'Pintr']
    else:
        norder = ['PEN', 'PEG', 'EPG', 'Pintr']
    norder_map = dict(zip(norder,range(len(norder))))
    neuron_df['order'] = neuron_df.apply(lambda r: norder_map[r.type], axis=1)
    # Determine hemisphere/index from instance
   
    neuron_df['index'] = neuron_df.apply(parse_instance,args=(1,), axis=1)
    neuron_df['hemisphere'] = neuron_df.apply(parse_instance,args=(0,), axis=1)
    neuron_df['index'] = neuron_df['index'].astype(int)

    # Subtract 1 from PEN or PEG indices if desired
    if pen_rewrap:
        neuron_df.loc[(neuron_df.type == "PEN"), 'index'] = neuron_df.loc[(neuron_df.type == "PEN"), 'index'] - 1
        neuron_df.loc[(neuron_df.type == "PENa"), 'index'] = neuron_df.loc[(neuron_df.type == "PENa"), 'index'] - 1
        neuron_df.loc[(neuron_df.type == "PENb"), 'index'] = neuron_df.loc[(neuron_df.type == "PENb"), 'index'] - 1
    if peg_rewrap:
        neuron_df.loc[(neuron_df.type == "PEG"), 'index'] = neuron_df.loc[(neuron_df.type == "PEG"), 'index'] - 1
    
    index_fix = []
    # find the wrapped index of each of the left sided neurons - this reflects the difference in Kakaria and Hemibrain indexing
    for i in range(neuron_df.shape[0]):
        if neuron_df["hemisphere"].iloc[i] == which_hemi:
            if neuron_df["type"].iloc[i] == "EPG":
                if si1 < 0:
                    index_fix.append(int(neuron_df["index"].iloc[i]))
                else:
                    if epg_special and int(neuron_df["index"].iloc[i]) == 1:
                        index_fix.append(int(neuron_df["index"].iloc[i]))
                    else:
                        index_fix.append(si1 - int(neuron_df["index"].iloc[i]))
            elif neuron_df["type"].iloc[i][:3] == "PEN":
                if si2 < 0:
                    index_fix.append(int(neuron_df["index"].iloc[i]))
                else:
                    index_fix.append(si2 - int(neuron_df["index"].iloc[i]))
            else:
                if si3 < 0:
                    index_fix.append(int(neuron_df["index"].iloc[i]))
                else:
                    index_fix.append(si3 - int(neuron_df["index"].iloc[i]))

        else:
            index_fix.append(int(neuron_df["index"].iloc[i]))
            
    neuron_df["index_fix"] = index_fix
    conn_df = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance', 'order', 'hemisphere','index', "index_fix"])
    # Merge in neuron properties into the connenction dataframe such that each connection shows 
    # both the presynaptic type/instance and the postsynaptic type/instance
    conn_df = conn_df[np.logical_not(conn_df["type_post"] == "Pintr")]
    conn_df = conn_df[np.logical_not(conn_df["type_pre"] == "Pintr")]


    return neuron_df, conn_df

def gen_matrix(conn_df, n_of_each = 16, wrap = True, weighted=False, both=False, synthetic_pintr=0, threshold=0, symmetric=False):
    """
    Create connectivity matrix from hemibrain wedge centrality
    Arguments:
        conn_df (Pandas Dataframe): Synapse level connectivity matrix
        n_of_each (int): number of total wedges per type to output
        wrap (Boolean): Whether or not to wrap +- 5 or greater wedge differences
        weighted (Boolean): True yields the count of synapses for each wedge connection, False will put in just the offset
        both (Boolean): True if both PENa and PENb are included
        synthetic_pintr (float): Flag and weight to be the weight of the pintr connections
    Returns:
        Numpy matrix of connectivity
    """
    n_epgs_per_hemi = n_of_each
    n_pens_per_hemi = n_of_each
    n_pegs_per_hemi = n_of_each
    n_hemi_wedges = 16

    LR = set(["L", "R"])

    if both:
        total_wedges = n_epgs_per_hemi + n_pens_per_hemi + n_pegs_per_hemi + n_pens_per_hemi
    else:
        total_wedges = n_epgs_per_hemi + n_pens_per_hemi + n_pegs_per_hemi

    wedge_matrix = np.full((total_wedges, total_wedges), np.nan)
    if both:
        pre_types = ["EPG", "PENa", "PENb", "PEG"]
        post_types = ["EPG", "PENa", "PENb", "PEG"]
    else:
        pre_types = ["EPG", "PEN", "PEG"]
        post_types = ["EPG", "PEN", "PEG"]
    # iterate through each pre_type, post_type, and hemisphere
    for n1, prtype in enumerate(pre_types):
        for n2, potype in enumerate(post_types):
            for hemi_pre in ["L", "R"]:
                if prtype[:3] == "PEN" and potype == "PEG":
                    continue
                if prtype[:3] == "PEG" and potype == "PENa":
                    continue
                if prtype[:3] == "PEG" and potype == "PEG":
                    continue
                if prtype[:3] == "PEN" and potype[:3]  == "PEN":
                    continue
                if prtype[:3] == "PEG" and potype == "EPG":
                    continue
                # Return hists yields the average centrality between wedge {i} and each post index
                l, r, _, _, _, _, num_l, num_r = return_hists_full(conn_df, prtype, potype, "index_fix_post", hemi_pre, True, wrap, weighted=True)

                # Conditions to omit connections that do not exist
                # adl and adr are the offset, and wl and wr are the weights
                if np.isnan(l) or np.isnan(r):
                    continue
                
                l = int(round(l))
                wl = num_l / (n_hemi_wedges // 2)
                adl = l

                r = int(round(r))
                wr = num_r / (n_hemi_wedges // 2)

                adr = r

                if symmetric:
                    other_hemi = LR.difference(hemi_pre).pop()
                    l2, r2, _, _, _, _, num_l2, num_r2 = return_hists_full(conn_df, prtype, potype, "index_fix_post", other_hemi, True, wrap, weighted=True)
                    wl2 = num_l2 / (n_hemi_wedges // 2)
                    wr2 = num_r2 / (n_hemi_wedges // 2)
                    wr = (wr + wl2) / 2
                    wl = (wl + wr2) / 2 

                weil = wl  
                weir = wr
                if not weighted:
                    wr=1
                    wl=1
                if weighted:
                    adl=1
                    adr=1
                for i in range(n_epgs_per_hemi//2):

                    for hemi_post in ["L", "R"]:
                        if hemi_post == "L" and weil < threshold:
                            continue
                        if hemi_post == "R" and weir < threshold:
                            continue
                        # These conditions ensure each index is placed correctly within the resulting weight matrix given its pre hemi, post hemi, and index
                        if hemi_pre == "L":
                            if hemi_post == "L":
                                if i + l < 0:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + i + l] = adl * wl
                                elif i + l >= n_epgs_per_hemi//2:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + i + l] = adl * wl
                                else:
                                    wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + i + l] = adl * wl
                            else:
                                if i + r < 0:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + i + r + n_epgs_per_hemi//2] = adr * wr
                                elif i + r >= n_epgs_per_hemi//2:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + i + r + n_epgs_per_hemi//2] = adr * wr
                                else:
                                    wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + i + r + n_epgs_per_hemi//2] = adr * wr

                        else:
                            if hemi_post == "L":
                                if i + l < 0:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + i + l] = adl * wl

                                elif i + l >= n_epgs_per_hemi//2:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + i + l] = adl * wl
                                else:
                                    wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + i + l] = adl * wl
                            else:
                                if i + r < 0:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + i + r + n_epgs_per_hemi//2] = adr * wr

                                elif i + r >= n_epgs_per_hemi//2:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + i + r + n_epgs_per_hemi//2] = adr * wr
                                else:
                                    wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + i + r + n_epgs_per_hemi//2] = adr * wr
    # Inhibition purely from Kakaria
    if not synthetic_pintr == 0:
        new_wedge_matrix = np.full((total_wedges + n_of_each // 2, total_wedges + n_of_each // 2), np.nan)
        new_wedge_matrix[:-n_of_each // 2, :-n_of_each // 2] = wedge_matrix
        new_wedge_matrix[:n_of_each, -n_of_each // 2:] = synthetic_pintr
        new_wedge_matrix[-n_of_each//2:, -n_of_each // 2:] = synthetic_pintr
        for i in range(n_of_each // 2):
            # PINTR -> PEG
            new_wedge_matrix[-i-1, i + 2*n_epgs_per_hemi] = synthetic_pintr
            new_wedge_matrix[-i-1, i + 2*n_epgs_per_hemi + n_epgs_per_hemi//2] = synthetic_pintr
            # PINTR -> PEN
            new_wedge_matrix[-i-1, i + n_epgs_per_hemi] = synthetic_pintr
            new_wedge_matrix[-i-1, i + n_epgs_per_hemi + n_epgs_per_hemi//2] = synthetic_pintr
            # PINTR -> PENb
            if both:
                new_wedge_matrix[-i-1, i + 3*n_epgs_per_hemi] = synthetic_pintr
                new_wedge_matrix[-i-1, i + 3*n_epgs_per_hemi + n_epgs_per_hemi//2] = synthetic_pintr            
            # PINTR -> PINTR
            new_wedge_matrix[-i-1, -n_of_each//2 + i] = np.nan
        wedge_matrix = new_wedge_matrix
    return wedge_matrix

def return_hists(df, pre_type, post_type, indx, hemi_pre="L", diffs=False, wrap = True, wedge_idx = 1, sub_ind = 8, centrality="mean"):
    """
    Function to return histograms of {pre_type}_{hemisphere} to {post_type} (both hemispheres)
    Arguments:
        df (Pandas DataFrame): Formatted dataframe output from process_hemibrain function
        pre_type (string): "EPG" "PEN" or "PEG" presynaptic type
        post_type (string): "EPG" "PEN" or "PEG" postsynaptic type
        indx (string): Dataframe index to be used for plotting. If using rewrapped labels, use
            "index_fix_post", otherwise use "index_post"
        hemi_pre (string): "L" or "R"
        diffs (boolean): True to plot difference between pre_index and post_index, False to plot raw values
        wrap (boolean): True to wrap a difference of +/- 5 or greater to other side
        weighted (boolean): Whether to weight histogram by synaptic weight 
        ct (int): index to start at
        sub_ind(int): integer to consider the "wrapping maximum" ie: 
            if wrap==True and diff_ij = -5 -> wrapped_value = sub_ind + diff_ij
        mean(string): type of centrality to use "mean", "median", or "mode"
    Returns:
        max_l, max_r, mu_l, mu_r, sigma_l, sigma_r, num_l , num_r: average offset, mean and std of normal distribution, and number of connections
    """
    df[indx] = df[indx].astype(int)

    df["intidx"] = df[indx].astype(int)
    fdf = df[df.type_pre == pre_type]

    fdf=fdf[fdf["hemisphere_pre"]==hemi_pre]

    fdf = fdf[fdf["type_post"]==post_type]


    fdf = fdf[fdf["index_fix_pre"]==wedge_idx]
    fdf = fdf[fdf.hemisphere_pre == hemi_pre]
    # Creating difference values for the difference between pre_index and post_index
    # This subtracts the current wedge index from each post_index
    vals = np.array(fdf["intidx"]) - wedge_idx
    # Wrapping code using masking
    if wrap:
        vals[vals < -4] = sub_ind + vals[vals < -4]
        vals[vals > 4] = -sub_ind + vals[vals > 4]
        real_vals = vals
    # If no wrap, only include values within middle range that doesn't overflow
    else:
        fdf["mask"] = vals
        vals = vals[vals > -5]
        real_vals = vals[vals < 5]
        fdf = fdf[(fdf["mask"] > -5) & (fdf["mask"] < 5)]
        
        
    fdf["diffs"] = real_vals
    fdf = fdf.sort_values(by="intidx")

    hemi_r = fdf[fdf["hemisphere_post"]=="R"]
    hemi_l = fdf[fdf["hemisphere_post"]=="L"]

    if diffs:
        to_plot = "diffs"
    else:
        to_plot = "intidx"

    if centrality == "mean":
        max_l = np.mean(hemi_l[to_plot])
        max_r = np.mean(hemi_r[to_plot])
    elif centrality == "median":
        max_l = np.median(hemi_l[to_plot])
        max_r = np.median(hemi_r[to_plot])
    elif centrality == "mode":
        max_l = mode(hemi_l[to_plot])
        max_r = mode(hemi_r[to_plot])

    mu_l, sigma_l = norm.fit(hemi_l[to_plot])
    mu_r, sigma_r = norm.fit(hemi_r[to_plot])

    num_l = len(hemi_l[to_plot])
    num_r = len(hemi_r[to_plot])

    return max_l, max_r, mu_l, mu_r, sigma_l, sigma_r, num_l , num_r

def return_hists_full(df, pre_type, post_type, indx, hemi_pre="L", diffs=False, wrap = True, wedge_start = 1, wedge_end = 8, sub_ind = 8, centrality="mean", weighted=True):
    """
    Function to return histograms of {pre_type}_{hemisphere} to {post_type} (both hemispheres)
    Arguments:
        df (Pandas DataFrame): Formatted dataframe output from process_hemibrain function
        pre_type (string): "EPG" "PEN" or "PEG" presynaptic type
        post_type (string): "EPG" "PEN" or "PEG" postsynaptic type
        indx (string): Dataframe index to be used for plotting. If using rewrapped labels, use
            "index_fix_post", otherwise use "index_post"
        hemi_pre (string): "L" or "R"
        diffs (boolean): True to plot difference between pre_index and post_index, False to plot raw values
        wrap (boolean): True to wrap a difference of +/- 5 or greater to other side
        weighted (boolean): Whether to weight histogram by synaptic weight 
        ct (int): index to start at
        sub_ind(int): integer to consider the "wrapping maximum" ie: 
            if wrap==True and diff_ij = -5 -> wrapped_value = sub_ind + diff_ij
        mean(string): type of centrality to use "mean", "median", or "mode"
    Returns:
        max_l, max_r, mu_l, mu_r, sigma_l, sigma_r, num_l , num_r: average offset, mean and std of normal distribution, and number of connections
    """
    df[indx] = df[indx].astype(int)

    df["intidx"] = df[indx].astype(int)
    fdf = df[df.type_pre == pre_type]

    fdf=fdf[fdf["hemisphere_pre"]==hemi_pre]

    fdf_full = fdf[fdf["type_post"]==post_type]
    df_r = pd.DataFrame()
    df_l = pd.DataFrame()

    for i in range(wedge_start, wedge_end):
        fdf = fdf_full[fdf_full["index_fix_pre"]==i]
        fdf = fdf[fdf.hemisphere_pre == hemi_pre]
        # Creating difference values for the difference between pre_index and post_index
        # This subtracts the current wedge index from each post_index
        vals = np.array(fdf["intidx"]) - i
        # Wrapping code using masking
        if wrap:
            vals[vals < -4] = sub_ind + vals[vals < -4]
            vals[vals > 4] = -sub_ind + vals[vals > 4]
            real_vals = vals
        # If no wrap, only include values within middle range that doesn't overflow
        else:
            fdf["mask"] = vals
            vals = vals[vals > -5]
            real_vals = vals[vals < 5]
            fdf = fdf[(fdf["mask"] > -5) & (fdf["mask"] < 5)]
            
            
        fdf["diffs"] = real_vals
        fdf = fdf.sort_values(by="intidx")

        hemi_r = fdf[fdf["hemisphere_post"]=="R"]
        hemi_l = fdf[fdf["hemisphere_post"]=="L"]
        df_r = pd.concat((df_r,hemi_r))
        df_l = pd.concat((df_l,hemi_l))

    if diffs:
        to_plot = "diffs"
    else:
        to_plot = "intidx"

    if centrality == "mean":
        if weighted == True:
            max_l = np.sum(df_l[to_plot] * df_l["weight"]) / np.sum(df_l["weight"])
            max_r = np.sum(df_r[to_plot] * df_r["weight"]) / np.sum(df_r["weight"])
        else:
            max_l = np.mean(df_l[to_plot])
            max_r = np.mean(df_r[to_plot])
    elif centrality == "median":
        if weighted:
            df_l.sort_values(to_plot, inplace=True)
            cumsum = df_l.weight.cumsum()
            cutoff = df_l.weight.sum() / 2.0
            max_l = df_l[to_plot][cumsum >= cutoff].iloc[0]
            df_l.sort_values(to_plot, inplace=True)
            cumsum = df_r.weight.cumsum()
            cutoff = df_r.weight.sum() / 2.0
            max_r = df_r[to_plot][cumsum >= cutoff].iloc[0]
        else:
            max_l = np.median(df_l[to_plot])
            max_r = np.median(df_r[to_plot])
    elif centrality == "mode":
        max_l = mode(df_l[to_plot])
        max_r = mode(df_r[to_plot])
    if weighted:
        rweights = np.array(df_r["weight"])
        lweights = np.array(df_l["weight"])
        rconn = np.array(df_r[to_plot])
        lconn = np.array(df_l[to_plot])
        mu_l, sigma_l = norm.fit(np.repeat(lconn, lweights))
        mu_r, sigma_r = norm.fit(np.repeat(rconn, rweights))
    else:
        mu_l, sigma_l = norm.fit(df_l[to_plot])
        mu_r, sigma_r = norm.fit(df_r[to_plot])

    num_l = len(df_l[to_plot])
    num_r = len(df_r[to_plot])

    return max_l, max_r, mu_l, mu_r, sigma_l, sigma_r, num_l , num_r

def gauss_pdf(x, mu, sig):
    """
    Implements the normal distribution pdf
    Arguments:
        x (float): Value to calc pdf of 
        mu (float): normal distribution mean
        sig (float): normal distribution std
    Returns:
        Value of the PDF
    """   
    if mu == "L":
        return 0
    elif sig == 0:
        if round(x) == round(mu):
            return 1
        else:
            return 0
    else:
        return 1/(sig*np.sqrt(2*np.pi)) * np.exp(-.5*((x - mu)/sig)**2)

def radian_offset(conn_df, wrap = True, both=False, synthetic_pintr=0, threshold=0, symmetric=False, resolution_per_wedge=16):
    radians_per_RA = 2*np.pi / 8
    radians_per_wedge = 2*np.pi / (resolution_per_wedge // 2)

    total_indices = resolution_per_wedge * 3
    if both:
        total_indices = resolution_per_wedge * 4

    wedge_matrix = np.full((total_indices, total_indices), 0.0)
    if both:
        pre_types = ["EPG", "PENa", "PENb", "PEG"]
        post_types = ["EPG", "PENa", "PENb", "PEG"]        
    else:
        pre_types = ["EPG", "PEN", "PEG"]
        post_types = ["EPG", "PEN", "PEG"]
    LR = set(["L", "R"])
    # iterate through each pre_type, post_type, and hemisphere
    for n1, prtype in enumerate(pre_types):
        for n2, potype in enumerate(post_types):
            for hemi_pre in ["L", "R"]:
                if prtype[:3] == "PEN" and potype == "PEG":
                    continue
                if prtype[:3] == "PEN" and potype[:3]  == "PEN":
                    continue
                if prtype[:3] == "PEG" and potype == "PENa":
                    continue
                if prtype[:3] == "PEG" and potype == "PEG":
                    continue
                if prtype[:3] == "PEG" and potype == "EPG":
                    continue
                # Iterate through each wedge
                # Return hists yields the average centrality between wedge {i} and each post index
                l, r, mu_l, mu_r, std_l, std_r, num_l, num_r = return_hists_full(conn_df, prtype, potype, "index_fix_post", hemi_pre, True, wrap, weighted=True)
                l, r, mu_l, mu_r, std_l, std_r = tuple(np.array([l, r, mu_l, mu_r, std_l, std_r]) * radians_per_RA)
                # Conditions to omit connections that do not exist
                # adl and adr are the offset, and wl and wr are the weights
                l = int(round(l))
                wl = num_l / (16 // 2)

                r = int(round(r))
                wr = num_r / (16 // 2)

                if symmetric:
                    other_hemi = LR.difference(hemi_pre).pop()
                    l2, r2, mu_l2, mu_r2, std_l2, std_r2, num_l2, num_r2 = return_hists_full(conn_df, prtype, potype, "index_fix_post", other_hemi, True, wrap, weighted=True)
                    l2, r2, mu_l2, mu_r2, std_l2, std_r2 = tuple(np.array([l2, r2, mu_l2, mu_r2, std_l2, std_r2]) * radians_per_RA)

                    wl2 = num_l2 / (16 // 2)
                    wr2 = num_r2 / (16 // 2)
                    wr = (wr + wl2) / 2
                    wl = (wl + wr2) / 2 
                    mu_magL = (abs(mu_l) + abs(mu_r2)) / 2
                    mu_magR = (abs(mu_r) + abs(mu_l2)) / 2
                    mu_l = mu_magL * np.sign(mu_l)
                    mu_r = mu_magR * np.sign(mu_r)
                    std_l = (std_l + std_r2) / 2
                    std_r = (std_r + std_l2) / 2
                weil = wl  
                weir = wr
                for i in range(resolution_per_wedge//2):
                    for j in range(-resolution_per_wedge//4-1, resolution_per_wedge//2+resolution_per_wedge//4+1):

                        abs_diff = j - i

                        if wrap:
                            if abs_diff < -resolution_per_wedge//4:
                                abs_diff = resolution_per_wedge//2 - abs_diff
                            if abs_diff > resolution_per_wedge//4:
                                abs_diff = -resolution_per_wedge//2 + abs_diff
                        abs_diff = abs_diff * radians_per_wedge
                        for hemi_post in ["L", "R"]:
                        # These conditions ensure each index is placed correctly within the resulting weight matrix given its pre hemi, post hemi, and index
                            if hemi_post == "L" and weil < threshold:
                                continue
                            if hemi_post == "R" and weir < threshold:
                                continue
                            if hemi_pre == "L":
                                if hemi_post == "L":
                                    if j < 0:
                                        if wrap:
                                            wedge_matrix[resolution_per_wedge*n1 + i, resolution_per_wedge*n2 + resolution_per_wedge//2 + j] = gauss_pdf(abs_diff, mu_l, std_l)
                                    elif j >= resolution_per_wedge//2:
                                        if wrap:
                                            wedge_matrix[resolution_per_wedge*n1 + i, resolution_per_wedge*n2 - resolution_per_wedge//2 + j] = gauss_pdf(abs_diff, mu_l, std_l)
                                    else:
                                        wedge_matrix[resolution_per_wedge*n1 + i, resolution_per_wedge*n2 + j] = gauss_pdf(abs_diff, mu_l, std_l)
                                else:
                                    if j < 0:
                                        if wrap:

                                            wedge_matrix[resolution_per_wedge*n1 + i, resolution_per_wedge*n2 + resolution_per_wedge//2 + j + resolution_per_wedge//2] = gauss_pdf(abs_diff, mu_r, std_r)
                                    elif j >= resolution_per_wedge//2:
                                        if wrap:
                                            wedge_matrix[resolution_per_wedge*n1 + i, resolution_per_wedge*n2 - resolution_per_wedge//2 + j + resolution_per_wedge//2] = gauss_pdf(abs_diff, mu_r, std_r)
                                    else:
                                        wedge_matrix[resolution_per_wedge*n1 + i, resolution_per_wedge*n2 + j  + resolution_per_wedge//2] = gauss_pdf(abs_diff, mu_r, std_r)

                            else:
                                if hemi_post == "L":
                                    if j < 0:
                                        if wrap:
                                            wedge_matrix[resolution_per_wedge*n1 + i + resolution_per_wedge//2, resolution_per_wedge*n2 + resolution_per_wedge//2 + j] = gauss_pdf(abs_diff, mu_l, std_l)

                                    elif j >= resolution_per_wedge//2:
                                        if wrap:
                                            wedge_matrix[resolution_per_wedge*n1 + i + resolution_per_wedge//2, resolution_per_wedge*n2 - resolution_per_wedge//2 + j] = gauss_pdf(abs_diff, mu_l, std_l)
                                    else:
                                        wedge_matrix[resolution_per_wedge*n1 + i + resolution_per_wedge//2, resolution_per_wedge*n2 + j] = gauss_pdf(abs_diff, mu_l, std_l)

                                else:
                                    if j < 0:
                                        if wrap:
                                            wedge_matrix[resolution_per_wedge*n1 + i + resolution_per_wedge//2, resolution_per_wedge*n2 + resolution_per_wedge//2 + j + resolution_per_wedge//2] = gauss_pdf(abs_diff, mu_r, std_r)

                                    elif j >= resolution_per_wedge//2:
                                        if wrap:
                                            wedge_matrix[resolution_per_wedge*n1 + i + resolution_per_wedge//2, resolution_per_wedge*n2 - resolution_per_wedge//2 + j + resolution_per_wedge//2] = gauss_pdf(abs_diff, mu_r, std_r)
                                    else:
                                        wedge_matrix[resolution_per_wedge*n1 + i + resolution_per_wedge//2, resolution_per_wedge*n2 + j + resolution_per_wedge//2] = gauss_pdf(abs_diff, mu_r, std_r)

    return wedge_matrix

def gen_matrix_gauss(conn_df, n_per = 16, wrap = True, both=False, synthetic_pintr=0, threshold=0, symmetric=False, radians=False, null_offset=False):
    """
    Create connectivity Gaussian matrix from hemibrain wedge centrality

    Arguments:
        conn_df (Pandas Dataframe): Synapse level connectivity matrix
        n_of_each (int): number of total wedges per type to output
        wrap (Boolean): Whether or not to wrap +- 5 or greater wedge differences
        weighted (Boolean): True yields the count of synapses for each wedge connection, False will put in just the offset
        synthetic_pintr (float): Flag and weight to be the weight of the pintr connections

    Returns:
        Numpy matrix of connectivity
    """
    n_epgs_per_hemi = n_per
    n_pens_per_hemi = n_per
    n_pegs_per_hemi = n_per
    n_hemi_wedges = 16

    total_wedges = n_epgs_per_hemi + n_pens_per_hemi + n_pegs_per_hemi
    if both:
        total_wedges += n_pens_per_hemi

    wedge_matrix = np.full((total_wedges, total_wedges), 0.0)
    if both:
        pre_types = ["EPG", "PENa", "PENb", "PEG"]
        post_types = ["EPG", "PENa", "PENb", "PEG"]        
    else:
        pre_types = ["EPG", "PEN", "PEG"]
        post_types = ["EPG", "PEN", "PEG"]
    LR = set(["L", "R"])
    # iterate through each pre_type, post_type, and hemisphere
    for n1, prtype in enumerate(pre_types):
        for n2, potype in enumerate(post_types):
            for hemi_pre in ["L", "R"]:
                if prtype[:3] == "PEN" and potype == "PEG":
                    continue
                if prtype[:3] == "PEG" and potype == "PENa":
                    continue
                if prtype[:3] == "PEG" and potype == "PEG":
                    continue
                if prtype[:3] == "PEN" and potype[:3]  == "PEN":
                    continue
                # Iterate through each wedge
                # Return hists yields the average centrality between wedge {i} and each post index
                l, r, mu_l, mu_r, std_l, std_r, num_l, num_r = return_hists_full(conn_df, prtype, potype, "index_fix_post", hemi_pre, True, wrap, weighted=True)

                # Conditions to omit connections that do not exist
                # adl and adr are the offset, and wl and wr are the weights
                try:
                    l = int(round(l))
                except:
                    l = 0
                wl = num_l / (n_hemi_wedges // 2)
                try:
                    r = int(round(r))
                except:
                    r = 0
                wr = num_r / (n_hemi_wedges // 2)

    
                if symmetric:
                    other_hemi = LR.difference(hemi_pre).pop()
                    l2, r2, mu_l2, mu_r2, std_l2, std_r2, num_l2, num_r2 = return_hists_full(conn_df, prtype, potype, "index_fix_post", other_hemi, True, wrap, weighted=True)
                    wl2 = num_l2 / (n_hemi_wedges // 2)
                    wr2 = num_r2 / (n_hemi_wedges // 2)
                    wr = (wr + wl2) / 2
                    wl = (wl + wr2) / 2 
                    mu_magL = (abs(mu_l) + abs(mu_r2)) / 2
                    mu_magR = (abs(mu_r) + abs(mu_l2)) / 2
                    mu_l = mu_magL * np.sign(mu_l)
                    mu_r = mu_magR * np.sign(mu_r)
                    std_l = (std_l + std_r2) / 2
                    std_r = (std_r + std_l2) / 2
                weil = wl  
                weir = wr
                if null_offset:
                    if prtype[:3] == "PEN" and potype[:3] == "EPG":
                        pass
                    else:
                        mu_l = 0
                        mu_r = 0
                for i in range(n_epgs_per_hemi//2):
                    for j in range(-n_epgs_per_hemi//4-1, n_epgs_per_hemi//2+n_epgs_per_hemi//4+1):

                        abs_diff = j - i

                        if wrap:
                            if abs_diff < -n_epgs_per_hemi//4:
                                abs_diff = n_epgs_per_hemi//2 - abs_diff
                            if abs_diff > n_epgs_per_hemi//4:
                                abs_diff = -n_epgs_per_hemi//2 + abs_diff

                        for hemi_post in ["L", "R"]:
                        # These conditions ensure each index is placed correctly within the resulting weight matrix given its pre hemi, post hemi, and index
                            if hemi_post == "L" and weil < threshold:
                                continue
                            if hemi_post == "R" and weir < threshold:
                                continue
                            if hemi_pre == "L":
                                if hemi_post == "L":
                                    if j < 0:
                                        if wrap:
                                            wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + j] = gauss_pdf(abs_diff, mu_l, std_l) * wl
                                    elif j >= n_epgs_per_hemi//2:
                                        if wrap:
                                            wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + j] = gauss_pdf(abs_diff, mu_l, std_l) * wl
                                    else:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + j] = gauss_pdf(abs_diff, mu_l, std_l) * wl
                                else:
                                    if j < 0:
                                        if wrap:

                                            wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + j + n_epgs_per_hemi//2] = gauss_pdf(abs_diff, mu_r, std_r) * wr
                                    elif j >= n_epgs_per_hemi//2:
                                        if wrap:
                                            wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + j + n_epgs_per_hemi//2] = gauss_pdf(abs_diff, mu_r, std_r) * wr
                                    else:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + j  + n_epgs_per_hemi//2] = gauss_pdf(abs_diff, mu_r, std_r) * wr

                            else:
                                if hemi_post == "L":
                                    if j < 0:
                                        if wrap:
                                            wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + j] = gauss_pdf(abs_diff, mu_l, std_l) * wl

                                    elif j >= n_epgs_per_hemi//2:
                                        if wrap:
                                            wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + j] = gauss_pdf(abs_diff, mu_l, std_l) * wl
                                    else:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + j] = gauss_pdf(abs_diff, mu_l, std_l) * wl

                                else:
                                    if j < 0:
                                        if wrap:
                                            wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + j + n_epgs_per_hemi//2] = gauss_pdf(abs_diff, mu_r, std_r) * wr

                                    elif j >= n_epgs_per_hemi//2:
                                        if wrap:
                                            wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + j + n_epgs_per_hemi//2] = gauss_pdf(abs_diff, mu_r, std_r) * wr
                                    else:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + j + n_epgs_per_hemi//2] = gauss_pdf(abs_diff, mu_r, std_r) * wr
    # Inhibition purely from Kakaria
    if not synthetic_pintr == 0:
        new_wedge_matrix = np.full((total_wedges + n_per // 2, total_wedges + n_per // 2), np.nan)
        new_wedge_matrix[:-n_per // 2, :-n_per // 2] = wedge_matrix
        # EPG -> PINTR
        new_wedge_matrix[:n_per, -n_per // 2:] = -synthetic_pintr
        # PINTR -> PINTR
        new_wedge_matrix[-n_per//2:, -n_per // 2:] = synthetic_pintr
        for i in range(n_per // 2):
            # PINTR -> PEG
            new_wedge_matrix[i + -n_per//2, i + 2*n_epgs_per_hemi] = synthetic_pintr
            new_wedge_matrix[i + -n_per//2, i + 2*n_epgs_per_hemi + n_epgs_per_hemi//2] = synthetic_pintr
            # PINTR -> PEN
            new_wedge_matrix[i + -n_per//2, i + n_epgs_per_hemi] = synthetic_pintr
            new_wedge_matrix[i + -n_per//2, i + n_epgs_per_hemi + n_epgs_per_hemi//2] = synthetic_pintr
            # PINTR -> PENb
            if both:
                new_wedge_matrix[i + -n_per//2, i + 2*n_epgs_per_hemi] = synthetic_pintr
                new_wedge_matrix[i + -n_per//2, i + 2*n_epgs_per_hemi + n_epgs_per_hemi//2] = synthetic_pintr            
            # PINTR -> PINTR
            new_wedge_matrix[-n_per//2 + i, -n_per//2 + i] = np.nan
        wedge_matrix = new_wedge_matrix
    return wedge_matrix

def gen_matrix_kakaria(kakaria_mat, n_per = 16, wrap = True, synthetic_pintr=0):
    """
    Create connectivity matrix based on Kakaria.
    TODO : function can be folded into another
    Arguments:
        conn_df (Pandas Dataframe): Synapse level connectivity matrix
        n_of_each (int): number of total wedges per type to output
        wrap (Boolean): Whether or not to wrap +- 5 or greater wedge differences
        weighted (Boolean): True yields the count of synapses for each wedge connection, False will put in just the offset
        synthetic_pintr (float): Flag and weight to be the weight of the pintr connections
    Returns:
        Numpy matrix of connectivity
    """
    n_epgs_per_hemi = n_per
    n_pens_per_hemi = n_per
    n_pegs_per_hemi = n_per
    total_wedges = n_epgs_per_hemi + n_pens_per_hemi + n_pegs_per_hemi
    wedge_matrix = np.full((total_wedges, total_wedges), np.nan)
    pre_types = ["EPG", "PEN", "PEG"]
    post_types = ["EPG", "PEN", "PEG"]
    hemi_pre = "L"

    for n1, prtype in enumerate(pre_types):
        for n2, potype in enumerate(post_types):
            for n3, hemi_pre in enumerate(["L", "R"]):
            
                l = kakaria_mat[2*n1 + n3, 2*n2]
                r = kakaria_mat[2*n1 + n3, 2*n2+1]
                if np.isnan(l):
                    l = int(0)
                    wl=np.nan
                else:
                    l = int(round(l))
                    wl=20
                if np.isnan(r):
                    r = int(0)
                    wr=np.nan
                else:
                    r = int(round(r))
                    wr=20   
                # print(prtype, potype, l, r)
                for i in range(n_epgs_per_hemi//2):

                    for hemi_post in ["L", "R"]:
                        if hemi_pre == "L":
                            if hemi_post == "L":
                                if i + l < 0:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + i + l] = wl
                                elif i + l >= n_epgs_per_hemi//2:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + i + l] = wl
                                else:
                                    wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + i + l] = wl
                            else:
                                if i + r < 0:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + i + r + n_epgs_per_hemi//2] = wr
                                elif i + r >= n_epgs_per_hemi//2:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + i + r + n_epgs_per_hemi//2] = wr
                                else:
                                    wedge_matrix[n_epgs_per_hemi*n1 + i, n_epgs_per_hemi*n2 + i + r + n_epgs_per_hemi//2] = wr

                        else:
                            if hemi_post == "L":
                                if i + l < 0:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + i + l] = wl

                                elif i + l >= n_epgs_per_hemi//2:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + i + l] = wl
                                else:
                                    wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + i + l] = wl
                            else:
                                if i + r < 0:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + n_epgs_per_hemi//2 + i + r + n_epgs_per_hemi//2] = wr

                                elif i + r >= n_epgs_per_hemi//2:
                                    if wrap:
                                        wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 - n_epgs_per_hemi//2 + i + r + n_epgs_per_hemi//2] = wr
                                else:
                                    wedge_matrix[n_epgs_per_hemi*n1 + i + n_epgs_per_hemi//2, n_epgs_per_hemi*n2 + i + r + n_epgs_per_hemi//2] = wr
    # Inhibition purely from Kakaria
    if not synthetic_pintr == 0:
        new_wedge_matrix = np.full((total_wedges + n_per // 2, total_wedges + n_per // 2), np.nan)
        new_wedge_matrix[:-n_per // 2, :-n_per // 2] = wedge_matrix
        new_wedge_matrix[:n_per, -n_per // 2:] = synthetic_pintr
        new_wedge_matrix[-n_per//2:, -n_per // 2:] = synthetic_pintr
        for i in range(n_per // 2):
            # PINTR -> PEG
            new_wedge_matrix[-i-1, i + 2*n_epgs_per_hemi] = synthetic_pintr
            new_wedge_matrix[-i-1, i + 2*n_epgs_per_hemi + n_epgs_per_hemi//2] = synthetic_pintr
            # PINTR -> PEN
            new_wedge_matrix[-i-1, i + n_epgs_per_hemi] = synthetic_pintr
            new_wedge_matrix[-i-1, i + n_epgs_per_hemi + n_epgs_per_hemi//2] = synthetic_pintr          
            # PINTR -> PINTR
            new_wedge_matrix[-i-1, -n_per//2 + i] = np.nan
        wedge_matrix = new_wedge_matrix
    return wedge_matrix


def plot_nodes(y, both=False):
    import matplotlib.pyplot as plt

    # Helper function created by Brian to help created connectivity diagrams
    if both:
         colors = {'PENa-L':'tab:blue',
                   'PENa-R':'y',
                   'PENb-L':'c',
                   'PENb-R':'purple',
                   'EPG-R':'tab:orange',
                   'EPG-L':'tab:orange',
                   'PEG-R':'g',
                   'PEG-L':'g'}
    else:
        colors = {'PEN-L':'tab:blue',
                'PEN-R':'y',
                'EPG-R':'tab:orange',
                'EPG-L':'tab:orange',
                'PEG-R':'g',
                'PEG-L':'g'}
    for k in y.keys():
        plt.plot(0,y[k],'o',color=colors[k],markersize=15,label=k)
    plt.legend()
    
def plot_wedge(wedge,color,x_intermediate, y, weights = None, pretype=None, posttype=None):
    # Helper function created by Brian to help created connectivity diagrams
    import matplotlib.pyplot as plt

    if weights:
        max_w = 0
        for conn in weights:
            if weights[conn] > max_w:
                max_w = weights[conn]
    for conn in wedge.keys():
        if pretype:
            dash_loc = conn[0].find("-")
            if not pretype == conn[0][:dash_loc]:
                continue
        if posttype:
            dash_loc = conn[1].find("-")
            if not posttype == conn[1][:dash_loc]:
                continue
        #the line always start at x=0 and go to x=the mean offset as specified in the wedge
        x0 = 0
        xE = wedge[conn]
        #to make sure all of the lines don't bunch up at x=0, lets add a mid point thats off to the side
        if np.abs(xE) < 0.2:
            xMid = x_intermediate[conn[0]]
        else:
            xMid = (x0+xE)/2
        #the y values for the motif can be sterotyped by cell type
        try:
            y0 = y[conn[0]]
            yE = y[conn[1]]
        except KeyError:
            continue
        yMid = (y0+yE)/2
        
        #to make smooth lines, lets interpolate based on the start, mid, and end positions of each arrow
        x_pl = np.array([x0,xMid,xE])
        y_pl = np.array([y0,yMid,yE])
        tck,u = interpolate.splprep( [x_pl,y_pl], k = 2)
        xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
        
        #lets replace the last few line points with an arrowhead
        n_replace=5
        dx = xnew[-1]-xnew[-n_replace]
        dy = ynew[-1]-ynew[-n_replace]
        # plot the line
        if weights:
            plt.plot(xnew[:-n_replace],ynew[:-n_replace],linewidth=2*weights[conn] / max_w,color=color)
        else:
            plt.plot(xnew[:-n_replace],ynew[:-n_replace],linewidth=.5,color=color)
        # plot the arrowhead (using plt.arrow has funky arrowheads)
        #         plt.arrow(xnew[-n_replace],ynew[-n_replace],dx,dy,width=.03,head_starts_at_zero=False,
        #                  length_includes_head=True,color='k')
        plt.gca().annotate("", xy=(xnew[-1], ynew[-1]), xytext=(xnew[-n_replace], ynew[-n_replace]),
             arrowprops=dict(width=0.1,headwidth=5, color=color))
        
        # divide the L and R
        plt.axhline(0,color='k',linestyle=':')
        plt.xlabel('Offset Connectivity Between Wedges')
    plt.xlim([-3,3])

def gen_connectivity(conn_df, rounded = False, both=False, symmetric=True,wrap=True,threshold=0):
    """
    Create connectivity matrix from hemibrain wedge centrality for connectivity wedge diagram
    Arguments:
        conn_df (Pandas Dataframe): Synapse level connectivity matrix
        n_of_each (int): number of total wedges per type to output
        wrap (Boolean): Whether or not to wrap +- 5 or greater wedge differences
        weighted (Boolean): True yields the count of synapses for each wedge connection, False will put in just the offset
    Returns:
        Numpy matrix of connectivity
    """
    LR = set(["L", "R"])

    if both:
        pre_types = ["EPG", "PENa", "PENb", "PEG"]
        post_types = ["EPG", "PENa", "PENb", "PEG"]
    else:
        pre_types = ["EPG", "PEN", "PEG"]
        post_types = ["EPG", "PEN", "PEG"]
    wedge_offsets = {}
    wedge_weights = {}
    # iterate through each pre_type, post_type, and hemisphere
    for n1, prtype in enumerate(pre_types):
        for n2, potype in enumerate(post_types):
            for hemi_pre in ["L", "R"]:
                # Iterate through each wedge
                if prtype[:3] == "PEN" and potype == "PEG":
                    continue
                if prtype[:3] == "PEN" and potype[:3]  == "PEN":
                    continue
                if prtype[:3] == "PEN" and potype == "PEG":
                    continue
                if prtype[:3] == "PEG" and potype == "PENa":
                    continue
                if prtype[:3] == "PEG" and potype == "PEG":
                    continue
                if prtype[:3] == "PEN" and potype[:3]  == "PEN":
                    continue
                if prtype[:3] == "PEG" and potype == "EPG":
                    continue
                # print(prtype, potype, prtype[:3])

                l, r, _, _, _, _, num_l, num_r = return_hists_full(conn_df, prtype, potype, "index_fix_post", hemi_pre, True, wrap, weighted=True)
                if np.isnan(l) or np.isnan(r):
                    continue
                wl = num_l / (16 // 2)
                wr = num_r / (16 // 2)
                if symmetric:
                    other_hemi = LR.difference(hemi_pre).pop()
                    l2, r2, _, _, _, _, num_l2, num_r2 = return_hists_full(conn_df, prtype, potype, "index_fix_post", other_hemi, True, wrap, weighted=True)
                    if np.isnan(l2) or np.isnan(r2):
                        continue
                    l = (abs(l) + abs(r2)) / 2 * np.sign(l)
                    r = (abs(r) + abs(l2)) / 2 * np.sign(r)
                    wl2 = num_l2 / (16 // 2)
                    wr2 = num_r2 / (16 // 2)
                    wr = (wr + wl2) / 2
                    wl = (wl + wr2) / 2 
                if rounded:
                    if wr > threshold:
                        wedge_offsets[(f"{prtype}-{hemi_pre}", f"{potype}-R")] = int(round(r))
                        wedge_weights[(f"{prtype}-{hemi_pre}", f"{potype}-R")] = wr
                    if wl > threshold:
                        wedge_offsets[(f"{prtype}-{hemi_pre}", f"{potype}-L")] = int(round(l))
                        wedge_weights[(f"{prtype}-{hemi_pre}", f"{potype}-L")] = wl
                else:
                    if wr > threshold:
                        wedge_offsets[(f"{prtype}-{hemi_pre}", f"{potype}-R")] = r
                        wedge_weights[(f"{prtype}-{hemi_pre}", f"{potype}-R")] = wr
                    if wl > threshold:
                        wedge_offsets[(f"{prtype}-{hemi_pre}", f"{potype}-L")] = l
                        wedge_weights[(f"{prtype}-{hemi_pre}", f"{potype}-L")] = wl
                
    return wedge_offsets, wedge_weights

def plot_hemis(conn_df, pre_hemi = "L",post_hemi = "L", pret="PEN",postt="EPG", ax=None):
    """
    Plot hemi to hemi connectivity for pre and post  type
    Arguments:
        conn_df (Pandas Dataframe): Synapse level connectivity matrix
        pre_hemi, post_hemi (string): Pre and Post hemispheres to plot
        pret, postt (string): Pre and Post Type to plot
        ax (plt axis): If given, plot there
    Returns:
        None: Draws matplotlib plot
    """
    import matplotlib.pyplot as plt

    df = []
    df_fix = []
    mat_sm = np.zeros((8,8))
    mat_flip_sm = np.zeros((8,8))
    mat_fix_sm = np.zeros((8,8))
    mat_fix_flip_sm = np.zeros((8,8))
    conn_df_pen_epg = conn_df[(conn_df.type_pre == pret) & (conn_df.type_post == postt)]

    for i in range(conn_df_pen_epg.shape[0]):
        row = conn_df_pen_epg.iloc[i]
        ipre = row["index_pre"]
        ipost = row["index_post"]
        ifixpre = row["index_fix_pre"]
        ifixpost = row["index_fix_post"]
        hemipre = row["hemisphere_pre"]
        hemipost = row["hemisphere_post"]
        weight = row["weight"]

        if hemipre == pre_hemi and hemipost == post_hemi:

            df.append({'pen_pre': ipre, 'epg_post': ipost, 'weight': weight})
            df_fix.append({'pen_pre': ifixpre, 'epg_post': ifixpost, 'weight': weight})

            mat_sm[ipre - 1, ipost - 1] += weight
            mat_fix_sm[ifixpre - 1, ifixpost - 1] += weight
            mat_flip_sm[ipre - 1, ipost - 1] += weight
            mat_fix_flip_sm[ifixpre - 1, ifixpost - 1] += weight

    xlabs = []
    for i in range(8):
        xlabs.append(f"{postt} {i + 1}")
    ylabs = []
    for i in range(8):
        ylabs.append(f"{pret} {i + 1}")
    if not ax:
        plt.xticks(np.arange(8), xlabs, rotation=45)
        plt.yticks(np.arange(8), ylabs)#, rotation=45)

        for i in range(8):

            plt.vlines(i,0,8, color='k')
        for i in range(8):

            plt.hlines(i,0,8, color='k')

        plt.pcolor(mat_fix_sm)
    else:
        ax.set_xticks(np.arange(8), xlabs)

        ax.set_yticks(np.arange(8), ylabs)#, rotation=45)

        for i in range(8):
            ax.vlines(i,0,8, color='k')
        for i in range(8):

            ax.hlines(i,0,8, color='k')
        ax.set_title(f"{pret} {pre_hemi} to {postt} {post_hemi}")
        ax.pcolor(mat_fix_sm)   

def plot_conn(conn_df, type_pre, type_post, both=False):
    """
    Create connectivity matrix from hemibrain wedge centrality for connectivity wedge diagram
    Arguments:
        conn_df (Pandas Dataframe): Synapse level connectivity matrix
        type_pre, type_post (string): Pre and Post Type to plot
    Returns:
        None: Draws matplotlib plot
    """
    import matplotlib.pyplot as plt

    conn_df_pen_epg = conn_df[(conn_df.type_pre == type_pre) & (conn_df.type_post == type_post)]

    df = []
    df_fix = []
    if both:
        n_conn = 16
    else:
        n_conn = 16
    mat = np.zeros((n_conn,n_conn))
    mat_flip = np.zeros((n_conn,n_conn))
    mat_fix = np.zeros((n_conn,n_conn))
    mat_fix_flip = np.zeros((n_conn,n_conn))

    for i in range(conn_df_pen_epg.shape[0]):
        row = conn_df_pen_epg.iloc[i]
        ipre = row["index_pre"]
        ipost = row["index_post"]
        ifixpre = row["index_fix_pre"]
        ifixpost = row["index_fix_post"]
        hemipre = row["hemisphere_pre"]
        hemipost = row["hemisphere_post"]
        weight = row["weight"]

        if hemipre == "R":
            ipre += 8
            ifixpre += 8
        if hemipost == "R":
            ipost += 8
            ifixpost += 8
        df.append({f'{type_pre}_pre': ipre, f'{type_post}_post': ipost, 'weight': weight})
        df_fix.append({f'{type_pre}_pre': ifixpre, f'{type_post}_post': ifixpost, 'weight': weight})

        mat[ipre - 1, ipost - 1] += weight
        mat_fix[ifixpre - 1, ifixpost - 1] += weight
        mat_flip[ipre - 1, ipost - 1] += weight
        mat_fix_flip[ifixpre - 1, ifixpost - 1] += weight

    xlabs = []
    for i in range(8):
        xlabs.append(f"{type_post} L {i + 1}")
    for i in range(8):
        xlabs.append(f"{type_post} R {i + 1}")
    plt.xticks(np.arange(16), xlabs, rotation=45)

    ylabs = []
    for i in range(8):
        ylabs.append(f"{type_pre} L {i + 1}")
    for i in range(8):
        ylabs.append(f"{type_pre} R {i + 1}")

    plt.yticks(np.arange(16), ylabs)#, rotation=45)

    for i in range(n_conn):

        plt.vlines(i,0,16, color='k')
    for i in range(16):

        plt.hlines(i,0,16, color='k')

    plt.xticks(np.arange(16), xlabs, rotation=45)

    plt.yticks(np.arange(16), ylabs)#, rotation=45)

    plt.pcolor(mat_fix)
    return mat_fix

def just_offset(conn_df, n_per = 16, weighted=False, both=False):
    n_epgs_per_hemi = n_per
    n_pens_per_hemi = n_per
    n_pegs_per_hemi = n_per
    total_wedges = n_epgs_per_hemi + n_pens_per_hemi + n_pegs_per_hemi

    if both:
        n = 4
        pre_types = ["EPG", "PENa", "PENb", "PEG"]
        post_types = ["EPG", "PENa", "PENb", "PEG"]
    else:
        n = 3
        pre_types = ["EPG", "PEN", "PEG"]
        post_types = ["EPG", "PEN", "PEG"]

    mean_matrix = np.full((n*2, n*2), 0.0)
    mean_matrix_u = np.full((n*2, n*2), 0.0)
    std_matrix = np.full((n*2, n*2), 0.0)
    weight_matrix = np.full((n*2, n*2), 0.0)

    ct = 0
    for n1, prtype in enumerate(pre_types):
        for n2, potype in enumerate(post_types):
            for n3, hemi_pre in enumerate(["L", "R"]):

                l_diffs = []
                r_diffs = []
                l_mus = []
                r_mus = []
                l_stds = []
                r_stds = []
                l_nums = []
                r_nums = []
                for i in range(8):

                    l, r, mu_l, mu_r, std_l, std_r, num_l, num_r = return_hists(conn_df, prtype, potype, "index_fix_post", hemi_pre, True, True, wedge_idx = i + 1)
                    if not np.isnan(l):
                        l_diffs.append(l)
                        l_mus.append(mu_l)
                        l_stds.append(std_l)
                        l_nums.append(num_l)
                    if not np.isnan(r):
                        r_diffs.append(r)
                        r_mus.append(mu_r)
                        r_stds.append(std_r)
                        r_nums.append(num_r)

                if len(l_diffs) > 0:
                    l = np.mean(l_diffs)
                    mu_l = np.mean(l_mus)
                    std_l = np.mean(l_stds)
                    num_l = np.mean(l_nums)
                    adl = l
                else:
                    l = 0
                    mu_l = "L"
                    std_l = 0
                    num_l = 0
                    adl = 0
                if len(r_diffs) > 0:
                    r = np.mean(r_diffs)
                    mu_r = np.mean(r_mus)
                    std_r = np.mean(r_stds)
                    num_r = np.mean(r_nums)
                    adr = r
                else:
                    r = 0
                    mu_l = "R"
                    std_l = 0
                    num_r = 0
                    adr = 0
                if not weighted:
                    num_r = 1
                    num_l = 1
                # print(prtype, potype,hemi_pre, n1 * 2 + n3, n2*2, n1 * 2 + n3, n2*2 + 1, l, r)
                
                mean_matrix[n1 * 2 + n3, n2*2] = int(round(l))
                mean_matrix[n1 * 2 + n3, n2*2 + 1] = int(round(r))
                
                mean_matrix_u[n1 * 2 + n3, n2*2] = l
                mean_matrix_u[n1 * 2 + n3, n2*2 + 1] = r
                
                std_matrix[n1 * 2 + n3, n2*2] = std_l
                std_matrix[n1 * 2 + n3, n2*2 + 1] = std_r
                
                weight_matrix[n1 * 2 + n3, n2*2] = num_l
                weight_matrix[n1 * 2 + n3, n2*2 + 1] = num_r
                ct += 1

    return mean_matrix, mean_matrix_u, std_matrix, weight_matrix

def return_graph(token, both_pen = "a", si1=9, si2=9, si3=9, pen_rewrap=True, peg_rewrap=True, only_EBPB=True):
    neuron_df_a, conn_df_a = process_hemibrain(token, both_pen = both_pen, si1=si1, si2=si2, si3=si3, pen_rewrap=pen_rewrap, peg_rewrap=peg_rewrap, only_EBPB=only_EBPB)
    del conn_df_a["order_pre"]
    del conn_df_a["order_post"]
    conn_df_a = conn_df_a.rename(columns={'index_pre': 'wedge_num_pre', 'index_fix_pre': 'wedge_num_pre_corrected', 'index_post': 'wedge_num_post', 'index_fix_post': 'wedge_num_post_corrected'})
    neuron_df_a = neuron_df_a.rename(columns={'index_fix': 'wedge_num_corrected', 'index': 'wedge_num'})
    G = nx.from_pandas_edgelist(conn_df_a, 'bodyId_pre', 'bodyId_post', ['weight','roi'])

    nx.set_node_attributes(G, pd.Series(neuron_df_a.wedge_num, index=neuron_df_a.index).to_dict(), 'wedge_num')
    nx.set_node_attributes(G, pd.Series(neuron_df_a.wedge_num_corrected, index=neuron_df_a.index).to_dict(), 'wedge_num_corrected')
    nx.set_node_attributes(G, pd.Series(neuron_df_a.hemisphere, index=neuron_df_a.index).to_dict(), 'hemisphere')
    nx.set_node_attributes(G, pd.Series(neuron_df_a.instance, index=neuron_df_a.index).to_dict(), 'instance')
    nx.set_node_attributes(G, pd.Series(neuron_df_a.type, index=neuron_df_a.index).to_dict(), 'type')
    nx.set_node_attributes(G, pd.Series(neuron_df_a.bodyId, index=neuron_df_a.index).to_dict(), 'bodyId')

    return G

def return_hists_full_extra(df, pre_type, post_type, indx, fix=True,hemi_pre="L", diffs=False, wrap = True, wedge_start = 1, wedge_end = 10, sub_ind = 8, centrality="mean", weighted=True):
    """
    Function to return histograms of {pre_type}_{hemisphere} to {post_type} (both hemispheres)
    Arguments:
        df (Pandas DataFrame): Formatted dataframe output from process_hemibrain function
        pre_type (string): "EPG" "PEN" or "PEG" presynaptic type
        post_type (string): "EPG" "PEN" or "PEG" postsynaptic type
        indx (string): Dataframe index to be used for plotting. If using rewrapped labels, use
            "index_fix_post", otherwise use "index_post"
        hemi_pre (string): "L" or "R"
        diffs (boolean): True to plot difference between pre_index and post_index, False to plot raw values
        wrap (boolean): True to wrap a difference of +/- 5 or greater to other side
        weighted (boolean): Whether to weight histogram by synaptic weight 
        ct (int): index to start at
        sub_ind(int): integer to consider the "wrapping maximum" ie: 
            if wrap==True and diff_ij = -5 -> wrapped_value = sub_ind + diff_ij
        mean(string): type of centrality to use "mean", "median", or "mode"
    Returns:
        max_l, max_r, mu_l, mu_r, sigma_l, sigma_r, num_l , num_r: average offset, mean and std of normal distribution, and number of connections
    """
    df[indx] = df[indx].astype(int)

    df["intidx"] = df[indx].astype(int)
    fdf = df[df.type_pre == pre_type]

    fdf=fdf[fdf["hemisphere_pre"]==hemi_pre]

    fdf_full = fdf[fdf["type_post"]==post_type]
    df_r = pd.DataFrame()
    df_l = pd.DataFrame()

    for i in range(wedge_start, wedge_end):
        if fix:
            fdf = fdf_full[fdf_full["index_fix_pre"]==i]
        else:
            fdf = fdf_full[fdf_full["index_pre"]==i]
        fdf = fdf[fdf.hemisphere_pre == hemi_pre]
        # Creating difference values for the difference between pre_index and post_index
        # This subtracts the current wedge index from each post_index
        vals = np.array(fdf["intidx"]) - i
        raw_vals = np.array(fdf["intidx"]) - i
        # Wrapping code using masking
        if wrap:
            vals[vals < -4] = sub_ind + vals[vals < -4]
            vals[vals > 4] = -sub_ind + vals[vals > 4]
            real_vals = vals
        # If no wrap, only include values within middle range that doesn't overflow
        else:
            fdf["mask"] = vals
            vals = vals[vals > -5]
            real_vals = vals[vals < 5]
            fdf = fdf[(fdf["mask"] > -5) & (fdf["mask"] < 5)]            
            
        fdf["diffs"] = real_vals
        fdf["raw"] = raw_vals
        fdf = fdf.sort_values(by="intidx")
        fdf.loc[:,"neuron_avg"] = np.full(fdf.shape[0], -100)

        hemi_r = fdf.loc[fdf["hemisphere_post"]=="R"]
        hemi_l = fdf.loc[fdf["hemisphere_post"]=="L"]
        
        meanl = np.sum(hemi_l["diffs"] * hemi_l["weight"]) / np.sum(hemi_l["weight"])
        meanr = np.sum(hemi_r["diffs"] * hemi_r["weight"]) / np.sum(hemi_r["weight"])
    
        hemi_r.loc[:,"mean_diff"] = np.full(hemi_r.shape[0], meanr)
        hemi_l.loc[:,"mean_diff"] = np.full(hemi_l.shape[0], meanl)
        
        hemi_r.loc[:,"raw_diffs"] = hemi_r["raw"]
        hemi_l.loc[:,"raw_diffs"] = hemi_l["raw"]
        
        hemi_r.loc[:,"idx"] = np.full(hemi_r.shape[0], i)
        hemi_l.loc[:,"idx"] = np.full(hemi_l.shape[0], i)
        
        df_r = pd.concat((df_r,hemi_r))
        df_l = pd.concat((df_l,hemi_l))
        
    for instance_post in np.unique(df_r["instance_post"]):
        df_r_subsel = df_r.loc[df_r["instance_post"] == instance_post]
        df_r.loc[df_r["instance_post"] == instance_post, "neuron_avg"] = np.sum(df_r_subsel["diffs"] * df_r_subsel["weight"]) / np.sum(df_r_subsel["weight"])
    for instance_post in np.unique(df_l["instance_post"]):
        df_l_subsel = df_l.loc[df_l["instance_post"] == instance_post]
        df_l.loc[df_l["instance_post"] == instance_post, "neuron_avg"] = np.sum(df_l_subsel["diffs"] * df_l_subsel["weight"]) / np.sum(df_l_subsel["weight"])

    if diffs:
        to_plot = "diffs"
    else:
        to_plot = "intidx"

    if centrality == "mean":
        if weighted == True:
            max_l = np.sum(df_l[to_plot] * df_l["weight"]) / np.sum(df_l["weight"])
            max_r = np.sum(df_r[to_plot] * df_r["weight"]) / np.sum(df_r["weight"])
        else:
            max_l = np.mean(df_l[to_plot])
            max_r = np.mean(df_r[to_plot])
    elif centrality == "median":
        if weighted:
            df_l.sort_values(to_plot, inplace=True)
            cumsum = df_l.weight.cumsum()
            cutoff = df_l.weight.sum() / 2.0
            max_l = df_l[to_plot][cumsum >= cutoff].iloc[0]
            df_l.sort_values(to_plot, inplace=True)
            cumsum = df_r.weight.cumsum()
            cutoff = df_r.weight.sum() / 2.0
            max_r = df_r[to_plot][cumsum >= cutoff].iloc[0]
        else:
            max_l = np.median(df_l[to_plot])
            max_r = np.median(df_r[to_plot])
    elif centrality == "mode":
        max_l = mode(df_l[to_plot])
        max_r = mode(df_r[to_plot])
    if weighted:
        rweights = np.array(df_r["weight"])
        lweights = np.array(df_l["weight"])
        rconn = np.array(df_r[to_plot])
        lconn = np.array(df_l[to_plot])
        mu_l, sigma_l = norm.fit(np.repeat(lconn, lweights))
        mu_r, sigma_r = norm.fit(np.repeat(rconn, rweights))
    else:
        mu_l, sigma_l = norm.fit(df_l[to_plot])
        mu_r, sigma_r = norm.fit(df_r[to_plot])

    num_l = len(df_l[to_plot])
    num_r = len(df_r[to_plot])
    sym = np.sum(df_l["diffs"]) + np.sum(df_r["diffs"])
    
    return max_l, max_r, mu_l, mu_r, sigma_l, sigma_r, num_l , num_r, df_l, df_r, sym

def calc_diffs(conn_df, strategy, fix=True,trunc=False):
    diffs = np.array([])
    raw_diffs = np.array([])
    diffs_mean = np.array([])
    preposts=np.array([])
    types=np.array([])
    # post_types=np.array([])
    hemis=np.array([])
    hemi_type=np.array([])
    strategies=np.array([])
    weights = np.array([])
    idxs = np.array([])
    syms = np.array([])
    prtype = np.array([])
    potype = np.array([])
    roi = np.array([])
    neuron_avg = np.array([])
    sym_dict={"L":0, "R":0}
    size_track=0
    # conn_df_thresh = conn_df_both[conn_df_both["weight"] >=10]
    for pre_type in ["EPG", "PENa", "PENb", "PEG"]:
        for post_type in ["EPG", "PENa", "PENb", "PEG"]:

            for hemi_pre in ["L", "R"]:
                if pre_type[:3] == "PEN" and post_type == "PEG":
                    continue
                if pre_type[:3] == "PEN" and post_type[:3]  == "PEN":
                    continue
                if pre_type[:3] == "PEG" and post_type == "PENa":
                    continue
                if pre_type[:3] == "PEG" and post_type == "PEG":
                    continue
                if trunc:
                    l, r, mu_l, mu_r, std_l, std_r, num_l, num_r, df_l, df_r, sym = return_hists_full_extra(conn_df, pre_type, post_type, "index_fix_post", fix, hemi_pre, True, True, weighted=True,wedge_start=3,wedge_end=8)
                else:
                    l, r, mu_l, mu_r, std_l, std_r, num_l, num_r, df_l, df_r, sym = return_hists_full_extra(conn_df, pre_type, post_type, "index_fix_post", fix, hemi_pre, True, True, weighted=True)
                if pre_type[:3] == "PEG" and post_type == "EPG":
                    if hemi_pre == "L":
                        df_r=pd.DataFrame({"diffs":[],"raw_diffs":[], "mean_diff":[],"idx":[],"weight":[], "neuron_avg":[], "roi":[]})
                    if hemi_pre == "R":
                        df_l=pd.DataFrame({"diffs":[],"raw_diffs":[],"mean_diff":[],"idx":[],"weight":[], "neuron_avg":[], "roi":[]})

                dl = (df_l["diffs"] - l)
                dr = (df_r["diffs"] - r)
                dl_mean = (df_l["mean_diff"] - l)
                dr_mean = (df_r["mean_diff"] - r)
                
                diffs=np.append(diffs,np.array(dl))
                raw_diffs=np.append(raw_diffs,np.array(df_l["raw_diffs"]))
                diffs_mean=np.append(diffs_mean,np.array(dl_mean))
                preposts=np.append(preposts, np.array(df_l["diffs"]))
                types=np.append(types, np.full(dl.shape[0], f"{pre_type}-{post_type}"))
                hemis=np.append(hemis, np.full(dl.shape[0], f"{hemi_pre}-L"))
                hemi_type=np.append(hemi_type, np.full(dl.shape[0], f"{pre_type}{hemi_pre}-{post_type}L"))
                strategies=np.append(strategies, np.full(dl.shape[0], strategy))
                weights=np.append(weights, np.array(df_l["weight"]))
                idxs=np.append(idxs, np.array(df_l["idx"]))
                neuron_avg=np.append(neuron_avg, np.array(df_l["neuron_avg"]))
                prtype=np.append(prtype, np.full(dl.shape[0], pre_type))
                potype=np.append(potype, np.full(dl.shape[0], post_type))
                roi=np.append(roi, np.array(df_l["roi"]))


                diffs=np.append(diffs, np.array(dr))
                raw_diffs=np.append(raw_diffs,np.array(df_r["raw_diffs"]))
                diffs_mean=np.append(diffs_mean,np.array(dr_mean))
                preposts=np.append(preposts, np.array(df_r["diffs"]))
                types=np.append(types, np.full(dr.shape[0], f"{pre_type}-{post_type}"))
                hemis=np.append(hemis, np.full(dr.shape[0], f"{hemi_pre}-R"))
                hemi_type=np.append(hemi_type, np.full(dr.shape[0], f"{pre_type}{hemi_pre}-{post_type}R"))
                strategies=np.append(strategies, np.full(dr.shape[0], strategy))
                weights=np.append(weights, np.array(df_r["weight"]))
                idxs=np.append(idxs, np.array(df_r["idx"]))
                neuron_avg=np.append(neuron_avg, np.array(df_r["neuron_avg"]))
                prtype=np.append(prtype, np.full(dr.shape[0], pre_type))
                potype=np.append(potype, np.full(dr.shape[0], post_type))
                roi=np.append(roi, np.array(df_r["roi"]))

                size_track = size_track + dr.shape[0] + dl.shape[0]
                sym_dict[hemi_pre] = sym
            sym_sum = sym_dict["L"] + sym_dict["R"]
            syms = np.append(syms, np.full(size_track, sym_sum))
            size_track=0
    dfl = pd.DataFrame({"diff":diffs,
                        "raw_diffs":raw_diffs,
                        "diff_mean":diffs_mean,
                        "post-pre":preposts,
                        "type":types, 
                        "hemis":hemis,
                        "weights":weights,
                        "hemi_type":hemi_type,
                        "strategy":strategies,
                        "symmetry":syms,
                        "pre_type":prtype,
                        "post_type":potype,
                        "neuron_avg":neuron_avg,
                        "roi":roi,
                        "indices":idxs})
    weighted_postpre = np.repeat(dfl["post-pre"], dfl["weights"])
    weighted_type = np.repeat(dfl["type"], dfl["weights"])
    weighted_hemis = np.repeat(dfl["hemis"], dfl["weights"])
    weighted_hemitype = np.repeat(dfl["hemi_type"], dfl["weights"])
    weighted_diff = np.repeat(dfl["diff"], dfl["weights"])
    weighted_meandiff = np.repeat(dfl["diff_mean"], dfl["weights"])
    weighted_rawdiff = np.repeat(dfl["raw_diffs"], dfl["weights"])
    weighted_weight = np.repeat(dfl["weights"], dfl["weights"])
    weighted_strat = np.repeat(dfl["strategy"], dfl["weights"])
    weighted_indices = np.repeat(dfl["indices"], dfl["weights"])
    weighted_symmetry = np.repeat(dfl["symmetry"], dfl["weights"])
    weighted_prtype = np.repeat(dfl["pre_type"], dfl["weights"])
    weighted_potype = np.repeat(dfl["post_type"], dfl["weights"])
    weighted_navg = np.repeat(dfl["neuron_avg"], dfl["weights"])
    weighted_roi = np.repeat(dfl["roi"], dfl["weights"])
    df_weighted = pd.DataFrame({"diff":weighted_diff,
                                "diff_mean":weighted_meandiff,
                                "raw_diff":weighted_rawdiff,
                                "post-pre":weighted_postpre,
                                "type":weighted_type, 
                                "hemis":weighted_hemis,
                                "weights":weighted_weight,
                                "hemi_type":weighted_hemitype,
                                "indices":weighted_indices,
                                "symmetry":weighted_symmetry,
                                "neuron_avg":weighted_navg,
                                "pre_type":weighted_prtype,
                                "post_type":weighted_potype,
                                "roi":weighted_roi,
                                "strategy":weighted_strat})
    return dfl, df_weighted