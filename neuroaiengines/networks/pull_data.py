import neuprint
from neuroaiengines.networks import offset_utils
from argparse import ArgumentParser
import os
   
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('token', help='Your neuprint token from https://neuprint.janelia.org')
    parser.add_argument('--pen-a',help='Also download connectivity for just PENa and save to hemibrain_conn_df_a.csv',action='store_true')
    parser.add_argument('--pen-b',help='Also download connectivity for just PENb and save to hemibrain_conn_df_b.csv',action='store_true')
    args = parser.parse_args()
    token = args.token
    

    client = neuprint.Client('https://neuprint.janelia.org', token=token, dataset='hemibrain:v1.2.1')
    def download_and_save(pen='both'):
        
        neuron_df_both, conn_df_both = offset_utils.process_hemibrain(token, both_pen = pen, si1=10, si2=10,
                                                                si3=10, which_hemi="L", epg_special=10,pen_rewrap=False,
                                                                peg_rewrap=False, only_EBPB=True, connection_threshold=4)
    
        
        def save(df,name):
            path = os.path.join(os.path.dirname(__file__), name)
            df.to_csv(path)
        save(conn_df_both, 'hemibrain_conn_df_{pen}.csv')
    download_and_save()
    if args.pen_a:
        download_and_save('a')
    if args.pen_b:
        download_and_save('b')
    