import src.torch.pytorch_util as ptu

#----------Controller imports--------#
from model_init_study.controller.nn_controller \
    import NeuralNetworkController

#----------Environment imports--------#
import gym
import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch
import mb_ge ## Contains ball in cup
import redundant_arm ## contains redundant arm

#----------Utils imports--------#
import sys, os
import argparse
import re
import pandas as pd
import numpy as np
from tqdm import tqdm # for pretty print of progress bar

#----------Clustering imports--------#
from sklearn.cluster import KMeans

#----------Plot imports--------#
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

def str_to_tab(tab_as_str, dim=2):
    splitted_tab_as_str = tab_as_str.split(' ')
    tab = []
    ## preprocessing to remove non digits elements
    for substr in splitted_tab_as_str:
        clean_substr = ''
        for a in substr:
            if a.isdigit() or a == '.':
                clean_substr += a

        if clean_substr != '':
            tab.append(float(clean_substr))

    assert len(tab) == dim, "Obtained tab length does not correspond to given expected dimensions"

    return tab
def process_env(args):
    ### Environment initialization ###
    env_register_id = 'BallInCup3d-v0'
    gym_args = {}
    if args.environment == 'ball_in_cup':
        env_register_id = 'BallInCup3d-v0'
        ss_min = -0.33
        ss_max = 0.33
        dim_map = 3
        bd_inds = [0, 1, 2]
    elif args.environment == 'redundant_arm':
        env_register_id = 'RedundantArmPos-v0'
        ss_min = -1
        ss_max = 1
        dim_map = 2
        bd_inds = [-2, -1]
    elif args.environment == 'redundant_arm_no_walls':
        env_register_id = 'RedundantArmPosNoWalls-v0'
        ss_min = -1
        ss_max = 1
        dim_map = 2
        bd_inds = [-2, -1]
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        env_register_id = 'RedundantArmPosNoWallsNoCollision-v0'
        ss_min = -1
        ss_max = 1
        dim_map = 2
        bd_inds = [-2, -1]
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
        ss_min = -1
        ss_max = 1
        dim_map = 2
        gym_args['dof'] = 100
        bd_inds = [-2, -1]
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        ss_min = 0
        ss_max = 600
        dim_map = 2
        bd_inds = [0, 1]
    elif args.environment == 'empty_maze':
        env_register_id = 'FastsimEmptyMapNavigationPos-v0'
        # ss_min = -10
        # ss_max = 10
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        ss_min = np.array([0, 0, -1, -1, -1, -1])
        ss_max = np.array([600, 600, 1, 1, 1, 1])
        init_obs = np.array([300., 300., 0., 0., 0. , 0.])
        dim_map = 2
        bd_inds = [0, 1]
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        ss_min = 0
        ss_max = 600
        dim_map = 2
        gym_args['physical_traps'] = True
        bd_inds = [0, 1]
    elif args.environment == 'half_cheetah':
        env_register_id = 'HalfCheetah-v3'
        a_min = np.array([-1, -1, -1, -1, -1, -1])
        a_max = np.array([1, 1, 1, 1, 1, 1])
        ss_min = np.array([-10]*18)
        ss_max = np.array([10]*18)
        init_obs = np.array([0.]*18)
        dim_map = 1
        gym_args['exclude_current_positions_from_observation'] = False
        gym_args['reset_noise_scale'] = 0
        bd_inds = [0]
    elif args.environment == 'walker2d':
        env_register_id = 'Walker2d-v3'
        a_min = np.array([-1, -1, -1, -1, -1, -1])
        a_max = np.array([1, 1, 1, 1, 1, 1])
        ss_min = np.array([-10]*18)
        ss_max = np.array([10]*18)
        init_obs = np.array([0.]*18)
        dim_map = 1
        gym_args['exclude_current_positions_from_observation'] = False
        gym_args['reset_noise_scale'] = 0
        bd_inds = [0]
    elif args.environment == 'hexapod_omni':
        is_local_env = True
        max_step = 300 # ctrl_freq = 100Hz, sim_time = 3.0 seconds 
        obs_dim = 48
        act_dim = 18
        dim_x = 36
        ss_min = 0
        ss_max = 1
        dim_map = 2
    else:
        raise ValueError(f"{args.environment} is not a defined environment")
    
    gym_env = gym.make(env_register_id, **gym_args)

    try:
        max_step = gym_env._max_episode_steps
    except:
        try:
            max_step = gym_env.max_steps
        except:
            raise AttributeError("Env does not allow access to _max_episode_steps or to max_steps")

    return gym_env, max_step, ss_min, ss_max, dim_map, bd_inds

def select_inds(data, path, sel_size, search_method, horizon, sel_method, args):
    ok_flag = True
    ret_data = None
    if sel_method == 'random':
        ## Select sel_size individuals randomly from archive
        ret_data = data.sample(n=sel_size)
    elif sel_method == 'max':
        ## select sel_size individuals that cover the most from real archive
        filename = os.path.join(path,
                                f'archive_{args.asize}_real_all.dat')
        ret_data = pd.read_csv(filename)
        ## drop the last column which was made because there is a comma
        ## after last value i a line
        ret_data = ret_data.iloc[:,:-1]
        ## get env info
        _, _, ss_min, ss_max, dim_map, bd_inds = process_env(args)
        ## add bins field to data
        ret_data = get_data_bins(ret_data, args, ss_min,
                                 ss_max, dim_map, bd_inds)
        n_reached_bins = len(ret_data['bins'].unique())
        grouped_data = ret_data.groupby('bins') ## group data by bins
        ## a little nasty but need to increase sel size until we actually get
        ## sel size elements in ret data
        ret_size = 0
        curr_size = sel_size
        while (ret_size < sel_size):
            while_data = grouped_data.head(curr_size/n_reached_bins)
            ret_size = len(while_data)
            curr_size += 1
        ret_data = while_data.iloc[:100] ## cut in case we get too many
    elif sel_method == 'kmeans':
        ## select sel_size individuals closest to sel_size kmeans clusters
        if not 'ens' in search_method:
            # get env info
            _, _, ss_min, ss_max, dim_map, bd_inds = process_env(args)
            bd_cols = [f'bd{i}' for i in range(dim_map)]
            ret_data = get_closest_to_clusters_centroid(data, bd_cols, sel_size)
        else: ## can't do kmeans selection on model ensemble
            ok_flag = False
    elif sel_method == 'nov':
        ## select sel_size individuals that are the most novel on the model
        ## get env info
        _, _, ss_min, ss_max, dim_map, bd_inds = process_env(args)
        bd_cols = [f'bd{i}' for i in range(dim_map)]
        ## single model selection: max novelty
        if not 'ens' in search_method:
            ret_data = get_most_nov_data(data, sel_size, bd_cols)
        ## model ensemble selection: max of min novelty across ensemble
        else:
            ret_data = get_most_nov_data(data, sel_size, bd_cols, ens_size=4)
    return ret_data, ok_flag

def get_novelty_scores(data, bd_cols, k=15):
    from sklearn.neighbors import NearestNeighbors
    
    # Convert the dataset to a numpy array
    dataset = np.array(data[bd_cols])
    novelty_scores = np.empty((len(dataset)))

    # Compute the k-NN of the data point
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(dataset)

    for data_point, cpt in tqdm(zip(dataset, range(len(dataset))),total=len(dataset)):
        k_nearest_neighbors = neighbors.kneighbors([data_point], return_distance=False)[0]
        
        # Compute the average distance between the data point and its k-NN
        average_distance = np.mean(np.linalg.norm(dataset[k_nearest_neighbors] - data_point,
                                                  axis=1))
        novelty_scores[cpt] = average_distance
        # Print the average distance as a measure of novelty

    return novelty_scores

def get_most_nov_data(data, n, bd_cols, ens_size=1):
    if ens_size > 1:
        ens_data_nov = np.empty((ens_size, len(data)))
        for i in range(ens_size):
            bd_cols_loc = [f"{bd_cols[k]}_m{i}" for k in range(len(bd_cols))]
            data_nov = get_novelty_scores(data, bd_cols_loc)
            ens_data_nov[i,:] = data_nov
        data_nov = ens_data_nov.min(axis=0)
    else:
        data_nov = get_novelty_scores(data, bd_cols)
    sorted_data = data.assign(nov=data_nov)
    sorted_data = sorted_data.sort_values(by=['nov'], ascending=False)
    return sorted_data.head(n=n)

def get_closest_to_clusters_centroid(data, bd_cols, n_clusters):
    # Use k-means clustering to divide the data space into n_clusters clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data[bd_cols])
    
    # Select closest data point from each cluster
    data_centers = pd.DataFrame(columns=data.columns)

    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        cluster = data[kmeans.labels_ == i]
        # cluster = data[data['cluster'] == i]
        # Calculate the distance of each point in the cluster to the cluster center
        dist = 0
        for j in range(len(bd_cols)):
            dist += (cluster[bd_cols[j]] - cluster_center[j])**2
        cluster['distance'] = dist**0.5
        # Select the point with the minimum distance to the cluster center
        closest_point = cluster.loc[cluster['distance'].idxmin()]
        data_centers = data_centers.append(closest_point)

    return data_centers

def get_data_bins(data, args, ss_min, ss_max, dim_map, bd_inds):
    df_min = data.iloc[0].copy(); df_max = data.iloc[0].copy()

    for i in range(dim_map):
        df_min[f'bd{i}'] = ss_min[bd_inds[i]]
        df_max[f'bd{i}'] = ss_max[bd_inds[i]]

    ## Deprecated but oh well
    data = data.append(df_min, ignore_index = True)
    data = data.append(df_max, ignore_index = True)

    args.nb_div

    for i in range(dim_map):
        data[f'{i}_bin'] = pd.cut(x = data[f'bd{i}'],
                                  bins = args.nb_div, 
                                  labels = [p for p in range(args.nb_div)])

    ## assign data to bins
    data = data.assign(bins=pd.Categorical
                       (data.filter(regex='_bin')
                        .apply(tuple, 1)))

    ## remove df min and df max
    data.drop(data.tail(2).index,inplace=True)

    return data

def compute_cov(data, args):
    ## get env info
    _, _, ss_min, ss_max, dim_map, bd_inds = process_env(args)
    ## add bins field to data
    data = get_data_bins(data, args, ss_min, ss_max, dim_map, bd_inds)
    ## count number of bins filled
    counts = data['bins'].value_counts()
    total_bins = args.nb_div**dim_map
    ## return coverage (number of bins filled)
    return len(counts[counts>=1])/total_bins

def main(args):
    final_asize = args.final_asize
    
    ## Set params and rename some 
    gym_env, max_step, ss_min, ss_max, dim_map, bd_inds = process_env(args)
    dim_x = 0 # set when we open a data file for first time
    
    search_methods = args.search_methods
    sel_methods = args.sel_methods
    m_horizons = args.m_horizons
    n_reps = args.n_reps

    ab_methods = []
    for search_method in search_methods:
        if search_method == 'random-policies':
            ab_methods.append(search_method)
        else:
            for m_horizon in m_horizons:
                ab_methods.append(
                    f'{search_method}-h{m_horizon}'
                )

    ## Plot table with coverage for each search method and selection method    
    ## ab methods == search methods with model horizons
    column_headers = ab_methods
    row_headers = sel_methods
    cell_text_cov = [["" for _ in range(len(column_headers))]
                     for _ in range(len(row_headers))]
    
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    ## Get rep folders abs paths
    cwd = os.getcwd()
    
    archive_covs = np.zeros((len(ab_methods), len(sel_methods), n_reps))
    archive_covs[:] = np.nan
    # archive_std_covs = np.zeros((len(ab_methods), len(sel_methods), len(n_reps)))
    # archive_std_covs[:] = np.nan

    ## Main loop for printing figures
    searchm_cpt = 0
    abm_cpt = 0
    for search_method in search_methods:
        if search_method == 'random-policies':
            ## get abs path to working dir (dir containing reps)
            working_dir = os.path.join(cwd,
                                       f'{search_method.replace("-", "_")}')
            ## Open each archive file: warning budget is final_asize on these
            rep_folders = next(os.walk(working_dir))[1]
            abs_rep_folders = [os.path.join(working_dir, rep_folder)
                               for rep_folder in rep_folders]
            rep_cpt = 0
            for abs_rep_folder in abs_rep_folders:
                filename = os.path.join(abs_rep_folder,
                                        f'archive_{final_asize}_real_all.dat')
                rep_data = pd.read_csv(filename)
                # drop the last column which was made because there is a comma
                # after last value i a line
                rep_data = rep_data.iloc[:,:-1]
                if dim_x == 0:
                    dim_x = len([col for col in rep_data.columns if 'x' in col])

                # compute cov for given rep_data
                archive_covs[abm_cpt, 0, rep_cpt] = compute_cov(
                    rep_data, args
                )
                
                rep_cpt += 1
            abm_cpt += 1
        else:
            mh_cpt = 0
            for m_horizon in m_horizons:
                selm_cpt = 0
                for sel_method in sel_methods:
                    working_dir = os.path.join(cwd,
                                               f'{search_method}_h{m_horizon}')
                    ## Open each archive file
                    rep_folders = next(os.walk(working_dir))[1]
                    abs_rep_folders = [os.path.join(working_dir, rep_folder)
                                       for rep_folder in rep_folders]

                    rep_cpt = 0
                    for abs_rep_folder in abs_rep_folders:
                        filename = os.path.join(abs_rep_folder,
                                                f'archive_{args.asize}.dat')

                        rep_data = pd.read_csv(filename)
                        # drop the last column which was made because there is a
                        # comma after last value i a line
                        rep_data = rep_data.iloc[:,:-1] 

                        if dim_x == 0:
                            dim_x = len([col for col in rep_data.columns if 'x' in col])
                            
                        ## Select final_asize inds based on sel_method
                        sel_data, ok = select_inds(rep_data, abs_rep_folder,
                                                   final_asize, search_method,
                                                   m_horizon, sel_method, args)

                        
                        if ok:
                            ## Load real evaluations of individuals
                            filename = os.path.join(abs_rep_folder,
                                                    f'archive_{args.asize}_real_all.dat')
                            data_real_all = pd.read_csv(filename)
                            # drop the last column which was made because there is a
                            # comma after last value i a line
                            data_real_all = data_real_all.iloc[:,:-1] 
                            gen_cols = [f'x{i}' for i in range(dim_x)]
                            merged_data = sel_data.merge(data_real_all, on=gen_cols,
                                                         suffixes=('_model',''))
                            # compute cov for given rep_data
                            archive_covs[abm_cpt,
                                         selm_cpt,
                                         rep_cpt] = compute_cov(merged_data, args)

                        rep_cpt += 1
                    selm_cpt += 1

                abm_cpt += 1
                mh_cpt += 1
                
        searchm_cpt += 1

    #================================== PLOT ===================================#

    all_ab_methods_labels = []
    all_ab_methods_covs = []

    ab_cpt = 0
    for ab_method in ab_methods:
        sel_cpt = 0
        for sel_method in sel_methods:
            cov_vals = archive_covs[ab_cpt, sel_cpt]
            if not np.isnan(cov_vals).all():
                all_ab_methods_covs.append(cov_vals)
                all_ab_methods_labels.append(f'{ab_method}-{sel_method}')
            sel_cpt += 1
        ab_cpt += 1
        
    fig, ax = plt.subplots()
    ax.boxplot(all_ab_methods_covs)
    ax.set_xticklabels(all_ab_methods_labels)
    ax.set_ylabel("Coverage (max is 1)")
    
    plt.title(f'Coverage for each archive bootstrapping method\n' \
              f'(Behavior space division in {args.nb_div} parts per dimension)')
    fig.set_size_inches(27, 9)
    
    plt.savefig(f"{args.environment}_bp_coverage_{args.nb_div}", dpi=300, bbox_inches='tight')

    if args.show:
        plt.show()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--search-methods', nargs="*",
                        type=str, default=['random-policies', 'det', 'det_ens'])
    parser.add_argument('--m-horizons', nargs="*",
                        type=int, default=[10, 100])
    parser.add_argument('--sel-methods', nargs="*",
                        type=str, default=['random', 'max', 'nov', 'kmeans'])
    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')

    parser.add_argument("--show", help="Show the plot and saves it." \
                        " Unless specified, just save the mean plot over all" \
                        " repetitions", action="store_true")

    parser.add_argument("--nb_div", type=int, default=10, help="Number of " \
                        "equally separated bins to divide the outcome space")
    parser.add_argument("--n-reps", type=int, default=10, help="Number of " \
                        "repetitions of each experiment")
    parser.add_argument("--asize", type=int, default=10100, help="Saved " \
                        "final archive size for each experiment")
    parser.add_argument("--final-asize", type=int, default=10100,
                        help="Final archive size for bootstrapping")
    args = parser.parse_args()

    main(args)
