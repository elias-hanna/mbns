import src.torch.pytorch_util as ptu


#----------Init methods imports--------#
from model_init_study.initializers.random_policy_initializer \
    import RandomPolicyInitializer
from model_init_study.initializers.random_actions_initializer \
    import RandomActionsInitializer
from model_init_study.initializers.random_actions_random_policies_hybrid_initializer \
    import RARPHybridInitializer
from model_init_study.initializers.brownian_motion \
    import BrownianMotion
from model_init_study.initializers.colored_noise_motion \
    import ColoredNoiseMotion

#----------Controller imports--------#
from model_init_study.controller.nn_controller \
    import NeuralNetworkController

#----------Separator imports--------#
from model_init_study.visualization.fetch_pick_and_place_separator \
    import FetchPickAndPlaceSeparator
from model_init_study.visualization.ant_separator \
    import AntSeparator
from model_init_study.visualization.ball_in_cup_separator \
    import BallInCupSeparator
from model_init_study.visualization.redundant_arm_separator \
    import RedundantArmSeparator
from model_init_study.visualization.fastsim_separator \
    import FastsimSeparator

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

def plot_archive(data, plt, args, ss_min, ss_max):
    # FOR BINS / GRID
    if args.plot_type == "grid":
        ## Artificially add a min and max of state space element to auto set boundaries of grid
        df_min = data.iloc[0].copy(); df_max = data.iloc[0].copy()
        df_min[1] = ss_min; df_max[1] = ss_max
        df_min[2] = ss_min; df_max[2] = ss_max
        
        # Deprecated
        data = data.append(df_min, ignore_index = True)
        data = data.append(df_max, ignore_index = True)
        # data = pd.concat([data, df_min]) ## does ugly thingies cba to look at them rn
        # data = pd.concat([data, df_max])
        nb_div = 100
        
        data['x_bin']=pd.cut(x = data.iloc[:,1],
                             bins = nb_div, 
                             labels = [p for p in range(nb_div)])
        data['y_bin']=pd.cut(x = data.iloc[:,2],
                             bins = nb_div,
                             labels = [p for p in range(nb_div)])
        
        fig, ax = plt.subplots()
        data.plot.scatter(x="x_bin",y="y_bin",c=0,colormap="viridis", s=2, ax=ax)
        plt.xlim(0,nb_div)
        plt.ylim(0,nb_div)
    elif args.plot_type == "3d" or args.environment == "ball_in_cup":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = data.iloc[:,1]
        ys = data.iloc[:,2]
        zs = data.iloc[:,3]

        ax.scatter(xs, ys, zs, marker="x")
        plt.xlim(ss_min,ss_max)
        plt.ylim(ss_min,ss_max)
        
    else:
        #fig, ax = plt.subplots(nrows=1, ncols=2)
        fig, ax = plt.subplots()

        # FOR JUST A SCATTER PLOT OF THE DESCRIPTORS - doesnt work for interactive selection
        #data.plot.scatter(x=2,y=3,c=0,colormap='Spectral', s=2, ax=ax, vmin=-0.1, vmax=1.2)
        data.plot.scatter(x=1,y=2,c=0,colormap='viridis', s=2, ax=ax)
        plt.xlim(ss_min,ss_max)
        plt.ylim(ss_min,ss_max)

        #data.plot.scatter(x=1,y=2,s=2, ax=ax[0])
        #data.plot.scatter(x=3,y=4,c=0,colormap='viridis', s=2, ax=ax)
        #data.plot.scatter(x=4,y=5,s=2, ax=ax[1])
        #plt.xlim(-0.5,0.5)
        #plt.ylim(-0.5,0.5)

    return fig, ax

def process_env(args):
    ### Environment initialization ###
    env_register_id = 'BallInCup3d-v0'
    gym_args = {}
    if args.environment == 'ball_in_cup':
        env_register_id = 'BallInCup3d-v0'
        separator = BallInCupSeparator
        ss_min = -0.33
        ss_max = 0.33
        dim_map = 3
    elif args.environment == 'redundant_arm':
        env_register_id = 'RedundantArmPos-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls':
        env_register_id = 'RedundantArmPosNoWalls-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        env_register_id = 'RedundantArmPosNoWallsNoCollision-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
        dim_map = 2
        gym_args['dof'] = 100
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        separator = FastsimSeparator
        ss_min = 0
        ss_max = 600
        dim_map = 2
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        separator = FastsimSeparator
        ss_min = 0
        ss_max = 600
        dim_map = 2
        gym_args['physical_traps'] = True
    elif args.environment == 'hexapod_omni':
        is_local_env = True
        max_step = 300 # ctrl_freq = 100Hz, sim_time = 3.0 seconds 
        obs_dim = 48
        act_dim = 18
        dim_x = 36
        separator = None
        ss_min = -1
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

    return gym_env, max_step, ss_min, ss_max, dim_map

def main(args):
    
    gym_env, max_step, ss_min, ss_max, dim_map = process_env(args)
    
    data = pd.read_csv(args.filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line

    #=====================PLOT DATA===========================#

    archive_size = len(data.index)
    print('\nArchive size: ', archive_size, '\n')

    fig, ax = plot_archive(data, plt, args, ss_min, ss_max)

    plt.show() 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--init-methods', nargs="*",
                        type=str, default=['brownian-motion', 'colored-noise-beta-0', 'colored-noise-beta-1', 'colored-noise-beta-2', 'random-actions', 'random-policies'])
    parser.add_argument('--init-episodes', nargs="*",
                        type=int, default=[20])

    ## DAQD specific parameters
    parser.add_argument('--fitness-funcs', nargs="*",
                        type=str, default=['energy_minimization', 'disagr_minimization'])
    parser.add_argument('--transfer-selection', nargs="*",
                        type=str, default=['disagr', 'disagr_bd'])
    parser.add_argument('--nb-transfer', nargs="*",
                        type=int, default=[1, 10])
    parser.add_argument('--dump-vals', nargs="*",
                        type=str, default=['20', '40', '60', '80', '100'])
    
    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')
    parser.add_argument('--dump-path', type=str, default='default_dump/')
    parser.add_argument("--filename", type=str) # file to visualize rollouts from
    # parser.add_argument("--sim_time", type=float, help="simulation time depending on the type of archive you chose to visualize, 3s archive or a 5s archive")
    parser.add_argument("--show", help="Show the plot and saves it. Unless specified, just save the mean plot over all repetitions",
                    action="store_true")
    parser.add_argument("--plot_type", type=str, default="scatter", help="scatter plot, grid plot or 3d")
    parser.add_argument("--nb_div", type=int, default=10, help="Number of equally separated bins to divide the outcome space in")
    parser.add_argument("--random-budget", type=int, default=1000, help="Number of random evaluations for plotting, will look into full_random_{random-budget} folder")

    args = parser.parse_args()

    if args.filename is not None:
        main(args)

    else:
        ## Set params and rename some 
        n_init_method = len(args.init_methods) # 2
        init_methods = args.init_methods #['random-policies', 'random-actions']
        n_init_episodes = len(args.init_episodes) # 4
        init_episodes = args.init_episodes #[5, 10, 15, 20]
        n_fitness_funcs = len(args.fitness_funcs) # 2
        fitness_funcs = args.fitness_funcs #['energy_minimization', 'disagr_minimization']
        n_transfer_sels = len(args.transfer_selection) # 2
        transfer_sels = args.transfer_selection #['disagr', 'disagr_bd']
        n_nb_transfers = len(args.nb_transfer) # 2
        nb_transfers = args.nb_transfer #[10, 1]

        gym_env, max_step, ss_min, ss_max, dim_map = process_env(args)

        ## Plot table with mean prediction error for n step predictions    
        column_headers = [init_method for init_method in init_methods]
        # row_headers = [init_episode for init_episode in init_episodes]
        # row_headers = [20, 40, 60, 80, 100]
        row_headers = args.dump_vals
        cell_text_size = []
        cell_text_cov = []

        cell_text_model_size = []
        cell_text_model_cov = []
        for i in range(n_nb_transfers * n_transfer_sels):
            cell_text_size.append([["" for _ in range(len(column_headers))]
                                   for _ in range(len(row_headers))])
            cell_text_cov.append([["" for _ in range(len(column_headers))]
                                  for _ in range(len(row_headers))])
            cell_text_model_size.append([["" for _ in range(len(column_headers))]
                                         for _ in range(len(row_headers))])
            cell_text_model_cov.append([["" for _ in range(len(column_headers))]
                                        for _ in range(len(row_headers))])
        rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

        ## Get rep folders abs paths
        rep_folders = next(os.walk(f'.'))[1]
        rep_folders = [x for x in rep_folders if (x.isdigit())]

        archive_mean_sizes = np.zeros((len(init_methods), len(args.dump_vals)))
        archive_std_sizes = np.zeros((len(init_methods), len(args.dump_vals)))
        archive_mean_covs = np.zeros((len(init_methods), len(args.dump_vals)))
        archive_std_covs = np.zeros((len(init_methods), len(args.dump_vals)))

        model_archive_mean_sizes = np.zeros((len(init_methods), len(args.dump_vals)))
        model_archive_std_sizes = np.zeros((len(init_methods), len(args.dump_vals)))
        model_archive_mean_covs = np.zeros((len(init_methods), len(args.dump_vals)))
        model_archive_std_covs = np.zeros((len(init_methods), len(args.dump_vals)))

        model_to_real_diff_sizes = np.zeros((len(init_methods), len(args.dump_vals)))
        model_to_real_diff_covs = np.zeros((len(init_methods), len(args.dump_vals)))

        ## Get pred error data for each initialization method
        has_pred_errors = True
        try:
            # pred_error_data = np.load('pred_error_data.npz')
            pred_error_data = np.load(
                f"/home/elias-hanna/results/sigma_0.05/pred_errors/" \
                f"{args.environment}_0.05_results/" \
                f"pred_error_data.npz"
            )
        except FileNotFoundError:
            has_pred_errors = False
            print("\n\nWARNING: pred_error_data.npz file NOT FOUND! Visualisations" \
                  " that require it WON'T be printed \n\n")
        if has_pred_errors:
            ## mean/std pred errors shape: (method, init budget, prediction horizon(1, 20, full))
            mean_pred_errors = pred_error_data['mean_pred_errors']
            std_pred_errors = pred_error_data['std_pred_errors']

        has_random_data = True
        try:
            random_folder_path = f"/home/elias-hanna/results/full_random_{args.random_budget}/" \
                                 f"{args.environment}/"
            rep_folders = next(os.walk(random_folder_path))[1]
            rep_folders = [x for x in rep_folders if (x.isdigit())]
            random_cov = 0
            random_asize = 0 
            for rep_path in rep_folders:
                archive_folder = f'{random_folder_path}/{rep_path}/vanilla_20_energy_' \
                                 f'minimization_all_{nb_transfers[0]}_results/'
                archive_files = next(os.walk(archive_folder))[2]
                archive_files = [f for f in archive_files if 'archive' in f]
                archive_files = [f for f in archive_files if 'model' not in f]
                assert len(archive_files) == 1
                archive_file = archive_files[0]
                archive_file_path = os.path.join(archive_folder, archive_file)
                rep_data = pd.read_csv(archive_file_path)

                rep_data = rep_data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
                #=====================ARCHIVE SIZE===========================#
                random_asize += len(rep_data.index)

                #=====================COVERAGE===========================#

                df_min = rep_data.iloc[0].copy(); df_max = rep_data.iloc[0].copy()
                df_min[1] = ss_min; df_max[1] = ss_max
                df_min[2] = ss_min; df_max[2] = ss_max

                if args.environment == "ball_in_cup":
                    df_min[3] = ss_min; df_max[3] = ss_max

                ## Deprecated but oh well
                rep_data = rep_data.append(df_min, ignore_index = True)
                rep_data = rep_data.append(df_max, ignore_index = True)

                nb_div = args.nb_div

                rep_data['x_bin']=pd.cut(x = rep_data.iloc[:,1],
                                                                 bins = nb_div, 
                                         labels = [p for p in range(nb_div)])

                rep_data['y_bin']=pd.cut(x = rep_data.iloc[:,2],
                                         bins = nb_div,
                                         labels = [p for p in range(nb_div)])

                total_bins = nb_div**2

                if args.environment == "ball_in_cup":
                    rep_data['z_bin']=pd.cut(x = rep_data.iloc[:,3],
                                         bins = nb_div,
                                         labels = [p for p in range(nb_div)])

                    total_bins = nb_div**3

                rep_data = rep_data.assign(bins=pd.Categorical
                                           (rep_data.filter(regex='_bin')
                                            .apply(tuple, 1)))

                counts = rep_data['bins'].value_counts()

                random_cov += len(counts[counts>=1])/total_bins

            random_cov /= len(rep_folders)
            random_asize /= len(rep_folders)

        except FileNotFoundError:
            has_random_data = False
            print("\n\nWARNING: Some full_random data files were NOT FOUND! Visualisations" \
                  " that require it WON'T be printed \n\n")
            
        has_ns_cov = True
        ### Be careful need to regen the pred error data in the right order
        try:
            ns_bd_data = np.load(
                f"/home/elias-hanna/results/ns_cov_results/{args.environment}/" \
                f"archive_all_gen500.npz"
            )
            bd_keys = [key for key in ns_bd_data.keys() if 'ind' in key]
            ## we just load a random archive to have the good format
            archive_path = f"/home/elias-hanna/results/ns_cov_results/{args.environment}/" \
                           f"archive.dat"
            rep_data = pd.read_csv(archive_path)
            rep_data = rep_data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
            bd_data = pd.DataFrame().reindex(columns=rep_data.columns)

            df_min = rep_data.iloc[0].copy(); df_max = rep_data.iloc[0].copy()
            df_min[1] = ss_min; df_max[1] = ss_max
            df_min[2] = ss_min; df_max[2] = ss_max

            if args.environment == "ball_in_cup":
                df_min[3] = ss_min; df_max[3] = ss_max

            ## Deprecated but oh well
            bd_data = bd_data.append(df_min, ignore_index = True)
            bd_data = bd_data.append(df_max, ignore_index = True)


            for key in bd_keys:
                df_bd = rep_data.iloc[0].copy()
                df_bd[1] = ns_bd_data[key][0]
                df_bd[2] = ns_bd_data[key][1]

                if args.environment == "ball_in_cup":
                    df_bd[3] = ns_bd_data[key][2]

                ## Deprecated but oh well
                bd_data = bd_data.append(df_bd, ignore_index = True)

            nb_div = args.nb_div

            bd_data['x_bin']=pd.cut(x = bd_data.iloc[:,1],
                                     bins = nb_div, 
                                     labels = [p for p in range(nb_div)])

            bd_data['y_bin']=pd.cut(x = bd_data.iloc[:,2],
                                     bins = nb_div,
                                     labels = [p for p in range(nb_div)])

            total_bins = nb_div**2

            if args.environment == "ball_in_cup":
                bd_data['z_bin']=pd.cut(x = bd_data.iloc[:,3],
                                     bins = nb_div,
                                     labels = [p for p in range(nb_div)])

                total_bins = nb_div**3

            bd_data = bd_data.assign(bins=pd.Categorical
                                       (bd_data.filter(regex='_bin')
                                        .apply(tuple, 1)))

            counts = bd_data['bins'].value_counts()

            ns_cov = len(counts[counts>=1])/total_bins
            ns_asize = len(bd_keys)
            
        except FileNotFoundError:
            has_ns_cov = False
            print("\n\nWARNING: archive_all_gen500.npz file NOT FOUND! Visualisations" \
                  " that require it WON'T be printed \n\n")
            
        ## Main loop for printing figures
        
        for j in range(n_init_episodes):
            init_episode = init_episodes[j]
            for i in tqdm(range(n_init_method)):
                init_method =  init_methods[i]
                for k in range(n_fitness_funcs):
                    fitness_func = fitness_funcs[k]
                    tab_cpt = 0
                    for l in range(n_nb_transfers):
                        nb_transfer = nb_transfers[l]
                        for m in range(n_transfer_sels):
                            transfer_sel = transfer_sels[m]
                            rep_cpt = 0
                            mean_archive_size = 0

                            archive_sizes = np.empty((len(rep_folders), len(row_headers)))
                            archive_sizes[:] = np.nan
                            model_archive_sizes = np.empty((len(rep_folders), len(row_headers)))
                            model_archive_sizes[:] = np.nan

                            coverages = np.empty((len(rep_folders), len(row_headers)))
                            coverages[:] = np.nan
                            model_coverages = np.empty((len(rep_folders), len(row_headers)))
                            model_coverages[:] = np.nan

                            for rep_path in rep_folders:
                                archive_folder = f'{rep_path}/{init_method}_{init_episode}_{fitness_func}_{transfer_sel}_{nb_transfer}_results/'
                                try:
                                    archive_files = next(os.walk(archive_folder))[2]
                                except Exception as error:
                                    import pdb; pdb.set_trace()
                                    excelsior = e
                                archive_files = [f for f in archive_files if 'archive' in f]
                                
                                model_archive_files = [f for f in archive_files if 'model' in f]
                                model_archive_numbers = [int(re.findall(r'\d+', f)[0]) for f in model_archive_files]
                                sorted_model_archive_files = [f for _, f in sorted(zip(model_archive_numbers, model_archive_files), key=lambda pair: pair[0])]
                                
                                archive_files = [f for f in archive_files if 'model' not in f]
                                
                                archive_numbers = [int(re.findall(r'\d+', f)[0]) for f in archive_files]
                                sorted_archive_files = [f for _, f in sorted(zip(archive_numbers, archive_files), key=lambda pair: pair[0])]

                                for archives in zip(sorted_archive_files, sorted_model_archive_files):
                                    archive = archives[0]; model_archive = archives[1]
                                    a_sz = int(archive.replace('.','_').split('_')[1])
                                    loc_row_headers = [int(header) for header in row_headers]
                                    diff = lambda l : abs(l - a_sz)
                                    r_val = min(loc_row_headers, key=diff)
                                    r = loc_row_headers.index(r_val)
                                    
                                #for r in range(len(row_headers)):
                                    # if r != len(row_headers) - 1:
                                        # if init_method == 'vanilla':
                                            # continue
                                    #=====================LOAD DATA===========================#
                                    # if False:
                                    #     pass
                                    # # if init_method == 'vanilla':
                                    #     # archive = sorted_archive_files[-1]
                                    #     # model_archive = sorted_model_archive_files[-1]
                                    # else:
                                    #     try:
                                    #         archive = sorted_archive_files[r]
                                    #         try:
                                    #             model_archive = sorted_model_archive_files[r]
                                    #         except:
                                    #             print(f"WARNING: missing model_archive for {archive} with {init_method} on {args.environment} on rep {rep_path}")
                                    #     except:
                                    #         if r != len(row_headers) - 1: 
                                    #             import pdb; pdb.set_trace()
                                    #         else:
                                    #             print(f"WARNING: missing archive at {row_headers[r]} with {init_method} on {args.environment} on rep {rep_path}")
                                    #             continue
                                    archive_path = os.path.join(archive_folder, archive)
                                    if init_method != 'vanilla':
                                        model_archive_path = os.path.join(archive_folder,
                                                                          model_archive)
                                    rep_data = pd.read_csv(archive_path)
                                    rep_data = rep_data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
                                    if init_method != 'vanilla':
                                        model_rep_data = pd.read_csv(model_archive_path)

                                        model_rep_data = model_rep_data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
                                    #=====================ARCHIVE SIZE===========================#
                                    archive_size = len(rep_data.index)
                                    archive_sizes[rep_cpt, r] = archive_size

                                    if init_method != 'vanilla':
                                        model_archive_size = len(model_rep_data.index)
                                        model_archive_sizes[rep_cpt, r] = model_archive_size

                                    #=====================COVERAGE===========================#
                                    
                                    if args.environment == 'hexapod_omni':
                                        for idx in range(len(rep_data.iloc[:,1])):
                                            bd_tab = str_to_tab(rep_data.iloc[:,1][idx],
                                                                dim=dim_map)
                                            for dim in range(1, dim_map + 1):
                                                rep_data.iloc[:,dim][idx] = bd_tab[dim-1]
                                        if init_method != 'vanilla':
                                            for idx in range(len(model_rep_data.iloc[:,1])):
                                                bd_tab = str_to_tab(model_rep_data.iloc[:,1][idx],
                                                                    dim=dim_map)
                                                for dim in range(1, dim_map + 1):
                                                    model_rep_data.iloc[:,dim][idx] = bd_tab[dim-1]
                                                    
                                    df_min = rep_data.iloc[0].copy(); df_max = rep_data.iloc[0].copy()
                                    df_min[1] = ss_min; df_max[1] = ss_max
                                    df_min[2] = ss_min; df_max[2] = ss_max

                                    if args.environment == "ball_in_cup":
                                        df_min[3] = ss_min; df_max[3] = ss_max

                                    ## Deprecated but oh well
                                    rep_data = rep_data.append(df_min, ignore_index = True)
                                    rep_data = rep_data.append(df_max, ignore_index = True)

                                    if init_method != 'vanilla':
                                        model_rep_data = model_rep_data.append(df_min,
                                                                               ignore_index = True)
                                        model_rep_data = model_rep_data.append(df_max,
                                                                               ignore_index = True)
                                    ## Does ugly thingies cba to look at them rn
                                    # data = pd.concat([data, df_min]) 
                                    # data = pd.concat([data, df_max])
                                    
                                    nb_div = args.nb_div

                                    
                                    
                                    rep_data['x_bin']=pd.cut(x = rep_data.iloc[:,1],
                                                             bins = nb_div, 
                                                             labels = [p for p in range(nb_div)])

                                    rep_data['y_bin']=pd.cut(x = rep_data.iloc[:,2],
                                                             bins = nb_div,
                                                             labels = [p for p in range(nb_div)])

                                    if init_method != 'vanilla':
                                        model_rep_data['x_bin']=pd.cut(x= model_rep_data.iloc[:,1],
                                                                       bins = nb_div, 
                                                                       labels = [p for p
                                                                                 in range(nb_div)])
                                        
                                        model_rep_data['y_bin']=pd.cut(x= model_rep_data.iloc[:,2],
                                                                       bins = nb_div,
                                                                       labels = [p for p
                                                                                 in range(nb_div)])

                                    total_bins = nb_div**2

                                    if args.environment == "ball_in_cup":
                                        rep_data['z_bin']=pd.cut(x = rep_data.iloc[:,3],
                                                             bins = nb_div,
                                                             labels = [p for p in range(nb_div)])
                                        if init_method != 'vanilla':
                                            model_rep_data['z_bin']=pd.cut(x=model_rep_data.
                                                                           iloc[:,3],
                                                                           bins = nb_div,
                                                                           labels = [p for p
                                                                                     in range
                                                                                     (nb_div)])

                                        total_bins = nb_div**3

                                    rep_data = rep_data.assign(bins=pd.Categorical
                                                               (rep_data.filter(regex='_bin')
                                                                .apply(tuple, 1)))
                                    if init_method != 'vanilla':
                                        model_rep_data = model_rep_data.assign(bins=pd.Categorical
                                                                               (model_rep_data.
                                                                                filter(
                                                                                    regex='_bin')
                                                                                .apply(tuple, 1)))

                                    counts = rep_data['bins'].value_counts()

                                    coverages[rep_cpt, r] = len(counts[counts>=1])/total_bins

                                    if init_method != 'vanilla':
                                        counts = model_rep_data['bins'].value_counts()

                                        model_coverages[rep_cpt, r] = len(counts
                                                                          [counts>=1])/total_bins

                                ## Old ##
                                # last_archive = [f for f in archive_files if len(re.split('\.|\_', f)[1])==4]
                                # last_archive = last_archive[0]

                                # last_archive_path = os.path.join(archive_folder, last_archive)

                                # rep_data = pd.read_csv(last_archive_path)
                                # rep_data = rep_data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line


                                # archive_size = len(rep_data.index)
                                # archive_sizes[rep_cpt] = archive_size
                                ## ##

                                #=====================PLOT DATA===========================#

                                # print('\nArchive size: ', archive_size, '\n')

                                # fig, ax = plot_archive(rep_data, plt, args, ss_min, ss_max)

                                if args.show:
                                    plt.show() 

                                rep_cpt += 1

                            # mean_archive_size = np.mean(archive_sizes)
                            # std_archive_size = np.std(archive_sizes)
                            # cell_text_size[j][i] = f'{mean_archive_size} \u00B1 {round(std_archive_size,1)}'
                            mean_archive_size = np.nanmean(archive_sizes, axis=0)
                            std_archive_size = np.nanstd(archive_sizes, axis=0)

                            mean_cov = np.nanmean(coverages, axis=0)
                            std_cov = np.nanstd(coverages, axis=0)

                            ## For plotting as a graph
                            archive_mean_sizes[i] = mean_archive_size
                            archive_std_sizes[i] = std_archive_size

                            archive_mean_covs[i] = mean_cov
                            archive_std_covs[i] = std_cov

                            if init_method != 'vanilla':
                                mean_model_archive_size = np.nanmean(archive_sizes, axis=0)
                                std_model_archive_size = np.nanstd(archive_sizes, axis=0)

                                model_mean_cov = np.nanmean(model_coverages, axis=0)
                                model_std_cov = np.nanstd(model_coverages, axis=0)

                                model_archive_mean_sizes[i] = mean_model_archive_size
                                model_archive_std_sizes[i] = std_model_archive_size

                                model_archive_mean_covs[i] = mean_cov
                                model_archive_std_covs[i] = std_cov

                                model_to_real_diff_sizes[i] = np.nanmean(model_archive_sizes -
                                                                      archive_sizes,
                                                                      axis = 0)
                                
                                
                                model_to_real_diff_covs[i] = np.nanmean(model_coverages - coverages,
                                                                     axis = 0)

                            ## For plotting as a tab
                            for r in range(len(row_headers)):
                                cell_text_size[tab_cpt][r][i] = f'{round(mean_archive_size[r],2)} \u00B1 {round(std_archive_size[r],2)}'
                                cell_text_cov[tab_cpt][r][i] = f'{round(mean_cov[r],3)} \u00B1 {round(std_cov[r],3)}'
                                if init_method != 'vanilla':
                                    cell_text_model_size[tab_cpt][r][i] = f'{round(mean_model_archive_size[r],2)} \u00B1 {round(std_model_archive_size[r],2)}'
                                    cell_text_model_cov[tab_cpt][r][i] = f'{round(model_mean_cov[r],3)} \u00B1 {round(model_std_cov[r],3)}'

                            tab_cpt += 1

        cmap = plt.cm.get_cmap('hsv', n_init_method+1)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=n_init_method+1)

        colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        linestyles = ['-', '--', ':', '-.',
                  (0, (5, 10)), (0, (5, 1)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1))]
        
        tab_cpt = 0
        for i in range(n_nb_transfers):
            nb_transfer = nb_transfers[i]
            for j in range(n_transfer_sels):
                transfer_sel = transfer_sels[j]

                #=====================SAVE ARCHIVE SIZE FIGURE===========================#
                fig = plt.figure()

                for k in range(n_init_method):
                    plt.plot(args.dump_vals, archive_mean_sizes[k], label=init_methods[k],
                             color=colors.to_rgba(k), linestyle=linestyles[k])

                    # plt.plot(args.dump_vals, model_archive_mean_sizes[k],
                             # label=f'model_{init_methods[k]}',
                             # color=colors.to_rgba(k), linestyle=linestyles[k], alpha=0.5)

                if has_ns_cov:
                    plt.hlines(ns_asize, args.dump_vals[0], args.dump_vals[-1],
                               label=f'NS-asize-500g', color='g')
                if has_random_data:
                    plt.hlines(random_asize, args.dump_vals[0], args.dump_vals[-1],
                               label=f'rand-cov-{args.random_budget}', color='r')
                    
                plt.legend()

                plt.title(f'Mean archive size on {args.environment} ' \
                          f'environment for\n {transfer_sel} selection with {nb_transfer} ' \
                          f'individuals transferred')
                
                plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_graph_archive_size",
                            dpi=300, bbox_inches='tight')

                if has_pred_errors:
                    #=============SAVE ARCHIVE LAST SIZE VS PRED ERROR FIGURE================#
                    fig = plt.figure()

                    for k in range(n_init_method - 2): ## -2 because we remove vanilla and no init
                        plt.plot(mean_pred_errors[k][2][2], archive_mean_sizes[k][-1], 'o',
                                 label=init_methods[k],
                                 color=colors.to_rgba(k), linestyle=linestyles[k])

                    plt.legend()

                    plt.title(f'Mean archive size after last transfer vs prediction error on ' \
                              f'{args.environment} environment for\n {transfer_sel} ' \
                              f'selection with {nb_transfer} individuals transferred')

                    plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_graph_archive_last_size_vs_pred_error",
                                dpi=300, bbox_inches='tight')

                    #==============SAVE ARCHIVE FIRST SIZE VS PRED ERROR FIGURE===============#
                    fig = plt.figure()

                    for k in range(n_init_method - 2): ## -2 because we remove vanilla and no init
                        plt.plot(mean_pred_errors[k][2][2], archive_mean_sizes[k][0], 'o',
                                 label=init_methods[k],
                                 color=colors.to_rgba(k), linestyle=linestyles[k])

                    plt.legend()

                    plt.title(f'Mean archive size after first transfer vs prediction error on ' \
                              f'{args.environment} environment for\n {transfer_sel} ' \
                              f'selection with {nb_transfer} individuals transferred')

                    plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_graph_archive_first_size_vs_pred_error",
                                dpi=300, bbox_inches='tight')

                #=====================SAVE ARCHIVE DIFF SIZE FIGURE=====================#
                fig = plt.figure()
                for k in range(n_init_method):
                    plt.plot(args.dump_vals, model_to_real_diff_sizes[k], label=init_methods[k],
                             color=colors.to_rgba(k), linestyle=linestyles[k])

                plt.legend()

                plt.title(f'Mean archive size difference between model and real world ' \
                          f'evaluations on {args.environment} ' \
                          f'environment for\n {transfer_sel} selection with {nb_transfer} ' \
                          f'individuals transferred')
                
                plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_graph_archive_size_transfer",
                            dpi=300, bbox_inches='tight')

                #=====================SAVE ARCHIVE DIFF SIZE FIGURE=====================#
                fig = plt.figure()
                for k in range(n_init_method):
                    plt.plot(args.dump_vals, model_to_real_diff_covs[k], label=init_methods[k],
                             color=colors.to_rgba(k), linestyle=linestyles[k])

                plt.legend()

                plt.title(f'Mean archive coverage difference between model and real world ' \
                          f'evaluations on {args.environment} ' \
                          f'environment for\n {transfer_sel} selection with {nb_transfer} ' \
                          f'individuals transferred')
                
                plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_graph_archive_cov_transfer",
                            dpi=300, bbox_inches='tight')
                
                #=====================SAVE ARCHIVE DIFF COV TABLE=======================#
                fig, ax = plt.subplots()
                fig.patch.set_visible(False)
                ax.axis('off')
                ax.axis('tight')
                the_table = plt.table(cellText=cell_text_size[tab_cpt],
                                      rowLabels=row_headers,
                                      rowColours=rcolors,
                                      rowLoc='right',
                                      colColours=ccolors,
                                      colLabels=column_headers,
                                      loc='center')
                fig.tight_layout()
                the_table.auto_set_font_size(False)
                the_table.set_fontsize(4)
                plt.title(f'Mean archive size and standard deviation on {args.environment} ' \
                          f'environment for\n {transfer_sel} selection with {nb_transfer} ' \
                          f'individuals transferred', y=.8)

                plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_quant_archive_size",
                            dpi=300, bbox_inches='tight')

                #=====================SAVE COVERAGE TABLE===========================#
                fig, ax = plt.subplots()
                fig.patch.set_visible(False)
                ax.axis('off')
                ax.axis('tight')
                the_table = plt.table(cellText=cell_text_cov[tab_cpt],
                                      rowLabels=row_headers,
                                      rowColours=rcolors,
                                      rowLoc='right',
                                      colColours=ccolors,
                                      colLabels=column_headers,
                                      loc='center')
                fig.tight_layout()
                the_table.auto_set_font_size(False)
                the_table.set_fontsize(4)
                plt.title(f'Mean coverage and standard deviation on {args.environment} ' \
                          f'environment for\n {transfer_sel} selection with {nb_transfer} ' \
                          f'individuals transferred', y=.8)

                plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_quant_coverage", dpi=300, bbox_inches='tight')
    
                tab_cpt += 1

                #=====================SAVE COVERAGE FIGURE===========================#
                fig = plt.figure()

                for k in range(n_init_method):
                    plt.plot(args.dump_vals, archive_mean_covs[k], label=init_methods[k],
                             color=colors.to_rgba(k), linestyle=linestyles[k])
                if has_ns_cov:
                    plt.hlines(ns_cov, args.dump_vals[0], args.dump_vals[-1], label='NS-cov-500g',
                               color='g')
                if has_random_data:
                    plt.hlines(random_cov, args.dump_vals[0], args.dump_vals[-1],
                               label=f'rand-cov-{args.random_budget}', color='r')
                    
                plt.legend()

                plt.title(f'Mean coverage on {args.environment} ' \
                          f'environment for\n {transfer_sel} selection with {nb_transfer} ' \
                          f'individuals transferred')
                
                plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_graph_coverage",
                            dpi=300, bbox_inches='tight')

                if has_pred_errors:
                    #=============SAVE LAST COVERAGE VS PRED ERROR FIGURE================#
                    fig = plt.figure()

                    for k in range(n_init_method - 2): ## -2 because we remove vanilla and no init
                        plt.plot(mean_pred_errors[k][2][2], archive_mean_covs[k][-1], 'o',
                                 label=init_methods[k],
                                 color=colors.to_rgba(k), linestyle=linestyles[k])

                    plt.legend()

                    plt.title(f'Mean coverage after last transfer vs prediction error on ' \
                              f'{args.environment} environment for\n {transfer_sel} ' \
                              f'selection with {nb_transfer} individuals transferred')

                    plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_graph_archive_last_cov_vs_pred_error",
                                dpi=300, bbox_inches='tight')

                    #==============SAVE FIRST COVERAGE VS PRED ERROR FIGURE===============#
                    fig = plt.figure()

                    for k in range(n_init_method - 2): ## -2 because we remove vanilla and no init
                        plt.plot(mean_pred_errors[k][2][2], archive_mean_covs[k][0], 'o',
                                 label=init_methods[k],
                                 color=colors.to_rgba(k), linestyle=linestyles[k])

                    plt.legend()

                    plt.title(f'Mean coverage after first transfer vs prediction error on ' \
                              f'{args.environment} environment for\n {transfer_sel} ' \
                              f'selection with {nb_transfer} individuals transferred')

                    plt.savefig(f"{args.environment}_{transfer_sel}_{nb_transfer}_graph_archive_first_cov_vs_pred_error",
                                dpi=300, bbox_inches='tight')

