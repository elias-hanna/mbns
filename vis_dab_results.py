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
    elif args.environment == 'walker_2d':
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

def main(args):
    final_asize = 100
    
    ## Set params and rename some 
    gym_env, max_step, ss_min, ss_max, dim_map, bd_inds = process_env(args)

    search_methods = args.search_methods
    sel_methods = args.sel_methods
    m_horizons = args.m_horizons

    
    ## Plot table with mean prediction error for n step predictions    
    column_headers = [init_method for init_method in init_methods]
    # row_headers = [init_episode for init_episode in init_episodes]
    # row_headers = [20, 40, 60, 80, 100]
    row_headers = args.dump_vals
    cell_text_cov = []

    for i in range(n_nb_transfers * n_transfer_sels):
        cell_text_cov.append([["" for _ in range(len(column_headers))]
                              for _ in range(len(row_headers))])
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    ## Get rep folders abs paths
    rep_folders = next(os.walk(f'.'))[1]
    rep_folders = [x for x in rep_folders if (x.isdigit())]

    archive_mean_covs = np.zeros((len(init_methods), len(args.dump_vals)))
    archive_std_covs = np.zeros((len(init_methods), len(args.dump_vals)))

    ## Main loop for printing figures

    cpt_search_method = 0
    for search_method in search_methods:
        cpt_sel_method = 0
        for sel_method in sel methods:
            if sel_method == 'random-policies':
                ## Open each archive file: warning budget is final_asize on these
                pass
            else:
                cpt_m_horizons = 0
                for m_horizon in m_horizon:
                    pass


                
                
        transfer_sel = transfer_sels[m]
        rep_cpt = 0

        coverages = np.empty((len(rep_folders), len(row_headers)))
        coverages[:] = np.nan

        for rep_path in rep_folders:
            archive_folder = f'{rep_path}/{init_method}_{init_episode}_{fitness_func}_{transfer_sel}_{nb_transfer}_results/'
            try:
                archive_files = next(os.walk(archive_folder))[2]
            except Exception as error:
                import pdb; pdb.set_trace()
                excelsior = e
            archive_files = [f for f in archive_files if 'archive' in f]

            archive_numbers = [int(re.findall(r'\d+', f)[0]) for f in archive_files]
            sorted_archive_files = [f for _, f in sorted(zip(archive_numbers, archive_files), key=lambda pair: pair[0])]

            for archives in sorted_archive_files:
                archive = archives[0]
                a_sz = int(archive.replace('.','_').split('_')[1])
                loc_row_headers = [int(header) for header in row_headers]
                diff = lambda l : abs(l - a_sz)
                r_val = min(loc_row_headers, key=diff)
                r = loc_row_headers.index(r_val)

                archive_path = os.path.join(archive_folder, archive)
                rep_data = pd.read_csv(archive_path)
                rep_data = rep_data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
                #=====================COVERAGE===========================#

                if args.environment == 'hexapod_omni':
                    for idx in range(len(rep_data.iloc[:,1])):
                        bd_tab = str_to_tab(rep_data.iloc[:,1][idx],
                                            dim=dim_map)
                        for dim in range(1, dim_map + 1):
                            rep_data.iloc[:,dim][idx] = bd_tab[dim-1]

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

                coverages[rep_cpt, r] = len(counts[counts>=1])/total_bins

            if args.show:
                plt.show() 

            rep_cpt += 1

        mean_cov = np.nanmean(coverages, axis=0)
        std_cov = np.nanstd(coverages, axis=0)

        ## For plotting as a graph
        archive_mean_covs[i] = mean_cov
        archive_std_covs[i] = std_cov

        ## For plotting as a tab
        for r in range(len(row_headers)):
            cell_text_cov[tab_cpt][r][i] = f'{round(mean_cov[r],3)} \u00B1 {round(std_cov[r],3)}'

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

    main(args)
