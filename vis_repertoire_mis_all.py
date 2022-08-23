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

#----------Plot imports--------#
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt


def key_event(event, args):
    if event.key == 'escape':
        sys.exit(0)

def click_event(event, args):
    '''
    # reutrns a list of tupples of x-y points
    click_in = plt.ginput(1,-1) # one click only, should not timeout
    print("click_in: ",click_in)
    
    selected_cell = [int(click_in[0][0]), int(click_in[0][1])]
    print(selected_cell)
    selected_x = selected_cell[0]
    selected_y = selected_cell[1]
    '''
    #event.button ==1 is the left mouse click
    if event.button == 1:
        selected_x = int(event.xdata)
        selected_y = int(event.ydata)
        selected_solution = data[(data["x_bin"] == selected_x) & (data["y_bin"] == selected_y)]
        #selected_solution = data[(data["y_bin"] == selected_x) & (data["z_bin"] == selected_y)]

        # For hexapod omnitask
        print("SELECTED SOLUTION SHAPE: ",selected_solution.shape)
        selected_solution = selected_solution.iloc[0, :]
        print("Selected solution shape: ", selected_solution.shape)
        selected_ctrl = selected_solution.iloc[4:-2].to_numpy()
        print(selected_ctrl.shape) #(1,36)

        #hexapod uni
        #selected_solution = selected_solution.iloc[0, :]
        #selected_ctrl = selected_solution.iloc[8:-2].to_numpy()
        
        #print("Selected ctrl shape: ", selected_ctrl.shape) # should be 3661
        print("Selected descriptor bin: " ,selected_x, selected_y)
        print("Selected descriptor from archive: ", selected_solution.iloc[1:3].to_numpy())
        #print("Selected fitness from archive: ", selected_solution.iloc[0])

        # ---- SIMULATE THE SELECTED CONTROLLER -----#
        #simulate(selected_ctrl, 5.0, render=True) # Hexapod
        #env.evaluate_solution(selected_ctrl)
        
        #fit, desc, _, _ = env.evaluate_solution_uni(selected_ctrl, render=True)
        #print("fitness from simulation real eval:", fit)
        #fit, desc, _, _ = env.evaluate_solution_model_uni(selected_ctrl)
        #print("fitness from dynamics model :", fit)
        
        fit, desc, _, _ = env.evaluate_solution(selected_ctrl, render=True)
        #simulate(selected_ctrl, render=True) # panda bullet
        #simulate(selected_ctrl, 5.0, render=True) # panda dart 
        #evaluate_solution(selected_ctrl, gui=True) 
        print("SIMULATION DONE")


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

    parser.add_argument('--transfer-selections', nargs="*",
                        type=str, default=['all'])
    parser.add_argument('--fitness-funcs', nargs="*",
                        type=str, default=['energy_minimization', 'disagr_minimization'])
    
    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')
    parser.add_argument('--dump-path', type=str, default='default_dump/')
    parser.add_argument("--filename", type=str) # file to visualize rollouts from
    # parser.add_argument("--sim_time", type=float, help="simulation time depending on the type of archive you chose to visualize, 3s archive or a 5s archive")
    parser.add_argument("--show", help="Show the plot and saves it. Unless specified, just save the mean plot over all repetitions",
                    action="store_true")
    parser.add_argument("--plot_type", type=str, default="scatter", help="scatter plot, grid plot or 3d")
    parser.add_argument("--nb_div", type=int, default=100, help="Number of equally separated bins to divide the outcome space in")

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

        gym_env, max_step, ss_min, ss_max, dim_map = process_env(args)

        ## Plot table with mean prediction error for n step predictions    
        column_headers = [init_method for init_method in init_methods]
        # row_headers = [init_episode for init_episode in init_episodes]
        row_headers = [200, 400, 600, 800, 1000]
        cell_text_size = [["" for _ in range(len(column_headers))]
                          for _ in range(len(row_headers))]
        cell_text_cov = [["" for _ in range(len(column_headers))]
                         for _ in range(len(row_headers))]
        rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

        ## Get rep folders abs paths
        rep_folders = next(os.walk(f'.'))[1]
        rep_folders = [x for x in rep_folders if (x.isdigit())]
        
        for j in range(n_init_episodes):
            init_episode = init_episodes[j]
            for i in range(n_init_method):
                init_method =  init_methods[i]
                for k in range(n_fitness_funcs):
                    fitness_func = fitness_funcs[k]
                    rep_cpt = 0
                    mean_archive_size = 0
                    # archive_sizes = np.empty((len(rep_folders)))
                    archive_sizes = np.empty((len(rep_folders), len(row_headers)))
                    coverages = np.empty((len(rep_folders), len(row_headers)))
                    for rep_path in rep_folders:
                        archive_folder = f'{rep_path}/{init_method}_{init_episode}_{fitness_func}_results/'
                        archive_files = next(os.walk(archive_folder))[2]
                        archive_files = [f for f in archive_files if 'archive' in f]
                        archive_numbers = [int(re.findall(r'\d+', f)[0]) for f in archive_files]
                        sorted_archive_files = [f for _, f in sorted(zip(archive_numbers, archive_files), key=lambda pair: pair[0])]
                        
                        for r in range(len(row_headers)):
                            #=====================LOAD DATA===========================#
                            archive = sorted_archive_files[r]
                            archive_path = os.path.join(archive_folder, archive)

                            rep_data = pd.read_csv(archive_path)
                            rep_data = rep_data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
                            
                            #=====================ARCHIVE SIZE===========================#
                            archive_size = len(rep_data.index)
                            archive_sizes[rep_cpt, r] = archive_size

                            #=====================COVERAGE===========================#
    
                            df_min = rep_data.iloc[0].copy(); df_max = rep_data.iloc[0].copy()
                            df_min[1] = ss_min; df_max[1] = ss_max
                            df_min[2] = ss_min; df_max[2] = ss_max

                            if args.environment == "ball_in_cup":
                                df_min[3] = ss_min; df_max[3] = ss_max
                            
                            # Deprecated
                            rep_data = rep_data.append(df_min, ignore_index = True)
                            rep_data = rep_data.append(df_max, ignore_index = True)
                            # data = pd.concat([data, df_min]) ## does ugly thingies cba to look at them rn
                            # data = pd.concat([data, df_max])
                            nb_div = 10

                            rep_data['x_bin']=pd.cut(x = rep_data.iloc[:,1],
                                                 bins = nb_div, 
                                                 labels = [p for p in range(nb_div)])

                            bin_filled = [0, 0, 0]

                            counts = rep_data['x_bin'].value_counts()
                            for c in range(len(counts)):
                                if c != 0 and c != 99 and counts[c] >= 1:
                                    bin_filled[0] += 1
                                elif counts[c] >= 2:
                                    bin_filled[0] += 1
                                    
                            rep_data['y_bin']=pd.cut(x = rep_data.iloc[:,2],
                                                 bins = nb_div,
                                                 labels = [p for p in range(nb_div)])

                            counts = rep_data['y_bin'].value_counts()
                            for c in range(len(counts)):
                                if c != 0 and c != 99 and counts[c] >= 1:
                                    bin_filled[1] += 1
                                elif counts[c] >= 2:
                                    bin_filled[1] += 1

                            coverages[rep_cpt, r] = sum(bin_filled)/nb_div**2

                            total_bins = nb_div**2
                            
                            if args.environment == "ball_in_cup":
                                rep_data['z_bin']=pd.cut(x = rep_data.iloc[:,3],
                                                     bins = nb_div,
                                                     labels = [p for p in range(nb_div)])

                                counts = rep_data['z_bin'].value_counts()
                                for c in range(len(counts)):
                                    if c != 0 and c != 99 and counts[c] >= 1:
                                        bin_filled[2] += 1
                                    elif counts[i] >= 2:
                                        bin_filled[2] += 1
                            
                                coverages[rep_cpt, r] = sum(bin_filled)/nb_div**3
                                total_bins = nb_div**3
                                
                            rep_data = rep_data.assign(cartesian=pd.Categorical
                                                       (rep_data.filter(regex='_bin')
                                                        .apply(tuple, 1)))

                            counts = rep_data['cartesian'].value_counts()

                            coverages[rep_cpt, r] = len(counts[counts>=1])/total_bins
                            
                        # last_archive = [f for f in archive_files if len(re.split('\.|\_', f)[1])==4]
                        # last_archive = last_archive[0]

                        # last_archive_path = os.path.join(archive_folder, last_archive)
                        
                        # rep_data = pd.read_csv(last_archive_path)
                        # rep_data = rep_data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line


                        # archive_size = len(rep_data.index)
                        # archive_sizes[rep_cpt] = archive_size


                        #=====================PLOT DATA===========================#

                        # print('\nArchive size: ', archive_size, '\n')

                        # fig, ax = plot_archive(rep_data, plt, args, ss_min, ss_max)

                        if args.show:
                            plt.show() 

                        rep_cpt += 1

                    # mean_archive_size = np.mean(archive_sizes)
                    # std_archive_size = np.std(archive_sizes)
                    # cell_text_size[j][i] = f'{mean_archive_size} \u00B1 {round(std_archive_size,1)}'
                    mean_archive_size = np.mean(archive_sizes, axis=0)
                    std_archive_size = np.std(archive_sizes, axis=0)

                    mean_cov = np.mean(coverages, axis=0)
                    std_cov = np.std(coverages, axis=0)

                    for r in range(len(row_headers)):
                        cell_text_size[r][i] = f'{round(mean_archive_size[r],1)} \u00B1 {round(std_archive_size[r],1)}'
                        cell_text_cov[r][i] = f'{round(mean_cov[r],3)} \u00B1 {round(std_cov[r],3)}'

        #=====================SAVE ARCHIVE SIZE TABLE===========================#
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        the_table = plt.table(cellText=cell_text_size,
                              rowLabels=row_headers,
                              rowColours=rcolors,
                              rowLoc='right',
                              colColours=ccolors,
                              colLabels=column_headers,
                              loc='center')
        fig.tight_layout()
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(6)
        plt.title(f'Mean archive size and standard deviation on {args.environment} environment', y=.7)
        
        plt.savefig(f"{args.environment}_quant_archive_size", dpi=300, bbox_inches='tight')
    
        #=====================SAVE COVERAGE TABLE===========================#
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        the_table = plt.table(cellText=cell_text_cov,
                              rowLabels=row_headers,
                              rowColours=rcolors,
                              rowLoc='right',
                              colColours=ccolors,
                              colLabels=column_headers,
                              loc='center')
        fig.tight_layout()
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(6)
        plt.title(f'Mean coverage and standard deviation on {args.environment} environment', y=.7)
        
        plt.savefig(f"{args.environment}_quant_coverage", dpi=300, bbox_inches='tight')
    
        
        
