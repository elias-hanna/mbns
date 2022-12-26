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

#----------Clustering imports--------#
from sklearn.cluster import KMeans

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

def plot_archive(data, plt, args, ss_min, ss_max, bounds=False, name="", bd_col=0, c_by='bd',
                 real_and_model=False, bd_cols=['bd0', 'bd1']):
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

        if not real_and_model:
            # FOR JUST A SCATTER PLOT OF THE DESCRIPTORS - doesnt work for interactive selection
            #data.plot.scatter(x=2,y=3,c=0,colormap='Spectral', s=2, ax=ax, vmin=-0.1, vmax=1.2)
            # data.plot.scatter(x=1,y=2,c=0,colormap='viridis', s=2, ax=ax) # color by fitness
            data.plot.scatter(x=bd_cols[0],y=bd_cols[1],c='fit',colormap='viridis', s=10, ax=ax)
            plt.title('Trajectories end-point of model archive')
        else:
            from colour import Color
            # red = Color("red")
            # colors = list(red.range_to(Color("green"),len(data)))
            # colors = [c.hex for c in colors]

            red = Color("red")
            green = Color("green")
            blue = Color("blue")
            yellow = Color("yellow")

            colors = []
            colors += list(red.range_to(green,len(data)//3))
            colors += list(green.range_to(blue,len(data)//3))
            colors += list(blue.range_to(yellow,len(data)//3))
            if len(colors) < len(data):
                colors += [red]

            colors = [c.hex for c in colors]

            # data.plot.scatter(x='bd0_model',y='bd1_model',c=colors,s=2, ax=ax)
            # fig, ax = plt.subplots()
            # data.plot.scatter(x='bd0_real_all',y='bd1_real_all',c=colors,s=2, ax=ax)
            # data.plot.scatter(x='bd0_real_added',y='bd1_real_added',c=colors,s=2, ax=ax)
            if c_by == 'bd':
                data.plot.scatter(x=f'{bd_cols[0]}_model',y=f'{bd_cols[1]}_model',
                                  c=f'bd{bd_col}_model',
                                  colormap='viridis',s=10, ax=ax)
            elif c_by == 'fit':
                data.plot.scatter(x=f'{bd_cols[0]}_model',y=f'{bd_cols[1]}_model',
                                  c='fit_model',
                                  colormap='viridis',s=10, ax=ax)
            elif c_by == 'idx':
                data.plot.scatter(x=f'{bd_cols[0]}_model',y=f'{bd_cols[1]}_model',
                                  c=colors,s=10, ax=ax)
            plt.title(f'Trajectories end-point of {name} archive on model')

            if name:
                fig, ax = plt.subplots()
                if c_by == 'bd':
                    data.plot.scatter(x=f'{bd_cols[0]}_{name}',y=f'{bd_cols[1]}_{name}',
                                      c=f'bd{bd_col}_model',
                                      colormap='viridis',s=10, ax=ax)
                elif c_by == 'fit':
                    data.plot.scatter(x=f'{bd_cols[0]}_{name}',y=f'{bd_cols[1]}_{name}',
                                      c=f'fit_{name}',
                                      colormap='viridis',s=10, ax=ax)
                elif c_by == 'idx':
                    data.plot.scatter(x=f'{bd_cols[0]}_{name}',y=f'{bd_cols[1]}_{name}',
                                      c=colors,s=10, ax=ax)
            plt.title(f'Trajectories end-point of {name} archive on real system')

        if bounds:
            plt.xlim(ss_min,ss_max)
            plt.ylim(ss_min,ss_max)
            
            #data.plot.scatter(x=1,y=2,s=2, ax=ax[0])
            #data.plot.scatter(x=3,y=4,c=0,colormap='viridis', s=2, ax=ax)
            #data.plot.scatter(x=4,y=5,s=2, ax=ax[1])
            #plt.xlim(-0.5,0.5)
            #plt.ylim(-0.5,0.5)

    return fig, ax

def rename_df(df, dim_map):
    rename_dict = {}
    old_col_names = df.columns
    cpt = 0
    for old_name in old_col_names:
        if cpt == 0:
            new_name = 'fit'
        elif cpt < dim_map + 1:
            new_name = f'bd{cpt-1}'
        else:
            new_name = f'x{cpt-1-dim_map}'
        rename_dict[old_name] = new_name
        cpt += 1
        
    df.rename(columns=rename_dict, inplace=True)

def get_closest_to_cluster_centroid(data, n_clusters=100, bd_cols=['bd0', 'bd1']):
    # Use k-means clustering to divide the data space into 10 clusters
    kmeans = KMeans(n_clusters=n_clusters)
    # kmeans.fit(data[['bd0', 'bd1']])
    kmeans.fit(data[bd_cols])
    
    # Select closest data point from each cluster
    data_centers = pd.DataFrame(columns=data.columns)

    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        cluster = data[kmeans.labels_ == i]
        # cluster = data[data['cluster'] == i]
        # Calculate the distance of each point in the cluster to the cluster center
        cluster['distance'] = ((cluster[bd_cols[0]] - cluster_center[0])**2 + (cluster[bd_cols[1]] - cluster_center[1])**2)**0.5
        # Select the point with the minimum distance to the cluster center
        closest_point = cluster.loc[cluster['distance'].idxmin()]
        data_centers = data_centers.append(closest_point)

    return data_centers

def get_novelty_scores(data, k=15, bd_cols=['bd0', 'bd1']):
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

def get_most_nov_data(data, n=100, bd_cols=['bd0', 'bd1'], ens_size=1):
    if ens_size > 1:
        ens_data_nov = np.empty((ens_size, len(data)))
        for i in range(ens_size):
            bd_cols_loc = [f"{bd_cols[k]}_m{i}" for k in range(len(bd_cols))]
            data_nov = get_novelty_scores(data, bd_cols=bd_cols_loc)
            ens_data_nov[i,:] = data_nov
        data_nov = ens_data_nov.min(axis=0)
    else:
        data_nov = get_novelty_scores(data, bd_cols=bd_cols)
    sorted_data = data.assign(nov=data_nov)
    sorted_data = sorted_data.sort_values(by=['nov'], ascending=False)
    return sorted_data.head(n=n)
    
def main(args):
    gym_env, max_step, ss_min, ss_max, dim_map, bd_inds = process_env(args)
    # dim_x = 36
    
    data = pd.read_csv(args.filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
    dim_x = len([col for col in data.columns if 'x' in col])
    bds = [col for col in data.columns if 'bd' in col]
    ## keep only the final two because others are waypoints
    bd_cols = bds[-2:]
    ## Need to filter the mx
    if args.ens_size > 1:
        splitted_cols = [bd_col.split('_') for bd_col in bd_cols]
        no_m_bd_cols = [el[0] for el in splitted_cols]

    if args.show_model_trajs:
        fig, ax = plt.subplots()
        for idx in range(len(data)):
            traj_data = np.load(data.loc[idx]['ind_trajs'])
            bd_traj = np.take(traj_data['obs_traj'], bd_inds, axis=1)
            bdx = bd_traj[:,0]; bdy = bd_traj[:,1]
            ax.plot(bdx, bdy, alpha=0.1, marker='o')
            # print(data.loc[idx]['ind_trajs'])
            # print(traj_data['obs_traj'])
            # print(bd_traj)
            # exit()
        plt.title('Individuals trajectories on model')
        
    if args.model_and_real:
        n_inds = 100 if len(data) >= 100 else len(data)
        ## Get dataframes satisfying selection condition
        if args.ens_size > 1:
            data_most_nov = get_most_nov_data(data, n=n_inds, bd_cols=no_m_bd_cols,
                                              ens_size=args.ens_size)

        
            data = data.rename(
                columns={bd_cols[0]:no_m_bd_cols[0], bd_cols[1]:no_m_bd_cols[1]})
            data_most_nov = data_most_nov.rename(
                columns={bd_cols[0]:no_m_bd_cols[0], bd_cols[1]:no_m_bd_cols[1]})
            bd_cols = no_m_bd_cols
        else:
            data_most_nov = get_most_nov_data(data, n=n_inds, bd_cols=bd_cols,
                                              ens_size=args.ens_size)
        if not args.ens_size > 1:
            data_centers = get_closest_to_cluster_centroid(data, n_clusters=n_inds, bd_cols=bd_cols)
        
        ## Get other dataframes corresponding to real env conditions
        splitted_name = args.filename.split('.')
        real_added_data_filename = splitted_name[0] + "_real_added.dat"
        real_all_data_filename = splitted_name[0] + "_real_all.dat"

        data_real_added = pd.read_csv(real_added_data_filename)
        data_real_all = pd.read_csv(real_all_data_filename)

        data_real_added = data_real_added.iloc[:,:-1]
        data_real_all = data_real_all.iloc[:,:-1]

        ## reorder the dfs
        # Merge the two dataframes on the "id" and "name" columns
        gen_cols = [f'x{i}' for i in range(dim_x)]

        df_merged_all = data.merge(data_real_all, on=gen_cols,
                                   suffixes=('_model','_real_all'))

        if not args.ens_size > 1:
            df_merged_centers = data_centers.merge(data_real_all, on=gen_cols,
                                                   suffixes=('_model','_real_centers'))

        df_merged_nov = data_most_nov.merge(data_real_all, on=gen_cols,
                                            suffixes=('_model','_real_nov'))

        df_merged_added = data.merge(data_real_added, on=gen_cols,
                                     suffixes=('_model','_real_added'))

        fig, ax = plot_archive(df_merged_all, plt, args, ss_min, ss_max,
                               bounds=args.bounds, name="real_all",
                               c_by='fit', bd_col=1, real_and_model=True, bd_cols=bd_cols)
        if not args.ens_size > 1:
            fig, ax = plot_archive(df_merged_centers, plt, args, ss_min, ss_max,
                                   bounds=args.bounds, name="real_centers",
                                   c_by='fit', bd_col=1, real_and_model=True, bd_cols=bd_cols)
        fig, ax = plot_archive(df_merged_nov, plt, args, ss_min, ss_max,
                               bounds=args.bounds, name="real_nov",
                               c_by='fit', bd_col=1, real_and_model=True, bd_cols=bd_cols)
        fig, ax = plot_archive(df_merged_added, plt, args, ss_min, ss_max,
                               bounds=args.bounds, name="real_added",
                               c_by='fit', bd_col=1, real_and_model=True, bd_cols=bd_cols)
        # fig, ax = plot_archive(data_real_all, plt, args, ss_min, ss_max, bounds=True)
        # fig, ax = plot_archive(data_real_added, plt, args, ss_min, ss_max, bounds=True)

    else:
        fig, ax = plot_archive(data, plt, args, ss_min, ss_max, bounds=args.bounds)
        
    #=====================PLOT DATA===========================#

    archive_size = len(data.index)
    print('\nArchive size: ', archive_size, '\n')

    plt.show() 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', '-e', type=str, default='hexapod_omni')
    parser.add_argument("--filename", type=str) # file to visualize rollouts from
    # parser.add_argument("--sim_time", type=float, help="simulation time depending on the type of archive you chose to visualize, 3s archive or a 5s archive")
    parser.add_argument("--show", help="Show the plot and saves it. Unless specified, just save the mean plot over all repetitions", action="store_true")
    parser.add_argument('--model-and-real', action="store_true")
    parser.add_argument('--show-model-trajs', action="store_true")
    parser.add_argument('--bounds', action="store_true")

    parser.add_argument("--ens-size", type=int, default=1, help="Number of models in ensemble")
    parser.add_argument("--plot_type", type=str, default="scatter", help="scatter plot, grid plot or 3d")
    parser.add_argument("--nb_div", type=int, default=10, help="Number of equally separated bins to divide the outcome space in")

    args = parser.parse_args()

    if args.filename is not None:
        main(args)
    else:
        raise FileNotFoundError("Missing archive file after --filename")
