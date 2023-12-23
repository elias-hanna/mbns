## Local imports
from exps_utils import get_env_params, process_args, plot_cov_and_trajs, \
    save_archive_cov_by_gen

#----------Utils imports--------#
import os, sys
import argparse
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from multiprocessing import Pool

#----------Data manipulation imports--------#
import numpy as np
import copy
import pandas as pd
import itertools

################################################################################
#################### WARNING: Launch from data folder ##########################
################################################################################

def process_plot_args(args):
    plot_params = {}
    plot_params['nb_div'] = args.nb_div
    plot_params['ps_methods'] = args.ps_methods
    plot_params['n_ps_methods'] = len(args.ps_methods)
    plot_params['n_reps'] = args.n_reps
    return plot_params

def pd_read_csv_fast(filename):
    ## Only read the first line to get the columns
    data = pd.read_csv(filename, nrows=1)
    ## Only keep important columns and 5 genotype columns for merge purposes
    usecols = [col for col in data.columns if 'bd' in col or 'fit' in col]
    # usecols += [col for col in data.columns if 'x' in col][:5]
    usecols += [col for col in data.columns if 'x' in col][:2]
    ## Return the complete dataset (without the << useless >> columns
    # return pd.read_csv(filename, usecols=usecols, encoding_errors='ignore')
    return pd.read_csv(filename, usecols=usecols, engine='python')
    # return pd.read_csv(filename, usecols=usecols)

def filter_archive_fnames(cwd, ps_method):
    # get files in cwd
    try:
        all_fnames = next(os.walk(cwd))[2]
    except:
        import pdb; pdb.set_trace()
    # remove files that are not archive files
    all_archive_fnames = [fname for fname in all_fnames
                          if 'archive' in fname]
    # remove files that don't end with .dat
    all_archive_fnames = [fname for fname in all_archive_fnames
                          if '.dat' in fname[-4:]]
    # remove real_all files (keep only archive_budget.dat files)
    all_archive_fnames = [fname for fname in all_archive_fnames
                          if '_real_all.dat' not in fname]

    if ps_method == 'ns' or 'mbns' in ps_method:
        # remove archive_eval.dat files, keep only archive_evals_all_evals.dat
        # which corresponds to unstructured archives
        all_archive_fnames = [fname for fname in all_archive_fnames
                              if '_all_evals.dat' in fname]
        
    # now keep only the one with highest evaluation number
    # copy the file names and replace with _ for easier split 
    fnames_copy = [copy.copy(fname).replace('.', '_') for fname in all_archive_fnames]
    # split and keep only the evaluation number
    fnames_copy = [fname.split('_')[1] for fname in all_archive_fnames]
    # get highest eval number archive
    try:
        last_archive_idx = np.argmax(fnames_copy)
    except:
        print(f"No suitable archive found at path: {cwd}")
        return None
    abs_fname = os.path.join(cwd, all_archive_fnames[last_archive_idx])
    return abs_fname

################################################################################
##################### Coverage computation functions ###########################
################################################################################

def get_data_bins(data, ss_min, ss_max, dim_map, bd_inds, nb_div):
    df_min = data.iloc[0].copy(); df_max = data.iloc[0].copy()

    for i in range(dim_map):
        df_min[f'bd{i}'] = ss_min[bd_inds[i]]
        df_max[f'bd{i}'] = ss_max[bd_inds[i]]

    ## Deprecated but oh well
    data = data.append(df_min, ignore_index = True)
    data = data.append(df_max, ignore_index = True)

    for i in range(dim_map):
        # print(data)
        # data[f'{i}_bin'] = data[f'{i}_bin'].astype(str)
        # pd.set_option('display.max_rows', data.shape[0]+1)
        # print(data)
        data[f'{i}_bin'] = pd.cut(x = data[f'bd{i}'],
                                  bins = nb_div, 
                                  labels = [p for p in range(nb_div)])

    ## assign data to bins
    data = data.assign(bins=pd.Categorical
                       (data.filter(regex='_bin')
                        .apply(tuple, 1)))

    ## remove df min and df max
    data.drop(data.tail(2).index,inplace=True)

    return data

def compute_cov(data, ss_min, ss_max, dim_map, bd_inds, nb_div):
    ## add bins field to data
    data = get_data_bins(data, ss_min, ss_max, dim_map, bd_inds, nb_div)
    ## count number of bins filled
    counts = data['bins'].value_counts()
    total_bins = nb_div**dim_map
    ## return coverage (number of bins filled)
    return len(counts[counts>=1])/total_bins

def get_cov_per_gen(bds_per_gen, ss_min, ss_max, dim_map, bd_inds, nb_div, total_evals, ps_method):
    dict_vals = bds_per_gen.values()
    only_gen_num = [int(s.split('_')[1]) for s in bds_per_gen.keys()]
    max_gen = max(only_gen_num)
    gen_covs = []
    gen_evals = []
    step = (total_evals-100)//(max_gen-1)
    for (gen, n_evals) in zip(range(1, max_gen+1), range(100, total_evals+step, step)):
        gen_bd_data = bds_per_gen[f'bd_{gen}']
        gen_bd_df = pd.DataFrame(gen_bd_data, columns=[f'bd{i}' for i in range(dim_map)])
        gen_cov = compute_cov(gen_bd_df, ss_min, ss_max, dim_map, bd_inds, nb_div)
        gen_covs.append(gen_cov)
        gen_evals.append(n_evals)

    return gen_covs, gen_evals

def get_cov_per_gen_safe(bds_per_gen, ss_min, ss_max, dim_map, bd_inds, nb_div, total_evals, ps_method):
    dict_vals = bds_per_gen.values()
    only_gen_num = [int(s.split('_')[1]) for s in bds_per_gen.keys()]
    max_gen = max(only_gen_num)
    gen_covs = []
    gen_evals = []
    step = 100
    gen = 1
    range_step = max_gen // step if max_gen // step > 0 else 1
    step = total_evals // max_gen
    # for (gen, n_evals) in zip(range(1, max_gen+1, range_step), range(100, total_evals+total_evals//step, total_evals//step)):    
    for (gen, n_evals) in zip(range(1, max_gen+1), range(100, total_evals+total_evals//step, step)):    
    # for n_evals in range(100, total_evals+total_evals//step, total_evals//step):
        gen_bd_data = bds_per_gen[f'bd_{gen}']
        gen_bd_df = pd.DataFrame(gen_bd_data, columns=[f'bd{i}' for i in range(dim_map)])
        gen_cov = compute_cov(gen_bd_df, ss_min, ss_max, dim_map, bd_inds, nb_div)
        gen_covs.append(gen_cov)
        gen_evals.append(n_evals)
        # gen += 1

    return gen_covs, gen_evals

def get_err_per_gen_safe(errors_per_gen, total_evals, ps_method):
    dict_vals = errors_per_gen.values()
    max_gen = len(errors_per_gen['all_errors_medians'])
    ## Extract all error data
    all_errors_medians = errors_per_gen['all_errors_medians']
    all_errors_1q = errors_per_gen['all_errors_1q']
    all_errors_3q = errors_per_gen['all_errors_3q']
    add_errors_medians = errors_per_gen['add_errors_medians']
    add_errors_1q = errors_per_gen['add_errors_1q']
    add_errors_3q = errors_per_gen['add_errors_3q']
    discard_errors_medians = errors_per_gen['discard_errors_medians']
    discard_errors_1q = errors_per_gen['discard_errors_1q']
    discard_errors_3q = errors_per_gen['discard_errors_3q']
    
    gen_evals = np.linspace(100, total_evals, max_gen)
    
    return (all_errors_medians, all_errors_1q, all_errors_3q), \
        (add_errors_medians, add_errors_1q, add_errors_3q), \
        (discard_errors_medians, discard_errors_1q, discard_errors_3q), gen_evals


def compute_qd_score(data):
    return data['fit'].sum()

def process_rep(var_tuple):
    rep_dir, ps_method, ss_min, ss_max, dim_map, bd_inds, nb_div, bd_cols = var_tuple
    ## get the last archive file name
    archive_fname = filter_archive_fnames(rep_dir, ps_method)
    if archive_fname is None:
        return np.nan, [], [], []
    ## Load it as a pandas dataframe
    archive_data = pd_read_csv_fast(archive_fname)
    # drop the last column which was made because there is a
    # comma after last value i a line
    archive_data = archive_data.iloc[:,:-1]

    archive_size = len(archive_data)

    ## Compute coverage and save it
    final_cov = compute_cov(archive_data, ss_min,
                            ss_max, dim_map,
                            bd_inds, nb_div)

    final_qd_score = compute_qd_score(archive_data)
    
    ## Get coverage and evals per gen
    if ps_method == 'ns' or 'mbns' in ps_method:
        bds_per_gen_fpath = os.path.join(rep_dir, 'bds_per_gen_all_evals.npz')
    else:
        bds_per_gen_fpath = os.path.join(rep_dir, 'bds_per_gen.npz')
    gen_covs = []
    gen_evals = []
    try:
        bds_per_gen = np.load(bds_per_gen_fpath)
    
        total_evals = int(archive_fname.split('/')[-1].replace('.', '_').split('_')[1])
        gen_covs, gen_evals = get_cov_per_gen_safe(bds_per_gen, ss_min,
                                                   ss_max, dim_map,
                                                   bd_inds, nb_div,
                                                   total_evals, ps_method)
    except Exception as e:
        print(f'Received error {e} when processing file: {bds_per_gen_fpath}')

    ## Get desc error per gen (put it in perspective with evals)
    if ps_method == 'daqd' or 'mbns' in ps_method:
        errors_per_gen_fpath = os.path.join(rep_dir, 'desc_estimation_errors.npz')
        try:
            errors_per_gen = np.load(errors_per_gen_fpath)
            total_evals = int(archive_fname.split('/')[-1].replace('.', '_').split('_')[1])
            gen_all_err, gen_add_err, gen_discard_err, gen_evals_err = get_err_per_gen_safe(
                errors_per_gen, total_evals, ps_method)
        except Exception as e:
            print(f'Received error {e} when processing file: {bds_per_gen_fpath}')
    else:
        gen_all_err, gen_add_err, gen_discard_err = (0,1,2), (0,1,2), (0,1,2)
        gen_evals_err = []
    ## Get archive BDs array and save it
    archive_bds = archive_data[bd_cols].to_numpy()

    return final_cov, gen_covs, gen_evals, archive_bds, final_qd_score, \
        archive_size, gen_all_err, gen_add_err, gen_discard_err, gen_evals_err

def main(args):
    ## Process command-line args
    plot_params = process_plot_args(args)
    nb_div = plot_params['nb_div'] 
    ps_methods = plot_params['ps_methods']  
    n_ps_methods = plot_params['n_ps_methods']  
    n_reps = plot_params['n_reps']

    env_params = get_env_params(args)

    env_name = args.environment
    is_local_env = env_params['is_local_env'] 
    gym_args = env_params['gym_args']  
    env_register_id = env_params['env_register_id']
    a_min = env_params['a_min'] 
    a_max = env_params['a_max'] 
    ss_min = env_params['ss_min']
    ss_max = env_params['ss_max']
    init_obs = env_params['init_obs'] 
    state_dim = env_params['state_dim']
    obs_min = env_params['obs_min']
    obs_max = env_params['obs_max']
    dim_map = env_params['dim_map']
    bd_inds = env_params['bd_inds']
    bins = env_params['bins'] ## for grid based qd

    ## Cast state space lims in float in case they're not already
    ss_min = ss_min.astype(np.float64)
    ss_max = ss_max.astype(np.float64)
    ## Correct state space for plotting on hexapod
    if env_name == 'hexapod_omni':
        # ss_min = (ss_min+1.5)/3  
        # ss_max = (ss_max+1.5)/3
        # ss_min[:] = 0
        # ss_max[:] = 1

        ## Old work good
        # ss_min[bd_inds[0]] = 0.15 ; ss_min[bd_inds[1]] = 0.2
        # ss_max[bd_inds[0]] = 0.86 ; ss_max[bd_inds[1]] = 0.82

        # ss_min[bd_inds[0]] = 0 ; ss_min[bd_inds[1]] = 0
        # ss_max[bd_inds[0]] = 1 ; ss_max[bd_inds[1]] = 1

        ss_min[bd_inds[0]] = 0.13 ; ss_min[bd_inds[1]] = 0.18
        ss_max[bd_inds[0]] = 0.91 ; ss_max[bd_inds[1]] = 0.81


        # min mbns: [0.15273238 0.21077263]
        # max mbns: [0.85483918 0.81122953]
        
    ## get bd cols
    bd_cols = [f'bd{i}' for i in range(dim_map)]
    ## Create main data holders
    all_psm_bds = []
    coverages = np.empty((n_ps_methods, n_reps))
    coverages[:] = np.nan
    qd_scores = np.empty((n_ps_methods, n_reps))
    qd_scores[:] = np.nan
    archive_sizes = np.empty((n_ps_methods, n_reps))
    archive_sizes[:] = np.nan
    
    # Get current working dir (folder from which py script was executed)
    root_wd = os.getcwd()

    covs_per_gen = []
    evals_per_gen = []

    all_errors_per_gen = []
    add_errors_per_gen = []
    discard_errors_per_gen = []
    evals_per_gen_err = []
    
    pool = Pool(10) ## use 10 cores (10 reps...)
    
    ## Go over methods (daqd, qd, qd_grid, ns)
    for (ps_method, psm_cpt) in zip(ps_methods, range(len(ps_methods))):
        covs_per_gen.append([])
        evals_per_gen.append([])
        all_errors_per_gen.append([])
        add_errors_per_gen.append([])
        discard_errors_per_gen.append([])
        evals_per_gen_err.append([])
        
        print(f"Processing {ps_method} results on {args.environment}...")
        ## Go inside the policy search method folder
        # new working dir
        if 'daqd' in ps_method or 'mbns' in ps_method:
            method_wd = os.path.join(root_wd,
                                     f'{env_name}_{ps_method}_results/ffnn_2l_10n_prob_4_h-1_1wps_results/')
        else:
            method_wd = os.path.join(root_wd,
                                     f'{env_name}_{ps_method}_results/ffnn_2l_10n_1wps_results/') 
        # get rep dirs (from method_wd pov)
        try:
            rep_dirs = next(os.walk(method_wd))[1]
        except:
            import pdb; pdb.set_trace()
        # switch from method_wd pov to abs pov
        rep_dirs = [os.path.join(method_wd, rep_dir)
                    for rep_dir in rep_dirs]

        rep_cpt = 0
        all_psm_bds.append([])
        ## Go over repetitions

        processed_results = pool.imap(process_rep, zip(rep_dirs,
                                                       itertools.repeat(ps_method),
                                                       itertools.repeat(ss_min),
                                                       itertools.repeat(ss_max),
                                                       itertools.repeat(dim_map),
                                                       itertools.repeat(bd_inds),
                                                       itertools.repeat(nb_div),
                                                       itertools.repeat(bd_cols)))

        ## Debug(not parallel)
        # processed_results = []
        # for rep_dir in rep_dirs:
        #     processed_results.append(process_rep([rep_dir, ps_method, ss_min,
        #                                          ss_max, dim_map, bd_inds,
        #                                                nb_div, bd_cols]))
        

        
        # for rep_dir in rep_dirs:
        #     ## get the last archive file name
        #     archive_fname = filter_archive_fnames(rep_dir)
        #     if archive_fname is None:
        #         continue
        #     ## Load it as a pandas dataframe
        #     archive_data = pd_read_csv_fast(archive_fname)
        #     # drop the last column which was made because there is a
        #     # comma after last value i a line
        #     archive_data = archive_data.iloc[:,:-1]
        #     ## Compute coverage and save it
        #     coverages[psm_cpt, rep_cpt] = compute_cov(archive_data, ss_min,
        #                                               ss_max, dim_map,
        #                                               bd_inds, nb_div)

        #     bds_per_gen_fpath = os.path.join(rep_dir, 'bds_per_gen.npz')
        #     bds_per_gen = np.load(bds_per_gen_fpath)
        #     total_evals = int(archive_fname.split('/')[-1].replace('.', '_').split('_')[1])
        #     gen_covs, gen_evals = get_cov_per_gen(bds_per_gen, ss_min,
        #                                           ss_max, dim_map,
        #                                           bd_inds, nb_div,
        #                                           total_evals)
        #     covs_per_gen[psm_cpt].append(gen_covs)
        #     evals_per_gen[psm_cpt].append(gen_evals)
        #     ## Get archive BDs array and save it
        #     archive_bds = archive_data[bd_cols].to_numpy()
        #     all_psm_bds[psm_cpt].append(archive_bds)
        #     rep_cpt += 1
    
        for (final_cov, gen_covs, gen_evals, archive_bds, final_qd_score,
             archive_size, gen_all_err, gen_add_err, gen_discard_err,
             gen_evals_err) in processed_results:

            coverages[psm_cpt, rep_cpt] = final_cov 
            qd_scores[psm_cpt, rep_cpt] = final_qd_score 
            archive_sizes[psm_cpt, rep_cpt] = archive_size 
            covs_per_gen[psm_cpt].append(gen_covs)
            evals_per_gen[psm_cpt].append(gen_evals)
            all_psm_bds[psm_cpt].append(archive_bds)
            all_errors_per_gen[psm_cpt].append(gen_all_err)
            add_errors_per_gen[psm_cpt].append(gen_add_err)
            discard_errors_per_gen[psm_cpt].append(gen_discard_err)
            evals_per_gen_err[psm_cpt].append(gen_evals_err)
    
            rep_cpt += 1
            
    pool.close()

    fig, ax = plt.subplots()

    psm_covs_medians = []
    psm_n_evals_medians = []
    psm_covs_1qs = []
    psm_covs_3qs = []

    mbns_psm_idx = None
    daqd_psm_idx = None
    ns_psm_idx = None
    daqd_final_cov = None
    ns_final_cov = None
    
    ## Reformat data to plot it properly per evals, only need to this for daqd
    for (ps_method, psm_idx) in zip(ps_methods, range(len(ps_methods))):
        if 'mbns' in ps_method:
            mbns_psm_idx = psm_idx
        if ps_method == 'daqd' :
            daqd_psm_idx = psm_idx
        if ps_method == 'ns':
            ns_psm_idx = psm_idx
        if ps_method == 'mbns_population_novelty':
            ps_method = 'mbns_average_nov_novelty'
                
        max_gens = max([len(i) for i in covs_per_gen[psm_idx]])
        for (cov_per_gen, eval_per_gen) in zip(covs_per_gen[psm_idx], evals_per_gen[psm_idx]):
            if len(cov_per_gen) < max_gens:
                for _ in range(max_gens-len(cov_per_gen)):
                    cov_per_gen.append(np.nan)
                    eval_per_gen.append(np.nan)

        ## Compute median (excluding nans) for considered psm 
        psm_covs_medians.append(np.nanmedian(covs_per_gen[psm_idx], axis=0))
        psm_n_evals_medians.append(np.nanmedian(evals_per_gen[psm_idx], axis=0))
        psm_covs_1qs.append(np.nanquantile(covs_per_gen[psm_idx], 1/4, axis=0))
        psm_covs_3qs.append(np.nanquantile(covs_per_gen[psm_idx], 3/4, axis=0))

        ## Correct the psm covs to be only increasing
        ## (fix needed because unstruct archive is can jump from cell to cell)
        for gen in range(1, len(psm_covs_medians[-1])):
            if psm_covs_medians[-1][gen-1] > psm_covs_medians[-1][gen]:
                psm_covs_medians[-1][gen] = psm_covs_medians[-1][gen-1]
            if psm_covs_1qs[-1][gen-1] > psm_covs_1qs[-1][gen]:
                psm_covs_1qs[-1][gen] = psm_covs_1qs[-1][gen-1]
            if psm_covs_3qs[-1][gen-1] > psm_covs_3qs[-1][gen]:
                psm_covs_3qs[-1][gen] = psm_covs_3qs[-1][gen-1]
        
        # if ps_method == 'daqd' or 'mbns' in ps_method:
        #     psm_n_evals_medians[-1] = np.array(psm_n_evals_medians[-1])
        #     # psm_covs_medians[-1] = np.array(psm_covs_medians[-1])
        #     # psm_covs_1qs[-1] = np.array(psm_covs_1qs[-1])
        #     # psm_covs_3qs[-1] = np.array(psm_covs_3qs[-1])
        #     psm_n_evals_medians[-1] = psm_n_evals_medians[-1][psm_n_evals_medians[-1]<49000]
        #     psm_covs_medians[-1] = psm_covs_medians[-1][:len(psm_n_evals_medians)]
        #     psm_covs_1qs[-1] = psm_covs_1qs[-1][:len(psm_n_evals_medians)]
        #     psm_covs_3qs[-1] = psm_covs_3qs[-1][:len(psm_n_evals_medians)]
        
        cut_idx = -1
        if args.environment == 'empty_maze':
            ## When we go up to 10k evals
            # cut_idx = np.where(psm_n_evals_medians[psm_idx] < 10000)[0][-1]
            ## When we go up to 5k evals
            cut_idx = np.where(psm_n_evals_medians[psm_idx] < 5000)[0][-1]
        elif args.environment == 'ball_in_cup':
            cut_idx = np.where(psm_n_evals_medians[psm_idx] < 25000)[0][-1]            
        else:
            cut_idx = np.where(psm_n_evals_medians[psm_idx] < 50000)[0][-1]
        if cut_idx != -1:
            ax.plot(psm_n_evals_medians[psm_idx][:cut_idx+1],
                    psm_covs_medians[psm_idx][:cut_idx+1],
                    label=ps_method)
            ax.fill_between(psm_n_evals_medians[psm_idx][:cut_idx+1],
                            psm_covs_1qs[psm_idx][:cut_idx+1],
                            psm_covs_3qs[psm_idx][:cut_idx+1],
                            alpha=0.2)
        else:
            ax.plot(psm_n_evals_medians[psm_idx],
                    psm_covs_medians[psm_idx],
                    label=ps_method)
            ax.fill_between(psm_n_evals_medians[psm_idx],
                            psm_covs_1qs[psm_idx],
                            psm_covs_3qs[psm_idx],
                            alpha=0.2)

        if ps_method == 'daqd':
            daqd_final_cov = psm_covs_medians[psm_idx][cut_idx]#[-1]
        elif ps_method == 'ns':
            ns_final_cov = psm_covs_medians[psm_idx][cut_idx]#[-1]
        ## Plot hline for some baselines
        # if ps_method == 'daqd' or ps_method == 'ns':
        #     ax.axhline(y=psm_covs_medians[psm_idx][cut_idx],#[-1],
        #                # xmin=int(psm_n_evals_medians[psm_idx][0]),
        #                xmin=0,
        #                xmax=int(psm_n_evals_medians[psm_idx][cut_idx]),#[-1]),
        #                # xmax=50000,
        #                linewidth=1, linestyle='--')#,
        #                # label=ps_method+'_final_cov')

    ## Get the idxs where crossing happens
    mbns_x_daqd = np.argwhere(np.diff(
        np.sign(psm_covs_medians[mbns_psm_idx]
                - np.array([daqd_final_cov]*
                           len(psm_covs_medians[mbns_psm_idx]))))).flatten()
    mbns_x_ns = np.argwhere(np.diff(
        np.sign(psm_covs_medians[mbns_psm_idx]
                - np.array([ns_final_cov]*
                           len(psm_covs_medians[mbns_psm_idx]))))).flatten()

    ## Plot vline for some baselines
    print('NS FINAL COV REACHED: ',
          psm_n_evals_medians[mbns_psm_idx][mbns_x_ns[0]],
          0,
          ns_final_cov,
          mbns_x_ns)
    # ax.axvline(x=psm_n_evals_medians[mbns_psm_idx][mbns_x_ns[0]],
    #            ymin=-0.02,
    #            ymax=ns_final_cov*1/ax.get_ylim()[1],
    #            linewidth=1, linestyle='--')

    print('DAQD FINAL COV REACHED: ',
          psm_n_evals_medians[mbns_psm_idx][mbns_x_daqd[0]],
          0,
          daqd_final_cov,
          mbns_x_daqd)
    # ax.axvline(x=psm_n_evals_medians[mbns_psm_idx][mbns_x_daqd[0]],
    #            ymin=-0.04,
    #            ymax=daqd_final_cov*1/ax.get_ylim()[1],
    #            linewidth=1, linestyle='--')

    ## Otherway to do it, better as it allow overlap
    from matplotlib.lines import Line2D
    x_ns,y_ns = np.array([[psm_n_evals_medians[mbns_psm_idx][mbns_x_ns[0]],
                           psm_n_evals_medians[mbns_psm_idx][mbns_x_ns[0]],
                           50000],
                          [-0.02, ns_final_cov, ns_final_cov]])

    x_daqd,y_daqd = np.array([[psm_n_evals_medians[mbns_psm_idx][mbns_x_daqd[0]],
                               psm_n_evals_medians[mbns_psm_idx][mbns_x_daqd[0]],
                               50000],
                              [-0.04, daqd_final_cov, daqd_final_cov]])
    
    
    extraticks = [psm_n_evals_medians[mbns_psm_idx][mbns_x_ns[0]],
                  psm_n_evals_medians[mbns_psm_idx][mbns_x_daqd[0]]]

    if args.environment == 'empty_maze':
        # plt.xticks(list(np.arange(0,11000,10000//5)) + extraticks)
        # ax.set_xlim(0, 10100)
        x_ns[2] = 5000; y_ns[0] = -0.025
        x_daqd[2] = 5000; y_daqd[0] = -0.05
        ticks_pos = list(np.arange(0,5000,5000//5)) + extraticks 
        labels = [str(tick) for tick in ticks_pos]
        labels[-2] = '\nFinal NS coverage reached'
        labels[-1] = '\n\nFinal DAQD coverage reached'
        plt.xticks(ticks_pos, labels)
        plt.xticks()
        ax.set_xlim(0, 5100)
        ax.set_ylim(0, 1.05)
    elif args.environment == 'ball_in_cup':
        x_ns[2] = 25000; y_ns[0] = -0.005
        x_daqd[2] = 25000; y_daqd[0] = -0.01
        ticks_pos = list(np.arange(0,26000,25000//5)) + extraticks
        labels = [str(tick) for tick in ticks_pos]
        labels[-2] = '\nFinal NS coverage reached'
        labels[-1] = '\n\nFinal DAQD coverage reached'
        plt.xticks(ticks_pos, labels)
        ax.set_xlim(0, 25100)
        ax.set_ylim(0, 0.2)
    else:
        ticks_pos = list(np.arange(0,51000,50000//5)) + extraticks 
        labels = [str(tick) for tick in ticks_pos]
        labels[-2] = '\nFinal NS coverage reached'
        labels[-1] = '\n\nFinal DAQD coverage reached'
        plt.xticks(ticks_pos, labels)
        ax.set_xlim(0, 50100)
        ax.set_ylim(0, 0.7)
        ## Change the rotation of the extraticks
        # ticks = ax.get_xticklabels()
        # ticks[-2].set_rotation(70) ## rotate the NS extratick label
        # ticks[-2].set_text('Final NS coverage reached') # does not work
        # ticks[-1].set_rotation(70) ## rotate the DAQD extratick label
        # ticks[-1].set_text('Final DAQD coverage reached') # does not work

    ## Set the NS lines for pretty fig
    line = Line2D(x_ns, y_ns, linestyle='--', lw=2.5, color='orange', alpha=0.4)
    line.set_clip_on(False)
    ax.add_line(line)

    ## Set the DAQD lines for pretty fig
    line = Line2D(x_daqd, y_daqd, linestyle='--', lw=2.5, color='r', alpha=0.4)
    line.set_clip_on(False)
    ax.add_line(line)

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Archive coverage")
    
    plt.title(f'Archive coverage evolution of several EAs on {args.environment}')
    fig.set_size_inches(35, 14)
    plt.legend()
    
    plt.savefig(f"{args.environment}_coverage_vs_evals",
                dpi=300, bbox_inches='tight')

    ## Handle plotting now
    # filter nan values
    all_psm_covs = []
    all_psm_qd_scores = []
    all_psm_archive_sizes = []
    for (ps_method, psm_cpt) in zip(ps_methods, range(len(ps_methods))):
        cov_vals = coverages[psm_cpt]
        if not np.isnan(cov_vals).all():
            filtered_cov_vals = cov_vals[~np.isnan(cov_vals)]
            all_psm_covs.append(filtered_cov_vals)
        qd_score_vals = qd_scores[psm_cpt]
        if not np.isnan(qd_score_vals).all():
            filtered_qd_score_vals = qd_score_vals[~np.isnan(qd_score_vals)]
            all_psm_qd_scores.append(filtered_qd_score_vals)
        archive_size_vals = archive_sizes[psm_cpt]
        if not np.isnan(archive_size_vals).all():
            filtered_archive_size_vals = archive_size_vals[~np.isnan(archive_size_vals)]
            all_psm_archive_sizes.append(filtered_archive_size_vals)

    # replace 
    ps_methods = list(map(lambda x: x.replace('mbns_population_novelty', 'mbns_average_nov_novelty'), ps_methods))

    ## Figure for coverage boxplot of all methods on environment
    fig, ax = plt.subplots()
    ## Add to the coverage boxplot the policy search method
    ax.boxplot(all_psm_covs, 0, '') # don't show the outliers
    # ax.boxplot(all_psm_covs)
    ax.set_xticklabels(ps_methods)

    ax.set_ylabel("Coverage")

    plt.title(f"Final coverage for each policy search method on {args.environment} environment")
    fig.set_size_inches(28, 14)
    plt.savefig(f"{args.environment}_bp_coverage")

    ## Figure for qd-score boxplot of all methods on environment
    fig, ax = plt.subplots()
    ## Add to the qd score boxplot the policy search method
    ax.boxplot(all_psm_qd_scores, 0, '') # don't show the outliers
    # ax.boxplot(all_psm_qd_scores)

    ax.set_xticklabels(ps_methods)

    ax.set_ylabel("QD-Score")

    plt.title(f"Final QD-Score for each policy search method on {args.environment} environment")
    fig.set_size_inches(28, 14)
    plt.savefig(f"{args.environment}_bp_qd_score")
    
    ## Figure for archive size boxplot of all methods on environment
    fig, ax = plt.subplots()
    ## Add to the qd score boxplot the policy search method
    ax.boxplot(all_psm_archive_sizes, 0, '') # don't show the outliers
    # ax.boxplot(all_psm_archive_sizes)
    ax.set_xticklabels(ps_methods)

    ax.set_ylabel("Archive Size")

    plt.title(f"Final archive size for each policy search method on {args.environment} environment")
    fig.set_size_inches(28, 14)
    plt.savefig(f"{args.environment}_bp_archive_size")


    ### containers for (all) transfers median descriptor estimation errors
    daqd_all_errors_medians = []
    daqd_all_errors_1q = []
    daqd_all_errors_3q = []
    daqd_all_errors_evals = []
    mbns_all_errors_medians = []
    mbns_all_errors_1q = []
    mbns_all_errors_evals = []
    
    for (ps_method, psm_cpt) in zip(ps_methods, range(len(ps_methods))):
        if ps_method == 'mbns_population_novelty':
            ps_method = 'mbns_average_nov_novelty'
            
        ## Plot the cumulated (over repetitions) archives BDs
        fig, ax = plt.subplots()

        len_psm_bds = sum([len(array_psm_bds) for array_psm_bds in all_psm_bds[psm_cpt]])
        psm_bds = np.empty((len_psm_bds, dim_map))
        cur_len = 0
        for array_psm_bds in all_psm_bds[psm_cpt]:
            if len(array_psm_bds) > 0:
                psm_bds[cur_len:cur_len+len(array_psm_bds)] = array_psm_bds
                cur_len += len(array_psm_bds)

        ax.scatter(x=psm_bds[:,0], y=psm_bds[:,1], s=30, alpha=0.3)

        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")

        ax.set_xlim(ss_min[bd_inds[0]], ss_max[bd_inds[0]])
        ax.set_ylim(ss_min[bd_inds[1]], ss_max[bd_inds[1]])
        plt.title(f"Cumulated coverage over {n_reps} reps for {ps_method} on {args.environment} environment")
        fig.set_size_inches(28, 28)
        plt.savefig(f"{args.environment}_{ps_method}_cum_coverage")


        if ps_method == 'daqd' or 'mbns' in ps_method:

            #### WARNING: ONLY FOR MB Methods ######
            ## Plot descriptor estimation error at each generation
            ## Plotting all inds error, added inds error and discarded inds error
            ## Plotting median 1st and 3rd quartile for each
            fig, ax = plt.subplots()

            # gens = [gen+1 for gen in range(len(all_errors_medians))]

            ## Reformat data to be easier to plot
            all_errors_medians = []
            all_errors_1q = []
            all_errors_3q = []
            add_errors_medians = []
            add_errors_1q = []
            add_errors_3q = []
            discard_errors_medians = []
            discard_errors_1q = []
            discard_errors_3q = []
            evals = []
            
            for rep_cpt in range(len(all_errors_per_gen[psm_cpt])):
                ## remove spikes and replace with nan (only one or two in a few runs...)
                if args.environment == 'hexapod_omni':
                    thresh = 0.25 ## filter the spikes in daqd and mbns
                elif args.environment == 'empty_maze':
                    thresh = 600 ## doesn't filter anything
                elif args.environment == 'ball_in_cup':
                    thresh = 4.5 ## filter the initial spike in mbns 

                for k in range(3):
                    all_errors_per_gen[psm_cpt][rep_cpt][k]\
                        [all_errors_per_gen[psm_cpt][rep_cpt][k] > thresh] = np.nan 
                    add_errors_per_gen[psm_cpt][rep_cpt][k]\
                        [add_errors_per_gen[psm_cpt][rep_cpt][k] > thresh] = np.nan
                    discard_errors_per_gen[psm_cpt][rep_cpt][k]\
                        [discard_errors_per_gen[psm_cpt][rep_cpt][k] > thresh] = np.nan

                ## Format data
                all_errors_medians.append(list(all_errors_per_gen[psm_cpt][rep_cpt][0]))
                all_errors_1q.append(list(all_errors_per_gen[psm_cpt][rep_cpt][1]))
                all_errors_3q.append(list(all_errors_per_gen[psm_cpt][rep_cpt][2]))
                add_errors_medians.append(list(add_errors_per_gen[psm_cpt][rep_cpt][0]))
                add_errors_1q.append(list(add_errors_per_gen[psm_cpt][rep_cpt][1]))
                add_errors_3q.append(list(add_errors_per_gen[psm_cpt][rep_cpt][2]))
                discard_errors_medians.append(list(discard_errors_per_gen[psm_cpt][rep_cpt][0]))
                discard_errors_1q.append(list(discard_errors_per_gen[psm_cpt][rep_cpt][1]))
                discard_errors_3q.append(list(discard_errors_per_gen[psm_cpt][rep_cpt][2]))
                evals.append(list(evals_per_gen_err[psm_cpt][rep_cpt]))
                
            ## Harmonize the length by adding nans to the shorter runs (in terms of gens)
            max_gens = max([len(i) for i in all_errors_medians])
            for (all_err_median, all_err_1q, all_err_3q,
                 add_err_median, add_err_1q, add_err_3q,
                 disc_err_median, disc_err_1q, disc_err_3q, eval_per_gen_err) \
                 in zip(all_errors_medians,
                        all_errors_1q,
                        all_errors_3q,
                        add_errors_medians,
                        add_errors_1q,
                        add_errors_3q,
                        discard_errors_medians,
                        discard_errors_1q,
                        discard_errors_3q,
                        evals):

                if len(all_err_median) < max_gens:
                    for _ in range(max_gens-len(all_err_median)):
                        all_err_median.append(np.nan)
                        all_err_1q.append(np.nan)
                        all_err_3q.append(np.nan)
                        add_err_median.append(np.nan)
                        add_err_1q.append(np.nan)
                        add_err_3q.append(np.nan)
                        disc_err_median.append(np.nan)
                        disc_err_1q.append(np.nan)
                        disc_err_3q.append(np.nan)
                        eval_per_gen_err.append(np.nan)
                        
            all_errors_medians = np.nanmean(all_errors_medians, axis=0)
            all_errors_1q = np.nanmean(all_errors_1q, axis=0)
            all_errors_3q = np.nanmean(all_errors_3q, axis=0)
            add_errors_medians = np.nanmean(add_errors_medians, axis=0)
            add_errors_1q = np.nanmean(add_errors_1q, axis=0)
            add_errors_3q = np.nanmean(add_errors_3q, axis=0)
            discard_errors_medians = np.nanmean(discard_errors_medians, axis=0)
            discard_errors_1q = np.nanmean(discard_errors_1q, axis=0)
            discard_errors_3q = np.nanmean(discard_errors_3q, axis=0)
            evals = np.nanmean(evals, axis=0)

            if 'mbns' in ps_method:
                mbns_all_errors_medians = all_errors_medians.copy() 
                mbns_all_errors_1q = all_errors_1q.copy()
                mbns_all_errors_3q = all_errors_3q.copy()
                mbns_all_errors_evals = evals.copy()
            elif 'daqd' in ps_method:
                daqd_all_errors_medians = all_errors_medians.copy() 
                daqd_all_errors_1q = all_errors_1q.copy()
                daqd_all_errors_3q = all_errors_3q.copy()
                daqd_all_errors_evals = evals.copy()
            
            ## plot all
            ax.plot(evals, all_errors_medians,
                    label='Median descriptor estimation error (all)',
                    color='yellow')
            ax.fill_between(evals,
                            all_errors_1q,
                            all_errors_3q,
                            alpha=0.2,
                            color='yellow')
            ## plot added
            ax.plot(evals, add_errors_medians,
                    label='Median descriptor estimation error (add)',
                    color='green')
            ax.fill_between(evals,
                            add_errors_1q,
                            add_errors_3q,
                            alpha=0.2,
                            color='green')
            ## plot discard
            ax.plot(evals, discard_errors_medians,
                    label='Median descriptor estimation error (discard)',
                    color='red')
            ax.fill_between(evals,
                            discard_errors_1q,
                            discard_errors_3q,
                            alpha=0.2,
                            color='red')
            
            if args.environment == 'empty_maze':
                plt.xticks(list(np.arange(0,5000,5000//5)))
                ax.set_xlim(0, 5100)
            elif args.environment == 'ball_in_cup':
                plt.xticks(list(np.arange(0,26000,25000//5)))
                ax.set_xlim(0, 25100)
            else:
                plt.xticks(list(np.arange(0,51000,50000//5)))
                ax.set_xlim(0, 50100)
            
            ax.set_title(f'Descriptor estimation error evolution of {ps_method} on {args.environment}')
            ax.set_xlabel('Evaluations')
            ax.set_ylabel('Distance between estimated and real descriptor (L2 norm)')
            fig.set_size_inches(35, 14)
            ax.legend()
            fig.savefig(f"{args.environment}_{ps_method}_desc_error_by_gen",
                        dpi=300, bbox_inches='tight')

    ## plot all descriptor estimation error medians on same plot
    fig, ax = plt.subplots()
    ax.plot(mbns_all_errors_evals, mbns_all_errors_medians,
            label='Median descriptor estimation error for MBNS',
            color='Blue')
    ax.fill_between(mbns_all_errors_evals,
                    mbns_all_errors_1q,
                    mbns_all_errors_3q,
                    alpha=0.2,
                    color='Blue')
    ax.plot(daqd_all_errors_evals, daqd_all_errors_medians,
            label='Median descriptor estimation error for DAQD',
            color='Red')
    ax.fill_between(daqd_all_errors_evals,
                    daqd_all_errors_1q,
                    daqd_all_errors_3q,
                    alpha=0.2,
                    color='Red')

    if args.environment == 'empty_maze':
        plt.xticks(list(np.arange(0,5000,5000//5)))
        ax.set_xlim(0, 5100)
    elif args.environment == 'ball_in_cup':
        plt.xticks(list(np.arange(0,26000,25000//5)))
        ax.set_xlim(0, 25100)
    else:
        plt.xticks(list(np.arange(0,51000,50000//5)))
        ax.set_xlim(0, 50100)

    ax.set_title(f'Descriptor estimation error evolution of MBNS and DAQD on {args.environment}')
    ax.set_xlabel('Evaluations')
    ax.set_ylabel('Distance between estimated and real descriptor (L2 norm)')
    fig.set_size_inches(35, 14)
    ax.legend()
    fig.savefig(f"{args.environment}_mbns_vs_daqd_desc_error_by_gen",
                dpi=300, bbox_inches='tight')
        

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning) ## for pandas frame.append
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ps-methods', nargs="*",
                        type=str, default=['ns', 'qd', 'daqd'])
    parser.add_argument('--nb-div', type=int, default=50)
    parser.add_argument('--n-reps', type=int, default=10)
    args = process_args(parser)

    main(args)


