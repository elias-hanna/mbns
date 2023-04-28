## Local imports
from exps_utils import get_env_params, process_args, plot_cov_and_trajs, \
    save_archive_cov_by_gen

#----------Utils imports--------#
import os, sys
import argparse
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

    if ps_method == 'ns':
        # remove archive_eval.dat files, keep only archive_evals_all_evals.dat
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

def get_cov_per_gen(bds_per_gen, ss_min, ss_max, dim_map, bd_inds, nb_div, total_evals):
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
    ## Compute coverage and save it
    final_cov = compute_cov(archive_data, ss_min,
                            ss_max, dim_map,
                            bd_inds, nb_div)

    ## Get coverage and evals per gen
    if ps_method == 'ns':
        bds_per_gen_fpath = os.path.join(rep_dir, 'bds_per_gen_all_evals.npz')
    else:
        bds_per_gen_fpath = os.path.join(rep_dir, 'bds_per_gen.npz')
    gen_covs = []
    gen_evals = []
    try:
        bds_per_gen = np.load(bds_per_gen_fpath)
    
        total_evals = int(archive_fname.split('/')[-1].replace('.', '_').split('_')[1])
        gen_covs, gen_evals = get_cov_per_gen(bds_per_gen, ss_min,
                                              ss_max, dim_map,
                                              bd_inds, nb_div,
                                              total_evals)
    except Exception as e:
        print(f'Received error {e} when processing file: {bds_per_gen_fpath}')
    ## Get archive BDs array and save it
    archive_bds = archive_data[bd_cols].to_numpy()

    return final_cov, gen_covs, gen_evals, archive_bds

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
    if env_name == 'hexapod_omni':
        # ss_min = (ss_min+1.5)/3  
        # ss_max = (ss_max+1.5)/3
        ss_min[:] = 0
        ss_max[:] = 1
    init_obs = env_params['init_obs'] 
    state_dim = env_params['state_dim']
    obs_min = env_params['obs_min']
    obs_max = env_params['obs_max']
    dim_map = env_params['dim_map']
    bd_inds = env_params['bd_inds']
    bins = env_params['bins'] ## for grid based qd

    ## get bd cols
    bd_cols = [f'bd{i}' for i in range(dim_map)]
    ## Create main data holders
    all_psm_bds = []
    coverages = np.empty((n_ps_methods, n_reps))
    coverages[:] = np.nan

    # Get current working dir (folder from which py script was executed)
    root_wd = os.getcwd()

    covs_per_gen = []
    evals_per_gen = []

    pool = Pool(10) ## use 10 cores (10 reps...)
    
    ## Go over methods (daqd, qd, qd_grid, ns)
    for (ps_method, psm_cpt) in zip(ps_methods, range(len(ps_methods))):
        covs_per_gen.append([])
        evals_per_gen.append([])
        
        print(f"Processing {ps_method} results on {args.environment}...")
        ## Go inside the policy search method folder
        # new working dir
        if ps_method == 'daqd' or 'mbns' in ps_method:
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
    
        for (final_cov, gen_covs, gen_evals, archive_bds) in processed_results:
            coverages[psm_cpt, rep_cpt] = final_cov 
            covs_per_gen[psm_cpt].append(gen_covs)
            evals_per_gen[psm_cpt].append(gen_evals)
            all_psm_bds[psm_cpt].append(archive_bds)
            rep_cpt += 1
            
    pool.close()

    fig, ax = plt.subplots()

    psm_covs_medians = []
    psm_n_evals_medians = []
    psm_covs_1qs = []
    psm_covs_3qs = []
    
    ## Reformat data to plot it properly per evals, only need to this for daqd
    for (ps_method, psm_idx) in zip(ps_methods, range(len(ps_methods))):
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

        ax.plot(psm_n_evals_medians[psm_idx], psm_covs_medians[psm_idx],
                label=ps_method)
        ax.fill_between(psm_n_evals_medians[psm_idx],
                        psm_covs_1qs[psm_idx],
                        psm_covs_3qs[psm_idx],
                        alpha=0.2)

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Archive coverage")
    
    plt.title(f'Archive coverage evolution of several EAs on {args.environment}')
    fig.set_size_inches(35, 14)
    plt.legend()
    
    plt.savefig(f"{args.environment}_coverage_vs_evals",
                dpi=300, bbox_inches='tight')

    ## Handle plotting now
    ## Figure for boxplot of all methods on environment
    fig, ax = plt.subplots()
    # filter nan values
    all_psm_covs = []
    for (ps_method, psm_cpt) in zip(ps_methods, range(len(ps_methods))):
        cov_vals = coverages[psm_cpt]
        if not np.isnan(cov_vals).all():
            filtered_cov_vals = cov_vals[~np.isnan(cov_vals)]
            all_psm_covs.append(filtered_cov_vals)
        
    ## Add to the coverage boxplot the policy search method
    ax.boxplot(all_psm_covs)
    ax.set_xticklabels(ps_methods)

    ax.set_ylabel("Coverage")

    plt.title(f"Coverage for each policy search method on {args.environment} environment")
    fig.set_size_inches(28, 14)
    plt.legend()
    plt.savefig(f"{args.environment}_bp_coverage")
    
    for (ps_method, psm_cpt) in zip(ps_methods, range(len(ps_methods))):
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
        plt.title(f"Cumulated coverage (over {n_reps} reps for {ps_method} on {args.environment} environment")
        fig.set_size_inches(28, 28)
        plt.savefig(f"{args.environment}_{ps_method}_cum_coverage")        
        
        

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


