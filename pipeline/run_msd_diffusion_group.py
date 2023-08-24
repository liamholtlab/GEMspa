import numpy as np
import msd_diffusion as msd_diff
import sys
import glob
import pandas as pd
from multiprocessing import Pool
import os

# Input is a directory where Trajectory files (Mosaic output) are stored, the group number and group size
# This script will run msd and diffusion calculations for the files in the current group
# All files in path are read in and sorted first, then the designated group of files is run

max_D_cutoff=2
min_D_cutoff=0

def run_msd_diffusion(params):
    (file_name, time_step, micron_per_px,
     tlag_cutoff,min_len,min_len_s,max_tlag_s) = params

    track_data_df = pd.read_csv(file_name)
    track_data_df = track_data_df[['Trajectory', 'Frame', 'x', 'y']]
    track_data = track_data_df.to_numpy()

    msd_diff_obj = msd_diff.msd_diffusion()

    msd_diff_obj.min_track_len_linfit = min_len
    msd_diff_obj.min_track_len_loglogfit = min_len
    msd_diff_obj.min_track_len_ensemble = min_len
    msd_diff_obj.tlag_cutoff_linfit = tlag_cutoff
    msd_diff_obj.tlag_cutoff_loglogfit = tlag_cutoff
    msd_diff_obj.tlag_cutoff_linfit_ensemble = tlag_cutoff
    msd_diff_obj.tlag_cutoff_loglogfit_ensemble = tlag_cutoff
    msd_diff_obj.min_track_len_step_size = min_len_s
    msd_diff_obj.max_tlag_step_size = max_tlag_s
    msd_diff_obj.time_step = time_step
    msd_diff_obj.micron_per_px = micron_per_px

    msd_diff_obj.set_track_data(track_data)

    # Execution
    msd_diff_obj.msd_all_tracks()
    msd_diff_obj.fit_msd()
    msd_diff_obj.fit_msd_alpha()
    msd_diff_obj.calculate_ensemble_average()
    msd_diff_obj.fit_msd_ensemble()
    msd_diff_obj.fit_msd_ensemble_alpha()
    msd_diff_obj.average_velocity()
    msd_diff_obj.step_sizes_and_angles()

    return msd_diff_obj


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        input_path = str(sys.argv[1])
        output_path = str(sys.argv[2])
        group_start = int(sys.argv[3])
        group_end = int(sys.argv[4])
        time_step = float(sys.argv[5])
        micron_per_px = float(sys.argv[6])
        tlag_cutoff = int(sys.argv[7])
        min_len = int(sys.argv[8])
        min_len_s = int(sys.argv[9])
        max_tlag_s = int(sys.argv[10])
    else:
        print("Did not receive valid input parameters, attempting with default (test) values")
        input_path="/Users/snk218/Dropbox (NYU Langone Health)/mac_files/holtlab/data_and_results/GEMspa_Trial/Trajectories"
        output_path="/Users/snk218/Dropbox (NYU Langone Health)/mac_files/holtlab/data_and_results/GEMspa_Trial/Trajectories_output"
        group_start=0
        group_end=3
        time_step = 0.010
        micron_per_px = 0.0917
        tlag_cutoff = 10
        min_len = 11
        min_len_s = 3
        max_tlag_s = 10

    files_list = glob.glob(f"{input_path}/*.csv")
    files_list.sort()
    files_to_process = files_list[group_start:group_end]
    group_size = len(files_to_process)

    params_arr = list(zip(files_to_process,
                          [time_step]*group_size,
                          [micron_per_px]*group_size,
                          [tlag_cutoff]*group_size,
                          [min_len]*group_size,
                          [min_len_s]*group_size,
                          [max_tlag_s]*group_size))

    # Summary results
    colnames = ['directory',
                'file name',
                'D_median',
                'D_mean',
                'num_tracks',
                'num_tracks_D',
                'group',
                'ensemble_D',
                'ensemble_r_sq',
                'ensemble_loglog_K',
                'ensemble_loglog_aexp',
                'ensemble_loglog_r_sq']
    data_list_with_results = pd.DataFrame(np.empty((len(files_to_process), len(colnames)), dtype='str'),
                                          columns=colnames)

    # Full results
    colnames = ['directory',
                'file name',
                'Trajectory',
                'avg_velocity',
                'D',
                'err',
                'r_sq',
                'rmse',
                'track_len',
                'D_max_tlag',
                'K',
                'aexp',
                'aexp_r_sq',
                'aexp_rmse']

    D_allfits_full = pd.DataFrame()
    stepsizes_full = pd.DataFrame()
    angles_full = pd.DataFrame()
    with Pool(group_size) as p:
        msd_diff_obj_list = p.map(run_msd_diffusion, params_arr)
        for index,msd_diff_obj in enumerate(msd_diff_obj_list):

            # Fill all data array
            D_linfits = pd.DataFrame(msd_diff_obj.D_linfits[:, :],
                                     columns=['Trajectory', 'D', 'err', 'r_sq', 'rmse', 'track_len', 'D_max_tlag'])
            D_linfits['avg_velocity'] = msd_diff_obj.avg_velocity[np.isin(msd_diff_obj.avg_velocity[:, 0],
                                                                          msd_diff_obj.D_linfits[:, 0])][:, 1]
            D_linfits['directory']=input_path
            D_linfits['file name']=os.path.split(files_to_process[index])[1]
            D_linfits['group'] = f"{group_start}-{group_end}"
            D_linfits = D_linfits[['directory','file name','group','Trajectory', 'D', 'err', 'r_sq', 'rmse', 'track_len','avg_velocity']]

            D_loglogfits = pd.DataFrame(msd_diff_obj.D_loglogfits[:, :],
                                        columns=['Trajectory', 'K', 'aexp', 'K_err', 'aexp_err', 'aexp_r_sq',
                                                 'aexp_rmse', 'track_len', 'D_max_tlag'])
            D_loglogfits = D_loglogfits[['K', 'aexp', 'aexp_r_sq', 'aexp_rmse']]

            D_allfits=pd.concat([D_linfits,D_loglogfits], axis=1) # works bc same track length cutoff for linear and log fits
            D_allfits_full=pd.concat([D_allfits_full,D_allfits], axis=0, ignore_index=True)

            # Fill summary data array
            D_median = np.median(msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col])
            D_mean = np.mean(msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col])

            D_linfits_filtered = msd_diff_obj.D_linfits[np.where(
                (msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col] <= max_D_cutoff) &
                (msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col] >= min_D_cutoff))]

            if (len(D_linfits_filtered) == 0):
                D_median_filt = np.nan
                D_mean_filt = np.nan
            else:
                D_median_filt = np.median(D_linfits_filtered[:, msd_diff_obj.D_lin_D_col])
                D_mean_filt = np.mean(D_linfits_filtered[:, msd_diff_obj.D_lin_D_col])

            # Fill step sizes
            num_tlags=len(msd_diff_obj.step_sizes)
            rest_cols=np.asarray(range(len(msd_diff_obj.step_sizes[0]))).astype('str')
            stepsizes = pd.DataFrame(msd_diff_obj.step_sizes, columns=rest_cols)
            stepsizes['tlag'] = list(range(1, num_tlags + 1))
            stepsizes['directory'] = input_path
            stepsizes['file name'] = os.path.split(files_to_process[index])[1]
            stepsizes['group'] = f"{group_start}-{group_end}"
            colnames=["directory", "file name", "group", "tlag"]
            colnames.extend(rest_cols)
            stepsizes=stepsizes[colnames]
            stepsizes_full = pd.concat([stepsizes_full, stepsizes], axis=0, ignore_index=True)

            # Fill angles
            num_tlags = len(msd_diff_obj.angles)
            rest_cols = np.asarray(range(len(msd_diff_obj.angles[0]))).astype('str')
            angles = pd.DataFrame(msd_diff_obj.angles, columns=rest_cols)
            angles['tlag'] = list(range(1, num_tlags + 1))
            angles['directory'] = input_path
            angles['file name'] = os.path.split(files_to_process[index])[1]
            angles['group'] = f"{group_start}-{group_end}"
            colnames = ["directory", "file name", "group", "tlag"]
            colnames.extend(rest_cols)
            angles = angles[colnames]
            angles_full = pd.concat([angles_full, angles], axis=0, ignore_index=True)

            # fill summary data array
            data_list_with_results.at[index, 'directory'] = input_path
            data_list_with_results.at[index, 'file name'] = os.path.split(files_to_process[index])[1]
            data_list_with_results.at[index, 'D_median'] = D_median
            data_list_with_results.at[index, 'D_mean'] = D_mean
            data_list_with_results.at[index, 'D_median_filtered'] = D_median_filt
            data_list_with_results.at[index, 'D_mean_filtered'] = D_mean_filt
            data_list_with_results.at[index, 'group'] = f"{group_start}-{group_end}"
            data_list_with_results.at[index, 'num_tracks'] = len(msd_diff_obj.track_lengths)
            data_list_with_results.at[index, 'num_tracks_D'] = len(msd_diff_obj.D_linfits)
            data_list_with_results.at[index, 'ensemble_D'] = msd_diff_obj.ensemble_fit_D
            data_list_with_results.at[index, 'ensemble_r_sq'] = msd_diff_obj.ensemble_fit_rsq
            data_list_with_results.at[index, 'ensemble_loglog_K'] = msd_diff_obj.anomolous_fit_K
            data_list_with_results.at[index, 'ensemble_loglog_aexp'] = msd_diff_obj.anomolous_fit_alpha
            data_list_with_results.at[index, 'ensemble_loglog_r_sq'] = msd_diff_obj.anomolous_fit_rsq

        data_list_with_results.to_csv(f"{output_path}/summary-{group_start}-{group_end}.txt", sep='\t')
        D_allfits_full.to_csv(f"{output_path}/all_data-{group_start}-{group_end}.txt", sep='\t')
        stepsizes_full.to_csv(f"{output_path}/step_sizes-{group_start}-{group_end}.txt", sep='\t')
        angles_full.to_csv(f"{output_path}/angles-{group_start}-{group_end}.txt", sep='\t')


