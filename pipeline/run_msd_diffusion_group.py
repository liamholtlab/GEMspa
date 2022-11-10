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
    (file_name, time_step, micron_per_px) = params
    track_data_df = pd.read_csv(file_name)
    track_data_df = track_data_df[['Trajectory', 'Frame', 'x', 'y']]
    track_data = track_data_df.to_numpy()

    msd_diff_obj = msd_diff.msd_diffusion()

    msd_diff_obj.min_track_len_linfit = 11  # min. 11 track length gives at least 10 tlag values
    msd_diff_obj.min_track_len_loglogfit = 11
    msd_diff_obj.min_track_len_ensemble = 11
    msd_diff_obj.tlag_cutoff_linfit = 10
    msd_diff_obj.tlag_cutoff_loglogfit = 10
    msd_diff_obj.tlag_cutoff_linfit_ensemble = 10
    msd_diff_obj.tlag_cutoff_loglogfit_ensemble = 10
    msd_diff_obj.min_track_len_step_size = 3
    msd_diff_obj.max_tlag_step_size = 10
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

    return msd_diff_obj


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        input_path = str(sys.argv[1])
        output_path = str(sys.argv[2])
        group_start = int(sys.argv[3])
        group_end = int(sys.argv[4])
        time_step = float(sys.argv[5])
        micron_per_px = float(sys.argv[6])
    else:
        print("Did not receive valid input parameters, attempting with default (test) values")
        input_path="/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/GEMspa_Trial/Trajectories"
        output_path="/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/GEMspa_Trial/Trajectories_output"
        group_start=3
        group_end=6
        time_step = 0.010
        micron_per_px = 0.0917

    files_list = glob.glob(f"{input_path}/*.csv")
    files_list.sort()
    files_to_process = files_list[group_start:group_end]
    group_size = len(files_to_process)

    params_arr = list(zip(files_to_process,
                          [time_step]*group_size,
                          [micron_per_px]*group_size))

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
    data_list_with_results = pd.DataFrame(np.empty((len(files_to_process), len(colnames)), dtype=np.str),
                                          columns=colnames)


    with Pool(group_size) as p:
        msd_diff_obj_list = p.map(run_msd_diffusion, params_arr)
        for index,msd_diff_obj in enumerate(msd_diff_obj_list):
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


