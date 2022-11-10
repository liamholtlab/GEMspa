import numpy as np
import msd_diffusion as msd_diff
import pandas as pd
import argparse
import os

min_track_len_linfit = 11
min_track_len_loglogfit = 11
min_track_len_ensemble = 11
tlag_cutoff_linfit = 10
tlag_cutoff_loglogfit = 10
tlag_cutoff_linfit_ensemble = 10
tlag_cutoff_loglogfit_ensemble = tlag_cutoff_linfit_ensemble
min_track_len_step_size = 3
max_tlag_step_size = 10
time_step = 0.010
micron_per_px = 0.0917

def process_file(input_file, output_path):

    print("Running analyis")

    print(f"min track len: {min_track_len_linfit}")
    print(f"tlag cutoff: {tlag_cutoff_linfit}")
    print(f"min track len step size: {min_track_len_step_size}")
    print(f"max tlag step size: {max_tlag_step_size}")
    print(f"time step: {time_step}")
    print(f"micron per pixel: {micron_per_px}")
    print("")

    pre=os.path.splitext(os.path.split(input_file)[1])[0]

    # load track data file
    track_data_df = pd.read_csv(input_file)
    track_data_df = track_data_df[['Trajectory', 'Frame', 'x', 'y']]
    track_data = track_data_df.to_numpy()

    msd_diff_obj = msd_diff.msd_diffusion()

    msd_diff_obj.min_track_len_linfit = min_track_len_linfit
    msd_diff_obj.min_track_len_loglogfit = min_track_len_loglogfit
    msd_diff_obj.min_track_len_ensemble = min_track_len_ensemble
    msd_diff_obj.tlag_cutoff_linfit = tlag_cutoff_linfit
    msd_diff_obj.tlag_cutoff_loglogfit = tlag_cutoff_loglogfit
    msd_diff_obj.tlag_cutoff_linfit_ensemble = tlag_cutoff_linfit_ensemble
    msd_diff_obj.tlag_cutoff_loglogfit_ensemble = tlag_cutoff_loglogfit_ensemble
    msd_diff_obj.min_track_len_step_size = min_track_len_step_size
    msd_diff_obj.max_tlag_step_size = max_tlag_step_size
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

    #msd_diff_obj.radius_of_gyration()
    msd_diff_obj.average_velocity()

    D_median = np.median(msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col])
    a_median = np.median(msd_diff_obj.D_loglogfits[:,msd_diff_obj.D_loglog_alpha_col])

    D_linfits = pd.DataFrame(msd_diff_obj.D_linfits[:, :],
                             columns=['Trajectory', 'D', 'err', 'r_sq', 'rmse', 'track_len', 'D_max_tlag'])
    D_linfits = D_linfits[['Trajectory', 'D', 'err', 'r_sq', 'rmse', 'track_len']]
    D_loglogfits = pd.DataFrame(msd_diff_obj.D_loglogfits[:, :],
                             columns=['Trajectory', 'K', 'aexp', 'K_err', 'aexp_err', 'aexp_r_sq', 'aexp_rmse', 'track_len', 'D_max_tlag'])
    D_loglogfits = D_loglogfits[['Trajectory', 'K', 'aexp', 'aexp_r_sq', 'aexp_rmse', 'track_len']]

    D_linfits['avg_velocity'] = msd_diff_obj.avg_velocity[np.isin(msd_diff_obj.avg_velocity[:, 0], msd_diff_obj.D_linfits[:, 0])][:, 1]

    D_linfits.to_csv(f"{output_path}/{pre}_D_linfits.csv")
    D_loglogfits.to_csv(f"{output_path}/{pre}_D_loglogfits.csv")

    print(f"Median eff-D = {round(D_median,4)}")
    print(f"Median alpha = {round(a_median,4)}")
    print(f"Number of tracks = {len(msd_diff_obj.track_lengths)}")
    print(f"Number of tracks min length (eff-D) = {len(msd_diff_obj.D_linfits)}")

    # Angles and step sizes
    msd_diff_obj.step_sizes_and_angles()

    #msd_diff_obj.non_gaussian_1d()

    step_sizes = pd.DataFrame(msd_diff_obj.step_sizes)
    step_sizes=step_sizes.transpose()
    l = range(1, len(step_sizes.columns) + 1)
    step_sizes.columns=[str(x) + "_tlag" for x in l]

    angles = pd.DataFrame(msd_diff_obj.angles)
    angles = angles.transpose()
    l = range(1, len(angles.columns) + 1)
    angles.columns = [str(x) + "_tlag" for x in l]

    step_sizes.to_csv(f"{output_path}/{pre}_step_sizes.csv")
    angles.to_csv(f"{output_path}/{pre}_angles.csv")

    # Cosine theta
    arr=[]
    for tlag in angles.columns:
        cur_tlag_data = angles[tlag]
        cur_tlag_data = cur_tlag_data[np.logical_not(np.isnan(cur_tlag_data))]
        if (len(cur_tlag_data) > 1):
            # take the cosine of theta
            cos_theta = np.cos(np.deg2rad(cur_tlag_data))
            arr.append([tlag, np.mean(cos_theta), np.median(cos_theta), np.std(cos_theta),
             np.std(cos_theta) / np.sqrt(len(cos_theta)),len(cos_theta)])
        else:
            arr.append([tlag, np.nan, np.nan, np.nan, np.nan, np.nan])

    df=pd.DataFrame(arr, columns=['tlag','mean','median','stdev','sem','num_angles'])
    df.to_csv(f"{output_path}/{pre}_cos_theta.csv")

    # Ensemble average fit
    num_tracks_ens = msd_diff_obj.calculate_ensemble_average()
    msd_diff_obj.fit_msd_ensemble()
    msd_diff_obj.fit_msd_ensemble_alpha()

    arr=[]
    arr.append([msd_diff_obj.ensemble_fit_D,
                msd_diff_obj.ensemble_fit_rsq,
                msd_diff_obj.anomolous_fit_K,
                msd_diff_obj.anomolous_fit_alpha,
                msd_diff_obj.anomolous_fit_rsq,
                num_tracks_ens])
    df=pd.DataFrame(arr,columns=['ensemble_D',
                              'ensemble_r_sq',
                              'ensemble_loglog_K',
                              'ensemble_loglog_aexp',
                              'ensemble_loglog_r_sq',
                              'ensemble_num_tracks'])
    df.to_csv(f"{output_path}/{pre}_ensemble_fit.csv")

    # output the tau vs. MSD points for ensemble average
    arr=[]
    for tau in range(1, msd_diff_obj.tlag_cutoff_linfit_ensemble + 1):
        if (len(msd_diff_obj.ensemble_average) >= tau):
            arr.append([tau*msd_diff_obj.time_step,
                       msd_diff_obj.ensemble_average[tau - 1][1],
                       msd_diff_obj.ensemble_average[tau - 1][2]])
        else:
            arr.append([tau * msd_diff_obj.time_step,
                       np.nan,
                       np.nan])
    df = pd.DataFrame(arr, columns=['time','MSD_ave','MSD_std'])
    df.to_csv(f"{output_path}/{pre}_ensemble_msd.csv")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help="input csv trajectory file",
                        type=str)
    parser.add_argument("-o", "--output_path", help="full path to save output (.)",
                        type=str, default=".")

    parser.add_argument("-m", "--micron_per_px", help="pixel size in microns (0.0917)",
                        type=float, default=0.0917)
    parser.add_argument("-ts", "--time_step", help="time step in seconds (0.010)",
                        type=float, default=0.010)

    parser.add_argument("-tlag", "--tlag_cutoff", help="tlag cutoff for fits (10)",
                        type=int, default=10)
    parser.add_argument("-len", "--min_len", help="min track length for fits (11)",
                        type=int, default=11)

    parser.add_argument("-len_s", "--min_len_s", help="min track length for step size/angles (3)",
                        type=int, default=3)
    parser.add_argument("-tlag_s", "--max_tlag_s", help="max tlag for step size/angles (10)",
                        type=int, default=10)

    args = parser.parse_args()

    min_track_len_linfit = args.min_len
    min_track_len_loglogfit = args.min_len
    min_track_len_ensemble = args.min_len

    tlag_cutoff_linfit = args.tlag_cutoff
    tlag_cutoff_loglogfit = args.tlag_cutoff

    tlag_cutoff_linfit_ensemble = args.tlag_cutoff
    tlag_cutoff_loglogfit_ensemble = args.tlag_cutoff

    min_track_len_step_size = args.min_len_s
    max_tlag_step_size = args.max_tlag_s

    time_step = args.time_step
    micron_per_px = args.micron_per_px

    process_file(args.input_file,
                 args.output_path)

