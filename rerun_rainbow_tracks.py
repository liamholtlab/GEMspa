
import rainbow_tracks as rt
import pandas as pd
import os
import numpy as np

dir_input = "/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/GEMspa_Trial/results9"
dir_output = "/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/GEMspa_Trial/results9/test_rt"

def read_track_data_file(file_name):
    track_data_cols=['Trajectory', 'Frame', 'x', 'y']
    ext = (os.path.splitext(file_name)[1]).lower()
    if (ext == '.csv'):
        sep = ','
    elif (ext == '.txt'):
        sep = '\t'
    else:
        print(f"Error in reading {file_name}: all input files must have extension txt or csv. Skipping...")
        return pd.DataFrame()
    track_data_df = pd.read_csv(file_name, sep=sep)
    for col_name in track_data_cols:
        if (not (col_name in track_data_df.columns)):
            print(f"Error in reading {file_name}: required column {col_name} is missing.\n")
            return pd.DataFrame()

    track_data_df = track_data_df[track_data_cols]
    return track_data_df

def fill_track_sizes(tracks,micron_per_px):
    # add column to tracks array containing the step size for each step of each track (distance between points)
    # step size in *microns*
    tracks = np.append(tracks, np.zeros((len(tracks),1)), axis=1)
    ids = np.unique(tracks[:, 0]) # 0 is the track id column
    ss_i=0
    for i,id in enumerate(ids):
        cur_track = tracks[np.where(tracks[:, 0] == id)]
        ss_i+=1
        for j in range(1,len(cur_track),1):
            d = np.sqrt((cur_track[j, 2] - cur_track[j-1, 2]) ** 2 + (cur_track[j, 3] - cur_track[j-1, 3]) ** 2)
            tracks[ss_i,4] = d * micron_per_px
            ss_i+=1
    return tracks

def make_rainbow_tracks(input_dir,
                        tif_prefix,
                        output_dir,
                        micron_per_px):
    # input_dir is the directory where the GEMSpa output files are located
    # output_dir is the directory to SAVE the tif files (rainbow tracks drawn on images)
    # image locations are read from the GEMSpa output files
    # if input_dir == '', then the rainbow tracks files will be saved to input_dir

    if(output_dir == ''):
        output_dir = input_dir

    rainbow_tr = rt.rainbow_tracks()
    rainbow_tr.tracks_id_col = 0
    # 'frame' is column 1
    rainbow_tr.tracks_x_col = 2
    rainbow_tr.tracks_y_col = 3
    rainbow_tr.tracks_color_val_col = 4

    # read the all_data.txt file
    all_data_df = pd.read_csv(os.path.join(input_dir, "all_data.txt"),sep='\t')

    print("")
    # for each tif image, pull the track data and diffusion coefficient
    all_data_df['full_file_name']=all_data_df['directory']+'/'+all_data_df['file name']

    for traj_file in all_data_df['full_file_name'].unique():

        cur_all_data_df = all_data_df[all_data_df['full_file_name']==traj_file]
        img_dir = cur_all_data_df['image file dir'].iloc[0]
        img_file = os.path.splitext(os.path.split(traj_file)[1])[0]
        img_file = img_dir + '/' + tif_prefix + img_file[5:]

        if (not img_file.endswith(".tif")):
            img_file = img_file + ".tif"
        if (os.path.isfile(img_file)):
            # need track data - load Trajectory file
            tracks_df = read_track_data_file(traj_file)
            if(len(tracks_df)>0):
                out_file = os.path.split(img_file)[1][:-4] + '_tracks_Deff.tif'
                rainbow_tr.plot_diffusion(img_file,
                                          np.asarray(tracks_df),
                                          np.asarray(cur_all_data_df[['Trajectory','D']]),
                                          output_dir+'/'+out_file)
                out_file = os.path.split(img_file)[1][:-4] + '_tracks_ss.tif'
                tracks_arr=fill_track_sizes(np.asarray(tracks_df), micron_per_px)
                rainbow_tr.plot_step_sizes(img_file,
                                           tracks_arr,
                                           output_dir + '/' + out_file)


        else:
            print(f"Error! Image file not found: {img_file} for rainbow tracks/ROIs.\n")

        break



make_rainbow_tracks(dir_input, "BG_", dir_output, 0.11)

