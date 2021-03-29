#
# Read in the list of files
#
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import msd_diffusion as msd_diff
from scipy import stats
import datetime
import re
from tifffile import TiffFile
from nd2reader import ND2Reader
from skimage import io, draw, img_as_bool,measure
from read_roi import read_roi_zip
from read_roi import read_roi_file
from scipy import ndimage
from matplotlib import cm

def get_roi_name_list_from_mask(mask):
    mask=mask > 0
    labeled,n=ndimage.label(mask)
    names = list(np.unique(labeled))
    names=sorted(names)
    names.remove(0)
    return (names,labeled)

def get_roi_name_list(rois):
    name_list=[]
    for key in rois.keys():
        roi = rois[key]
        if ((roi['type'] == 'polygon') or
                (roi['type'] == 'freehand' and 'x' in roi and 'y' in roi) or
                (roi['type'] == 'rectangle') or
                (roi['type'] == 'oval')):
            name_list.append(key)
    return name_list

def limit_tracks_given_mask(mask_image, track_data):
    # loop through tracks, only use tracks that are fully within the ROI areas (all posiitons evaluate to 1)
    track_labels_full = np.zeros((len(track_data), 2))
    valid_id_list = []
    # 0: id, 1: label, 2: 0/1 inside an ROI
    id = track_data[0][0]
    prev_pos = 0
    for pos_i, pos in enumerate(track_data):
        if (pos[0] != id):
            if ((len(np.unique(track_labels_full[prev_pos:pos_i, 1])) == 1) and (track_labels_full[pos_i - 1][1] != 0)):
                valid_id_list.append(id)
            id = pos[0]
            prev_pos = pos_i

        label = mask_image[int(pos[3])][int(pos[2])]
        track_labels_full[pos_i][0] = pos[0]
        track_labels_full[pos_i][1] = label

    # check final track
    pos_i = pos_i + 1
    if ((len(np.unique(track_labels_full[prev_pos:pos_i, 1])) == 1) and (track_labels_full[pos_i - 1][1] != 0)):
        valid_id_list.append(id)

    valid_id_list = np.asarray(valid_id_list)
    return valid_id_list

def make_mask_from_roi(rois, roi_name, img_shape):
    # loop through ROIs, only set interior of selected ROI to 1
    final_img = np.zeros(img_shape, dtype='uint8')
    poly_error=False
    for key in rois.keys():
        if(key == roi_name):
            roi = rois[key]
            if (roi['type'] == 'polygon' or (roi['type'] == 'freehand' and 'x' in roi and 'y' in roi)):
                col_coords = roi['x']
                row_coords = roi['y']
                rr, cc = draw.polygon(row_coords, col_coords)
                final_img[rr, cc] = 1
            elif (roi['type'] == 'rectangle'):
                rr, cc = draw.rectangle((roi['top'], roi['left']), extent=(roi['height'], roi['width']))
                final_img[rr, cc] = 1
            elif (roi['type'] == 'oval'):
                rr, cc = draw.ellipse(roi['top'] + roi['height'] / 2, roi['left'] + roi['width'] / 2, roi['height'] / 2, roi['width'] / 2)
                final_img[rr, cc] = 1
            else:
                poly_error=True
            break
    return (final_img, poly_error)

def read_movie_metadata_tif(file_name):
    with TiffFile(file_name) as tif:
        metadata = tif.pages[0].tags['IJMetadata'].value
        metadata_list = metadata["Info"].split("\n")
        time_step_list=[]
        exposure=''
        cal=''
        steps=[]
        for item in metadata_list:
            if(item.startswith("dCalibration = ")):
                cal=float(item[15:])
            if(item.startswith("Exposure = ")):
                exposure = int(round(float(item[11:]), 0))
            #print(item)
            if(item.startswith("timestamp #")):
                mo=re.match(r'timestamp #(\d+) = (.+)', item)
                if(mo):
                    time_step_list.append(float(mo.group(2)))
        time_step_list=np.asarray(time_step_list)
        steps = time_step_list[1:] - time_step_list[:-1]
        steps = np.round(steps, 3)
        # convert from ms to s
        exposure = exposure / 1000
        return [cal, exposure, steps]

def read_movie_metadata_nd2(file_name):
    with ND2Reader(file_name) as images:
        steps = images.timesteps[1:] - images.timesteps[:-1]
        steps = np.round(steps, 0)
        #convert steps from ms to s
        steps=steps/1000
        microns_per_pixel=images.metadata['pixel_microns']
        return (microns_per_pixel,np.min(steps),steps)

def filter_tracks(track_data, min_len, time_steps, min_resolution):
    correct_ts = np.min(time_steps)
    traj_ids = np.unique(track_data['Trajectory'])
    track_data_cp=track_data.copy()
    track_data_cp['Remove']=True
    for traj_id in traj_ids:
        cur_track = track_data[track_data['Trajectory']==traj_id].copy()
        if(len(cur_track) >= min_len):
            new_seg=True
            longest_seg_start = cur_track.index[0]
            longest_seg_len = 0
            for row_i, row in enumerate(cur_track.iterrows()):
                if(row_i > 0):
                    if(new_seg):
                        cur_seg_start=row[0]-1
                        cur_seg_len=0
                        new_seg=False

                    fr=row[1]['Frame']
                    cur_ts = time_steps[int(fr-1)]
                    if((cur_ts - correct_ts) <= min_resolution):
                        #keep going
                        cur_seg_len+=1
                    else:
                        if(cur_seg_len > longest_seg_len):
                            longest_seg_len=cur_seg_len
                            longest_seg_start = cur_seg_start
                            new_seg=True
            if(not new_seg):
                if (cur_seg_len > longest_seg_len):
                    longest_seg_len = cur_seg_len
                    longest_seg_start = cur_seg_start

            if(longest_seg_len >= min_len):
                # add track segment to new DF - check # 10
                track_data_cp.loc[longest_seg_start:longest_seg_start+longest_seg_len,'Remove']=False

    track_data_cp = track_data_cp[track_data_cp['Remove']==False]
    track_data_cp.index=range(len(track_data_cp))
    track_data_cp = track_data_cp.drop('Remove', axis=1)
    return track_data_cp

class trajectory_analysis:
    def __init__(self, data_file, results_dir='.', use_movie_metadata=False, uneven_time_steps=False,
                 make_rainbow_tracks=True, limit_to_ROIs=False, img_file_prefix='DNA_', log_file=''):

        self.get_calibration_from_metadata=use_movie_metadata
        self.uneven_time_steps = uneven_time_steps
        self.make_rainbow_tracks = make_rainbow_tracks
        self.limit_to_ROIs = limit_to_ROIs

        self.calibration_from_metadata={}
        self.valid_img_files = {}
        self.valid_roi_files = {}

        self.min_ss_rainbow_tracks=0
        self.max_ss_rainbow_tracks=1
        self.min_D_rainbow_tracks=0
        self.max_D_rainbow_tracks=2

        self.img_file_prefix = img_file_prefix

        self.time_step = 0.010  # time between frames, in seconds
        self.micron_per_px = 0.11
        self.ts_resolution=0.005

        # track (trajectory) files columns - user can adjust as needed
        self.traj_id_col = 'Trajectory'
        self.traj_frame_col = 'Frame'
        self.traj_x_col = 'x'
        self.traj_y_col = 'y'

        self.min_track_len_linfit = 11
        self.min_track_len_step_size = 3
        self.track_len_cutoff_linfit = 11
        self.max_tlag_step_size=3

        self.use_D_cutoffs=True
        self.min_D_cutoff=0
        self.max_D_cutoff=2

        self.cell_column = False # will be set to True if cell column found below
        self.cell_column_name = "cell"

        self.results_dir = results_dir
        self.data_file = data_file
        if(log_file == ''):
            dir_, log_file_name = os.path.split(data_file)
            log_file_name = os.path.splitext(log_file_name)[0]
            log_file_name += '_' + datetime.datetime.now().strftime("%c").replace("/","-").replace("\\","-").replace(":","-").replace(" ","_") + '_log.txt'
            self.log_file = results_dir + '/' + log_file_name
        else:
            self.log_file = results_dir + '/' + log_file
        self.log = open(self.log_file, "w")

        self._index_col_name='id'
        self._dir_col_name='directory'
        self._file_col_name='file name'
        self._movie_file_col_name='movie file dir'
        self._img_file_col_name='image file dir'
        self._roi_col_name='roi'

        #read in the data file, check for the required columns and identify the label columns
        self.error_in_data_file = False
        self.data_list = pd.read_csv(data_file, sep='\t',dtype=str)

        # add ROI column
        self.data_list[self._roi_col_name] = ''
        self.known_columns = [self._index_col_name, self._dir_col_name, self._file_col_name,
                              self._movie_file_col_name, self._img_file_col_name, self._roi_col_name]

        col_names = list(self.data_list.columns)
        col_names=[x.lower() for x in col_names]
        self.data_list.columns = col_names
        for col in self.known_columns:
            if(col in col_names):
                col_names.remove(col)
            else:
                if( (col != self._movie_file_col_name and col != self._img_file_col_name) or
                (col == self._movie_file_col_name and self.get_calibration_from_metadata) or
                (col == self._img_file_col_name and (self.make_rainbow_tracks or self.limit_to_ROIs)) ):
                    self.log.write(f"Error! Required column {col} not found in input file {self.data_file}\n")
                    self.error_in_data_file=True
                    break

        # read in the time step information from the movie files
        if(not self.error_in_data_file):

            if(self.get_calibration_from_metadata):
                self.data_list[self._movie_file_col_name] = self.data_list[self._movie_file_col_name].fillna('')

            if(self.make_rainbow_tracks or self.limit_to_ROIs):
                self.data_list[self._img_file_col_name] = self.data_list[self._img_file_col_name].fillna('')

            if(self.limit_to_ROIs):
                # first, if limit_to_ROIs, then data_list needs to be extended - one row per ROI
                # get name from csv file name
                # csv file:           Traj_<file_name>.csv
                # roi file:           <file_name>.roi or <file_name>.zip
                # mask file:          <file_name>_mask.tif (values 0 and 255)
                new_data_list=[]
                ind=1
                for row in self.data_list.iterrows():
                    img_dir = row[1][self._img_file_col_name]
                    csv_file = row[1][self._file_col_name]
                    roi_file1 = img_dir + "/" + self.img_file_prefix + csv_file[5:-4]
                    roi_file2 = img_dir + "/" + csv_file[5:-4]

                    valid_roi_file=''
                    for roi_file in [roi_file1, roi_file2]:
                        if (roi_file.endswith(".tif")):
                            roi_file = roi_file[:-4]
                        if (os.path.isfile(roi_file + ".roi")):
                            valid_roi_file = roi_file + ".roi"
                            break
                        elif (os.path.isfile(roi_file + ".zip")):
                            valid_roi_file = roi_file + ".zip"
                            break
                        elif (os.path.isfile(roi_file + "_mask.tif")):
                            valid_roi_file = roi_file + "_mask.tif"
                            break

                    if (valid_roi_file != ''):
                        # read ROI file, get numbered list of ROIs, add rows and fill in
                        if(valid_roi_file.endswith('_mask.tif')):
                            img_mask = io.imread(valid_roi_file)
                            roi_name_list,labeled_img = get_roi_name_list_from_mask(img_mask)
                            to_save=os.path.split(valid_roi_file)[1][:-4]+'_LABELS.tif'
                            io.imsave(self.results_dir + '/' + to_save, labeled_img)
                        else:
                            if(valid_roi_file.endswith('.zip')):
                                rois = read_roi_zip(valid_roi_file)
                            else:
                                rois = read_roi_file(valid_roi_file)
                            roi_name_list = get_roi_name_list(rois)

                        for roi_name in roi_name_list:
                            row[1][self._roi_col_name]=roi_name
                            new_row = list(row[1])
                            new_data_list.append(new_row)
                            self.valid_roi_files[ind]=valid_roi_file
                            ind+=1
                    else:
                        row[1][self._roi_col_name] = ''
                        new_row = list(row[1])
                        new_data_list.append(new_row)
                        self.valid_roi_files[ind] = ''
                        ind+=1
                        self.log.write(f"Error! ROI file not found, tried: {roi_file1}.roi/zip/_mask.tif, {roi_file2}.roi/.zip/_mask.tif.\n")

                self.data_list=pd.DataFrame(new_data_list, columns=self.data_list.columns)
                self.data_list['id']=self.data_list.index + 1

            if(self.make_rainbow_tracks):
                for row in self.data_list.iterrows():
                    ind = row[1][self._index_col_name]
                    img_dir = row[1][self._img_file_col_name]
                    csv_file = row[1][self._file_col_name]

                    img_file = img_dir + "/" + self.img_file_prefix + csv_file[5:-4]  # Drop "Traj_" at beginning, add prefix, and drop ".csv" at end
                    if (not img_file.endswith(".tif")):
                        img_file = img_file + ".tif"
                    if (os.path.isfile(img_file)):
                        self.valid_img_files[ind] = img_file
                    else:
                        self.valid_img_files[ind] = ''
                        self.log.write(f"Error! Image file not found: {img_file} for rainbow tracks/ROIs.\n")

            if(self.get_calibration_from_metadata):
                for row in self.data_list.iterrows():
                    ind = row[1][self._index_col_name]
                    csv_file = row[1][self._file_col_name]

                    movie_dir = row[1][self._movie_file_col_name]
                    movie_file = movie_dir + "/" + csv_file[5:-4]  # Drop "Traj_" at beginning and ".csv" at end

                    # check first for .tif, then nd2
                    if (not movie_file.endswith(".tif")):
                        movie_file = movie_file + ".tif"
                    if (os.path.isfile(movie_file)):
                        ret_val = read_movie_metadata_tif(movie_file)
                        # some checks for reading from tiff files - since I'm not sure if the code will always work....!
                        if (ret_val[0] == ''):
                            ret_val[0] = self.micron_per_px
                            self.log.write(f"Error! Micron per pixel could not be read from tif movie file: {movie_file}.  Falling back to default settings.\n")
                        if (ret_val[1] == ''):
                            ret_val[1] = self.time_step
                            self.log.write(f"Error! Time step could not be read from tif movie file: {movie_file}.  Falling back to default settings.\n")
                        if (len(ret_val[2]) == 0 and self.uneven_time_steps):
                            self.log.write(f"Error! Full time step list could not be read from tif movie file: {movie_file}.  Falling back to default settings.\n")
                    else:
                        movie_file = movie_dir + "/" + csv_file[5:-4] + ".nd2"
                        if (os.path.isfile(movie_file)):
                            ret_val = read_movie_metadata_nd2(movie_file)
                        else:
                            ret_val = None
                            self.calibration_from_metadata[ind] = ''
                            self.log.write(f"Error! Movie file not found: {movie_file}.  Falling back to default settings.\n")
                    if (ret_val):
                        if (not self.uneven_time_steps):
                            ret_val[2] = []  # do not use full time step info / could be messed up anyway in case of tif file

                        self.calibration_from_metadata[ind] = ret_val
                        self.log.write(f"Movie file {movie_file}: microns-per-pixel={ret_val[0]}, exposure={ret_val[1]}\n")
                        if (len(ret_val[2]) > 0 and self.uneven_time_steps):
                            self.log.write(f"Full time step list: min={np.min(ret_val[2])}, {ret_val[2]}\n")

            # group by the label columns
            # set_index will throw ValueError if index col has repeats
            self.data_list.set_index(self._index_col_name, inplace=True, verify_integrity=True)
            self.label_columns = list(col_names)
            for col in self.label_columns:
                self.data_list[col] = self.data_list[col].fillna('')
            if(self.cell_column_name in self.label_columns):
                self.cell_column=True
                self.label_columns.remove(self.cell_column_name)

            self.grouped_data_list = self.data_list.groupby(self.label_columns)
            self.groups=list(self.grouped_data_list.groups.keys())

        self.group_str_to_readable={}

    def write_params_to_log_file(self):
        self.log.write("Run paramters:\n")
        self.log.write(f"Rainbow tracks: {self.make_rainbow_tracks}\n")
        self.log.write(f"Filter with ROI file: {self.limit_to_ROIs}\n")
        self.log.write(f"Read calibration from metadata: {self.get_calibration_from_metadata}\n")
        self.log.write(f"Filter for uneven time steps: {self.uneven_time_steps}\n")
        self.log.write(f"Min. time step resolution: {self.ts_resolution}\n")
        self.log.write(f"Time between frames (s): {self.time_step}\n")
        self.log.write(f"Scale (microns per px): {self.micron_per_px}\n")

        self.log.write(f"Min. track length (fit): {self.min_track_len_linfit}\n")
        self.log.write(f"Track length cutoff (fit): {self.track_len_cutoff_linfit}\n")
        self.log.write(f"Min track length (step size/angles): {self.min_track_len_step_size}\n")
        self.log.write(f"Max Tau (step size/angles): {self.max_tlag_step_size}\n")
        self.log.write(f"Min D for plots: {self.min_D_cutoff}\n")
        self.log.write(f"Max D for plots: {self.max_D_cutoff}\n")

        self.log.write(f"Max D for rainbow tracks: {self.max_D_rainbow_tracks}\n")
        self.log.write(f"Max s for rainbow tracks: {self.max_ss_rainbow_tracks}\n")

        self.log.write(f"Prefix for image file name: {self.img_file_prefix}\n")

        self.log.write(f"Results directory: {self.results_dir}\n")
        self.log.write(f"Data File: {self.data_file}\n")

        self.log.flush()

    def make_msd_diff_object(self):
        msd_diff_obj = msd_diff.msd_diffusion()
        msd_diff_obj.save_dir = self.results_dir
        msd_diff_obj.min_track_len_linfit = self.min_track_len_linfit
        msd_diff_obj.track_len_cutoff_linfit = self.track_len_cutoff_linfit
        msd_diff_obj.min_track_len_step_size = self.min_track_len_step_size
        msd_diff_obj.max_tlag_step_size = self.max_tlag_step_size

        msd_diff_obj.time_step = self.time_step
        msd_diff_obj.micron_per_px = self.micron_per_px

        return msd_diff_obj

    def make_traj_len_histograms(self):
        pass

    def sort_legend(self, ax, legend_title=True):
        handles, labels = ax.get_legend_handles_labels()
        if (legend_title):
            h1 = handles[0]
            l1 = labels[0]
            handles = handles[1:]
            labels = labels[1:]

        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

        if (legend_title):
            handles=list(handles)
            labels=list(labels)
            handles.insert(0, h1)
            labels.insert(0, l1)
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    def plot_distribution_Deff(self, plot_type='gkde', bin_size=0.02 , min_pts=20, make_legend=False):
        self.data_list_with_results_full = pd.read_csv(self.results_dir + "/all_data.txt", index_col=0, sep='\t', low_memory=False)

        self.data_list_with_results_full['group'] = self.data_list_with_results_full['group'].astype('str')
        self.data_list_with_results_full['group_readable'] = self.data_list_with_results_full['group_readable'].astype('str')

        for group in np.unique(self.data_list_with_results_full['group_readable']):
            group_data = self.data_list_with_results_full[self.data_list_with_results_full['group_readable'] == group]

            obs_dist = group_data['D']
            if(len(obs_dist)>min_pts):
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)

                fig2 = plt.figure()
                ax2 = fig2.add_subplot(1, 1, 1)

                plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)
                if(plot_type == 'gkde'):
                    gkde = stats.gaussian_kde(obs_dist)
                    plotting_kdepdf = gkde.evaluate(plotting_ind)
                    ax2.plot(plotting_ind, plotting_kdepdf)
                else:
                    ax2.hist(obs_dist, bins=plotting_ind, histtype="step", density=True)

                ax2.set_xlabel("Deff")
                ax2.set_ylabel("frequency")

                fig2.savefig(self.results_dir + "/combined_" + str(group) + "_Deff_"+plot_type+".pdf")
                fig2.clf()
                plt.close(fig2)

                #filter by file name
                for id in np.unique(group_data["id"]):
                    cur_data = group_data[group_data["id"]==id]
                    obs_dist=cur_data['D']
                    if(len(obs_dist)>min_pts):
                        plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)

                        if (plot_type == 'gkde'):
                            gkde = stats.gaussian_kde(obs_dist)
                            plotting_kdepdf = gkde.evaluate(plotting_ind)
                            ax.plot(plotting_ind, plotting_kdepdf, label=str(cur_data[self._file_col_name].iloc[0])+'-'+str(cur_data[self._roi_col_name].iloc[0]))
                        else:
                            ax.hist(obs_dist, bins=plotting_ind, histtype="step", density=True,
                                    label=str(cur_data[self._file_col_name].iloc[0])+'-'+str(cur_data[self._roi_col_name].iloc[0]), alpha=0.6)

                ax.set_xlabel("Deff")
                ax.set_ylabel("frequency")
                if(make_legend):
                    ax.legend()
                fig.savefig(self.results_dir + "/all_" + str(group) + "_Deff_"+plot_type+".pdf")
                fig.clf()
                plt.close(fig)

    def plot_distribution_step_sizes(self, tlags=[1,2,3], plot_type='gkde', bin_size=0.01, min_pts=20, make_legend=False):
        self.data_list_with_step_sizes_full = pd.read_csv(self.results_dir + "/all_data_step_sizes.txt", index_col=0, sep='\t', low_memory=False)
        start_pos = self.data_list_with_step_sizes_full.columns.get_loc("0")
        stop_pos=len(self.data_list_with_step_sizes_full.columns) - start_pos - 1

        tlags_ = []
        for tlag in tlags:
            if(tlag in np.unique(self.data_list_with_step_sizes_full['tlag'])):
                tlags_.append(tlag)

        for group in np.unique(self.data_list_with_step_sizes_full['group_readable']):
            group_data = self.data_list_with_step_sizes_full[self.data_list_with_step_sizes_full['group_readable'] == group]

            for tlag in tlags_:
                cur_tlag_data = group_data[group_data['tlag'] == tlag]
                obs_dist = np.asarray(cur_tlag_data.loc[:, "0":str(stop_pos)]).flatten()
                obs_dist = obs_dist[np.logical_not(np.isnan(obs_dist))]
                if(len(obs_dist)>min_pts):
                    plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)

                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)

                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(1, 1, 1)

                    if(plot_type == 'gkde'):
                        gkde = stats.gaussian_kde(obs_dist)
                        plotting_kdepdf = gkde.evaluate(plotting_ind)
                        ax2.plot(plotting_ind, plotting_kdepdf)
                    else:
                        ax2.hist(obs_dist, bins=plotting_ind, histtype="step", density=True)
                    ax2.set_xlabel("microns")
                    ax2.set_ylabel("frequency")

                    fig2.savefig(self.results_dir + '/combined_tlag' + str(tlag) + '_' + str(group) + '_steps_'+plot_type+'.pdf')
                    fig2.clf()
                    plt.close(fig2)

                    for id in cur_tlag_data['id'].unique():
                        cur_kde_data = cur_tlag_data[cur_tlag_data['id'] == id]
                        obs_dist=np.asarray(cur_kde_data.loc[:,"0":str(stop_pos)].iloc[0].dropna())
                        if(len(obs_dist)>min_pts):
                            plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)

                            if(plot_type == 'gkde'):
                                gkde = stats.gaussian_kde(obs_dist)
                                plotting_kdepdf=gkde.evaluate(plotting_ind)
                                ax.plot(plotting_ind, plotting_kdepdf, label=str(cur_kde_data[self._file_col_name].iloc[0])+'-'+str(cur_kde_data[self._roi_col_name].iloc[0]))
                            else:
                                ax.hist(obs_dist, bins=plotting_ind, histtype="step", density=True,
                                        label=str(cur_kde_data[self._file_col_name].iloc[0])+'-'+str(cur_kde_data[self._roi_col_name].iloc[0]), alpha=0.6)


                    ax.set_xlabel("microns")
                    ax.set_ylabel("frequency")
                    if(make_legend):
                        ax.legend()
                    fig.savefig(self.results_dir + '/all_tlag'+str(tlag)+'_'+str(group)+'_steps_'+plot_type+'.pdf')
                    fig.clf()
                    plt.close(fig)

    def plot_distribution_angles(self, tlags=[1, 2, 3], plot_type='gkde', bin_size=1, min_pts=20, make_legend=False):
        self.data_list_with_angles = pd.read_csv(self.results_dir + "/all_data_angles.txt", index_col=0,
                                                 sep='\t', low_memory=False)
        start_pos = self.data_list_with_angles.columns.get_loc("0")
        stop_pos = len(self.data_list_with_angles.columns) - start_pos - 1

        tlags_ = []
        for tlag in tlags:
            if (tlag in np.unique(self.data_list_with_angles['tlag'])):
                tlags_.append(tlag)

        for group in np.unique(self.data_list_with_angles['group_readable']):
            group_data = self.data_list_with_angles[self.data_list_with_angles['group_readable'] == group]

            for tlag in tlags_:
                cur_tlag_data = group_data[group_data['tlag'] == tlag]
                obs_dist = np.asarray(cur_tlag_data.loc[:, "0":str(stop_pos)]).flatten()
                obs_dist = obs_dist[np.logical_not(np.isnan(obs_dist))]
                if(len(obs_dist) > min_pts):
                    plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)

                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)

                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(1, 1, 1)

                    if (plot_type == 'gkde'):
                        gkde = stats.gaussian_kde(obs_dist)
                        plotting_kdepdf = gkde.evaluate(plotting_ind)
                        ax2.plot(plotting_ind, plotting_kdepdf)
                    else:
                        ax2.hist(obs_dist, bins=plotting_ind, histtype="bar", density=True)

                    ax2.set_xlabel("angle (degrees)")
                    ax2.set_ylabel("frequency")

                    fig2.savefig(self.results_dir + '/combined_tlag' + str(tlag) + '_' + str(
                        group) + '_angles_' + plot_type + '.pdf')
                    fig2.clf()
                    plt.close(fig2)

                    for id in cur_tlag_data['id'].unique():
                        cur_kde_data = cur_tlag_data[cur_tlag_data['id'] == id]
                        obs_dist = np.asarray(cur_kde_data.loc[:, "0":str(stop_pos)].iloc[0].dropna())
                        if(len(obs_dist)>min_pts):
                            plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)

                            if (plot_type == 'gkde'):
                                gkde = stats.gaussian_kde(obs_dist)
                                plotting_kdepdf = gkde.evaluate(plotting_ind)
                                ax.plot(plotting_ind, plotting_kdepdf, label=str(cur_kde_data[self._file_col_name].iloc[0])+'-'+str(cur_kde_data[self._roi_col_name].iloc[0]))
                            else:
                                ax.hist(obs_dist, bins=plotting_ind, histtype="bar", density=True,
                                        label=str(cur_kde_data[self._file_col_name].iloc[0])+'-'+str(cur_kde_data[self._roi_col_name].iloc[0]), alpha=0.6)

                    ax.set_xlabel("angles (degrees)")
                    ax.set_ylabel("frequency")
                    if(make_legend):
                        ax.legend()
                    fig.savefig(self.results_dir + '/all_tlag' + str(tlag) + '_' + str(group) + '_angles_' + plot_type + '.pdf')
                    fig.clf()
                    plt.close(fig)

    def make_by_cell_plot(self, label, label_order):
        #group is on the x-axis
        #separate plot for each combination of labels from all other columns
        #data is read in from msd/diff in the saved files

        self.data_list_with_results_full = pd.read_csv(self.results_dir + "/all_data.txt",index_col=0,sep='\t')

        label_columns = self.label_columns[:]
        label_columns.remove(label)
        if (len(label_columns) > 0):
            grouped_data_list = self.data_list.groupby(label_columns)
            groups = list(grouped_data_list.groups.keys())
        else:
            groups = []

        if(len(groups)>0):

            for group_i, group in enumerate(groups):
                group_str=''
                if(type(group)==type("")):
                    group_str = group
                else:
                    for g in group:
                        group_str += (g + '_')
                        group_str=group_str[:-1]
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                group_df = grouped_data_list.get_group(group)
                id_list = group_df.index
                cur_data=self.data_list_with_results_full[self.data_list_with_results_full.id.isin(id_list)].copy()

                cur_data[label + '_tonum'] = -1
                for order_i, order_label in enumerate(label_order):
                    cur_data[label + '_tonum'] = np.where(cur_data[label].astype('str') == str(order_label), order_i,
                                                          cur_data[label + '_tonum'])
                cur_data['cell'] = "cell " + cur_data['cell'].astype('str')

                sns.lineplot(x=label+'_tonum', y="D", data=cur_data, hue="cell", estimator=np.median, ax=ax)

                self.sort_legend(ax)

                ax.set_xticks(range(len(label_order)))
                ax.set_xticklabels(label_order)
                plt.xticks(rotation='vertical')
                plt.xlabel(label)

                plt.tight_layout()
                fig.savefig(self.results_dir + '/' + group_str + '_by_cell.pdf')
                fig.clf()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            cur_data = self.data_list_with_results_full.copy()

            cur_data[label + '_tonum'] = -1
            for order_i, order_label in enumerate(label_order):
                cur_data[label + '_tonum'] = np.where(cur_data[label].astype('str') == str(order_label), order_i,
                                                      cur_data[label + '_tonum'])
            cur_data['cell']="cell "+cur_data['cell'].astype('str')

            sns.lineplot(x=label + '_tonum', y="D", data=cur_data, hue="cell", estimator=np.median, ax=ax)

            self.sort_legend(ax)

            ax.set_xticks(range(len(label_order)))
            ax.set_xticklabels(label_order)
            plt.xticks(rotation='vertical')
            plt.xlabel(label)

            plt.tight_layout()
            fig.savefig(self.results_dir + '/all_by_cell.pdf')
            fig.clf()

    def make_heatmap_step_sizes(self, label_order=[], bin_width=0.001, min_step_size=0, max_step_size=0.101):
        #bin_width=0.009, min_step_size=0.1, max_step_size=1): #bin_width=0.001, min_step_size=0, max_step_size=0.1):
        #bin_width=0.01, min_step_size=0, max_step_size=1
        #bin_width=0.005, min_step_size=0.1, max_step_size=1):

        if (label_order):
            labels = label_order
        else:
            labels = np.unique(self.data_list_with_results['group_readable'])
            labels.sort()

        self.data_list_with_results_full = pd.read_csv(self.results_dir + '/' + "all_data_step_sizes.txt", index_col=0,sep='\t')
        start_pos = self.data_list_with_results_full.columns.get_loc("0")
        stop_pos = len(self.data_list_with_results_full.columns) - start_pos - 1

        for tlag in range(1, self.max_tlag_step_size + 1, 1):
            cur_data=self.data_list_with_results_full[self.data_list_with_results_full["tlag"]==tlag]
            the_bins=np.arange(min_step_size, max_step_size, bin_width)
            df = pd.DataFrame()
            df['step_sizes']=np.flip(the_bins[1:])

            df_full=pd.DataFrame()
            df_full['step_sizes']=np.flip(the_bins[1:])

            for label in labels:
                cur_data_at_label = cur_data[cur_data['group_readable'] == label]

                cur_heatmap_data=[]
                cur_medians=[]
                for row_i,row in enumerate(cur_data_at_label.iterrows()):
                    cur_row_data=np.asarray(row[1]["0":str(stop_pos)].transpose()).astype('float64')

                    to_plot = cur_row_data[np.logical_not(np.isnan(cur_row_data))]
                    to_plot = to_plot[to_plot[:]>=min_step_size]
                    to_plot = to_plot[to_plot[:] <= max_step_size]
                    fn=row[1]['file name']
                    mo=re.match('.+([ABCDEF]001)\.tif\.csv', fn)
                    if(mo):
                        cell=mo.group(1)
                    else:
                        mo=re.match('.+([ABCDEF]001) stack\.tif\.csv',fn)
                        if(mo):
                            cell=mo.group(1)
                        else:
                            print("Error could not location cell name in file name.")
                            cell="?"

                    cur_medians.append([row_i,np.median(to_plot),cell])
                    ret = plt.hist(to_plot, bins=the_bins,histtype='step')  # , density=True)
                    cur_heatmap_data.append(np.flip(ret[0] / np.sum(ret[0])) * 100)

                cur_medians.sort(key=lambda x: x[1])
                for m_i,cur_median in enumerate(cur_medians):
                    df_full[label+'_'+cur_median[2]]=cur_heatmap_data[cur_median[0]]

                cur_data_at_label = np.asarray(cur_data_at_label.loc[:, "0":str(stop_pos)].transpose())
                to_plot=cur_data_at_label[np.logical_not(np.isnan(cur_data_at_label))]
                ret=plt.hist(to_plot, bins=the_bins) #, density=True)
                df[label]=np.flip(ret[0]/np.sum(ret[0]))*100
                plt.clf()
                plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            df['step_sizes']=np.round(df['step_sizes'],3)
            df.set_index('step_sizes', inplace=True, )
            sns.heatmap(data=df, cmap='jet',yticklabels=False, ax=ax)
            plt.tight_layout()
            fig.savefig(self.results_dir + '/summary_combined_step_size_' + str(tlag) + '_heatmap.pdf')
            fig.clf()

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            df_full['step_sizes'] = np.round(df_full['step_sizes'], 3)
            df_full.set_index('step_sizes', inplace=True, )
            sns.heatmap(data=df_full, cmap='jet', xticklabels=True, yticklabels=20, ax=ax)
            plt.tick_params(axis='x', which='major', labelsize=4)
            plt.tick_params(axis='y', which='major', labelsize=6)
            plt.tight_layout()
            fig.savefig(self.results_dir + '/summary_combined_step_size_full_' + str(tlag) + '_heatmap.pdf')
            fig.clf()

    def make_plot_step_sizes(self, label_order=[], plot_labels=[], xlabel='', ylabel='', clrs=[], combine_data=False):

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.data_list_with_results['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.data_list_with_results['group_readable'])
            labels.sort()

        if (plot_labels):
            self.data_list_with_results['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.data_list_with_results['group_readable'] = np.where(
                    self.data_list_with_results['group'] == str(labels[i]),
                    plot_label, self.data_list_with_results['group_readable'])
            labels = plot_labels

        if(combine_data):

            self.data_list_with_results_full = pd.read_csv(self.results_dir + '/' + "all_data_step_sizes.txt", index_col=0,sep='\t')
            start_pos = self.data_list_with_results_full.columns.get_loc("0")
            stop_pos = len(self.data_list_with_results_full.columns) - start_pos - 1

            for tlag in range(1, self.max_tlag_step_size + 1, 1):
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)

                cur_data=self.data_list_with_results_full[self.data_list_with_results_full["tlag"]==tlag]
                to_plot=[]
                for label in labels:
                    cur_data_at_label=cur_data[cur_data['group_readable']==label]
                    cur_data_at_label = np.asarray(cur_data_at_label.loc[:, "0":str(stop_pos)].transpose())
                    to_append=cur_data_at_label[np.logical_not(np.isnan(cur_data_at_label))]
                    to_plot.append(list(to_append))

                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(1, 1, 1)

                    ret=ax2.hist(to_append, bins=np.arange(0, np.max(to_append),0.01), density=True)
                    fig2.savefig(self.results_dir + '/distribution_combined_step_size_' + str(tlag) + '_'+label+'.pdf')
                    plt.close(fig2)

                ax.boxplot(to_plot, labels=labels,showfliers=False)

                ax.set(xlabel=xlabel)
                if (ylabel != ''):
                    ax.set(ylabel=ylabel)
                else:
                    ax.set(ylabel=f"step sizes, tlag={tlag} (microns)")
                plt.xticks(rotation='vertical')
                plt.tight_layout()
                fig.savefig(self.results_dir + '/summary_combined_step_size_' + str(tlag) + '_nf.pdf')
                fig.clf()
        else:
            self.data_list_with_results = pd.read_csv(self.results_dir + '/'+"summary_step_sizes.txt",index_col=0,sep='\t')
            for tlag in range(1,self.max_tlag_step_size+1,1):
                y_col="step_size_"+str(tlag)+"_median"

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                if (clrs != []):
                    sns.boxplot(x="group_readable", y=y_col, data=self.data_list_with_results, order=labels,
                                fliersize=0, ax=ax, palette=clrs)
                else:
                    sns.boxplot(x="group_readable", y=y_col, data=self.data_list_with_results, order=labels,
                                fliersize=0, ax=ax)
                sns.swarmplot(x="group_readable", y=y_col, data=self.data_list_with_results, order=labels, color=".25",
                              size=4, ax=ax)
                ax.set(xlabel=xlabel)
                if (ylabel != ''):
                    ax.set(ylabel=ylabel)
                else:
                    ax.set(ylabel=f"med(step sizes), tlag={tlag} (microns)")
                plt.xticks(rotation='vertical')
                plt.tight_layout()
                fig.savefig(self.results_dir + '/summary_' + y_col + '.pdf')
                fig.clf()

    def make_plot_combined_data(self, label_order=[], plot_labels=[], xlabel='', ylabel='', clrs=[]):

        self.data_list_with_results_full = pd.read_csv(self.results_dir + '/'+"all_data.txt",index_col=0,sep='\t')

        self.data_list_with_results_full['group'] = self.data_list_with_results_full['group'].astype('str')
        self.data_list_with_results_full['group_readable'] = self.data_list_with_results_full['group_readable'].astype('str')

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.data_list_with_results['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.data_list_with_results_full['group_readable'])
            labels.sort()

        if (plot_labels):
            self.data_list_with_results_full['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.data_list_with_results_full['group_readable'] = np.where(
                    self.data_list_with_results_full['group'] == str(labels[i]),
                    plot_label,
                    self.data_list_with_results_full['group_readable'])
            labels = plot_labels

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if (clrs != []):
            sns.pointplot(x="group_readable", y='D', data=self.data_list_with_results_full, order=labels, estimator=np.median(), #showfliers=0, #fliersize=0,
                        ax=ax, palette=clrs)
        else:
            sns.pointplot(x="group_readable", y='D', data=self.data_list_with_results_full, order=labels, estimator=np.median, #showfliers=False, #fliersize=0,
                        ax=ax)

        ax.set(xlabel=xlabel)
        if (ylabel != ''):
            ax.set(ylabel=ylabel)
        else:
            ax.set(ylabel="Deff")

        plt.xticks(rotation='vertical')
        plt.tight_layout()
        fig.savefig(self.results_dir + '/summary_combined_D.pdf')
        fig.clf()

    def make_plot(self, label_order=[], plot_labels=[], xlabel='', ylabel='', clrs=[]):
        # label_order should match the group labels (i.e. group/group_readable)
        # it is just imposing an order for plotting

        # additionally to label_order, new names can be given using plot_labels
        # these must corresponding the the labels in label_order, position by position

        self.data_list_with_results = pd.read_csv(self.results_dir + '/' + "summary.txt", sep='\t')

        self.data_list_with_results['group']=self.data_list_with_results['group'].astype('str')
        self.data_list_with_results['group_readable'] = self.data_list_with_results['group_readable'].astype('str')

        if(label_order):
            label_order_=[]
            for l in label_order:
                if(l in list(self.data_list_with_results['group_readable'])):
                    label_order_.append(l)
            labels=label_order_
        else:
            labels = np.unique(self.data_list_with_results['group_readable'])
            labels.sort()

        if(plot_labels):
            self.data_list_with_results['group_readable']=''
            for i, plot_label in enumerate(plot_labels):
                self.data_list_with_results['group_readable']=np.where(self.data_list_with_results['group']==str(labels[i]),
                                                                       plot_label,self.data_list_with_results['group_readable'])
            labels=plot_labels

        for y_col in ['D_median','D_median_filtered']:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if(clrs != []):
                sns.boxplot(x="group_readable", y=y_col, data=self.data_list_with_results, order=labels, fliersize=0, ax=ax, palette=clrs)
            else:
                sns.boxplot(x="group_readable", y=y_col, data=self.data_list_with_results, order=labels, fliersize=0, ax=ax)

            sns.swarmplot(x="group_readable", y=y_col, data=self.data_list_with_results, order=labels, color=".25", size=4, ax=ax)
            #ax.set(xlabel="X Label", ylabel = "Y Label")
            ax.set(xlabel=xlabel)
            if (ylabel != ''):
                ax.set(ylabel = ylabel)
            else:
                ax.set(ylabel = "med(Deff)")
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            fig.savefig(self.results_dir + '/summary_'+y_col+'.pdf')
            fig.clf()

    def make_plot_roi_area(self, label_order=[], plot_labels=[], xlabel='', ylabel='', clrs=[]):
        #make plot of ROI Area vs. med(Deff)
        self.data_list_with_results = pd.read_csv(self.results_dir + '/' + "summary.txt", sep='\t')

        self.data_list_with_results['group'] = self.data_list_with_results['group'].astype('str')
        self.data_list_with_results['group_readable'] = self.data_list_with_results['group_readable'].astype('str')

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.data_list_with_results['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.data_list_with_results['group_readable'])
            labels.sort()

        if (plot_labels):
            self.data_list_with_results['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.data_list_with_results['group_readable'] = np.where(
                    self.data_list_with_results['group'] == str(labels[i]),
                    plot_label, self.data_list_with_results['group_readable'])
            labels = plot_labels

        for y_col in ['D_median','D_median_filtered']:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            if (clrs != []):
                sns.scatterplot(x="area", y=y_col, data=self.data_list_with_results, hue='group_readable',
                                ax=ax, palette=clrs)
            else:
                sns.scatterplot(x="area", y=y_col, data=self.data_list_with_results, hue='group_readable', ax=ax)

            if (xlabel != ''):
                ax.set(xlabel=xlabel)
            else:
                ax.set(xlabel="roi area (microns)")

            if (ylabel != ''):
                ax.set(ylabel=ylabel)
            else:
                ax.set(ylabel="Deff")

            plt.tight_layout()
            fig.savefig(self.results_dir + f"/roi-area_vs_{y_col}.pdf")
            fig.clf()

    def read_track_data_file(self, file_name):
        ext = (os.path.splitext(file_name)[1]).lower()
        if (ext == '.csv'):
            sep = ','
        elif (ext == '.txt'):
            sep = '\t'
        else:
            self.log.write(f"Error in reading {file_name}: all input files must have extension txt or csv. Skipping...")
            self.log.flush()
            return pd.DataFrame()
        track_data_df = pd.read_csv(file_name, sep=sep)
        for col_name in [self.traj_id_col, self.traj_frame_col, self.traj_x_col, self.traj_y_col]:
            if(not (col_name in track_data_df.columns)):
                self.log.write(f"Error in reading {file_name}: required column {col_name} is missing.\n")
                self.log.flush()
                return pd.DataFrame()

        track_data_df = track_data_df[[self.traj_id_col, self.traj_frame_col, self.traj_x_col, self.traj_y_col]]
        return track_data_df

    def filter_ROI(self, index, roi_name, df):
        if (index in self.valid_roi_files and self.valid_roi_files[index] != ''):
            roi_file = self.valid_roi_files[index]
            err = False
            if (roi_file.endswith('roi')):
                rois = read_roi_file(roi_file)
            elif (roi_file.endswith('zip')):
                rois = read_roi_zip(roi_file)
            elif (roi_file.endswith('_mask.tif')):
                mask = io.imread(roi_file)
                mask=mask>0
                labeled, n = ndimage.label(mask)
                mask= (labeled == roi_name)
                mask = mask.astype('uint8')
            else:
                self.log.write(f"Invalid ROI file. ({roi_file})\n")
                self.log.flush()
                err = True

            # make mask from ROI for track exclusion
            if ((not err) and (not roi_file.endswith('_mask.tif'))):  # make mask from ROI
                if (index in self.valid_img_files and self.valid_img_files[index] != ''):
                    img = io.imread(self.valid_img_files[index])

                    (mask, err) = make_mask_from_roi(rois, roi_name, (img.shape[0], img.shape[1]))
                    if (err):
                        self.log.write(f"Unsupported ROI type.  Make a mask instead. ({roi_file})\n")
                        self.log.flush()
                else:
                    self.log.write(f"Cannot load tif image file for ROI file: ({roi_file}).\n")
                    self.log.flush()
                    err = True
            roi_area=0
            if (not err):
                # limit the tracks to the ROI - returns track ids that are fully within mask
                valid_id_list = limit_tracks_given_mask(mask, df.to_numpy())
                df = df[df['Trajectory'].isin(valid_id_list)]
                roi_area=np.sum(mask.flatten())
        else:
            # no action here - error will already have been printed by class init function #
            pass
        return (df,roi_area)

    def set_rows_to_none_ss(self, i, a, max, g, gr):
        for tlag_i in range(1, max, 1):
            self.data_list_with_step_sizes.at[i, 'step_size_' + str(tlag_i) + '_median'] = np.nan
            self.data_list_with_step_sizes.at[i, 'step_size_' + str(tlag_i) + '_mean'] = np.nan
        self.data_list_with_step_sizes.at[i, 'area'] = a
        self.data_list_with_step_sizes.at[i, 'group'] = g
        self.data_list_with_step_sizes.at[i, 'group_readable'] = gr

    def calculate_step_sizes_and_angles(self, save_per_file_data=False):
        group_list = self.groups

        # get total number of tracks for all groups/all files so I can make a large dataframe to fill
        max_tlag1_ss_num_steps = 0
        max_tlag1_angle_num_steps = 0
        for group_i, group in enumerate(group_list):
            group_df = self.grouped_data_list.get_group(group)
            for index, data in group_df.iterrows():
                cur_dir = data[self._dir_col_name]
                cur_file = data[self._file_col_name]

                track_data = self.read_track_data_file(cur_dir + '/' + cur_file)
                track_data = track_data.to_numpy()
                if (len(track_data) == 0):
                    continue

                ids = np.unique(track_data[:, 0])
                track_lengths = np.zeros((len(ids), 2))
                for i, id in enumerate(ids):
                    cur_track = track_data[np.where(track_data[:, 0] == id)]
                    track_lengths[i, 0] = id
                    track_lengths[i, 1] = len(cur_track)

                filt_track_lengths = track_lengths[track_lengths[:, 1] >= self.min_track_len_step_size][:, 1]
                tlag1_ss_num_steps = int(np.sum(filt_track_lengths - 1))
                tlag1_angle_num_steps = int(np.sum(filt_track_lengths - 2))

                if(tlag1_ss_num_steps > max_tlag1_ss_num_steps):
                    max_tlag1_ss_num_steps = tlag1_ss_num_steps
                if (tlag1_angle_num_steps > max_tlag1_angle_num_steps):
                    max_tlag1_angle_num_steps = tlag1_angle_num_steps

        # make a full dataframe containing all data - step sizes
        nrows=self.max_tlag_step_size*len(self.data_list)
        ncols=len(self.data_list.columns)+max_tlag1_ss_num_steps
        colnames=list(self.data_list.columns)
        endpos=len(colnames)
        rest_cols=np.asarray(range(max_tlag1_ss_num_steps))
        rest_cols=rest_cols.astype('str')
        colnames.extend(rest_cols)
        self.data_list_with_step_sizes_full = pd.DataFrame(np.empty((nrows, ncols), dtype=np.str), columns=colnames)
        self.data_list_with_step_sizes_full.insert(loc=0, column='id', value='')
        self.data_list_with_step_sizes_full.insert(loc=endpos+1, column='group', value='')
        self.data_list_with_step_sizes_full.insert(loc=endpos+2, column='group_readable', value='')
        self.data_list_with_step_sizes_full.insert(loc=endpos+3, column='tlag', value=0)
        self.data_list_with_step_sizes_full['tlag']=np.tile(range(1,self.max_tlag_step_size+1),len(self.data_list))

        # make a dataframe containing only median and mean step size values for each movie
        self.data_list_with_step_sizes = self.data_list.copy()
        for tlag_i in range(1,self.max_tlag_step_size+1,1):
            self.data_list_with_step_sizes['step_size_'+str(tlag_i)+'_median'] = 0.0
            self.data_list_with_step_sizes['step_size_'+str(tlag_i)+'_mean'] = 0.0
        self.data_list_with_step_sizes['area'] = ''
        self.data_list_with_step_sizes['group'] = ''
        self.data_list_with_step_sizes['group_readable'] = ''

        # make a full dataframe containing all data - angles TODO test and uncomment this
        # nrows =  self.max_tlag_step_size * len(self.data_list)
        # ncols = len(self.data_list.columns) + max_tlag1_angle_num_steps
        # colnames = list(self.data_list.columns)
        # endpos = len(colnames)
        # rest_cols = np.asarray(range(max_tlag1_angle_num_steps))
        # rest_cols = rest_cols.astype('str')
        # colnames.extend(rest_cols)
        # self.data_list_with_angles = pd.DataFrame(np.empty((nrows, ncols), dtype=np.str), columns=colnames)
        # self.data_list_with_angles.insert(loc=0, column='id', value=0)
        # self.data_list_with_angles.insert(loc=endpos + 1, column='group', value='')
        # self.data_list_with_angles.insert(loc=endpos + 2, column='group_readable', value='')
        # self.data_list_with_angles.insert(loc=endpos + 3, column='tlag', value=0)
        # self.data_list_with_angles['tlag'] = np.tile(range(1, self.max_tlag_step_size+1), len(self.data_list))

        msd_diff_obj = self.make_msd_diff_object()

        full_data_ss_i = 0
        full_data_a_i = 0
        for group_i, group in enumerate(group_list):
            group_df = self.grouped_data_list.get_group(group)
            file_str = ""

            if (type(group) == type("")):
                file_str = str(group)
            else:
                for label_i, label in enumerate(group):
                    file_str += (str(label) + '_')
                file_str = file_str[:-1]

            group_readable = file_str
            if (self.group_str_to_readable and file_str in self.group_str_to_readable):
                group_readable = self.group_str_to_readable[file_str]

            for index, data in group_df.iterrows():
                cur_dir = data[self._dir_col_name]
                cur_file = data[self._file_col_name]

                print(cur_file, data[self._roi_col_name])

                roi_area=''
                track_data_df = self.read_track_data_file(cur_dir + '/' + cur_file)
                if (len(track_data_df) == 0):
                    self.log.write("Note!  File '" + cur_dir + "/" + cur_file + "' contains 0 tracks.\n")
                    self.log.flush()
                    self.set_rows_to_none_ss(index, '', self.max_tlag_step_size + 1, file_str, group_readable)
                    continue

                if (self.limit_to_ROIs):
                    (track_data_df, roi_area) = self.filter_ROI(index, data[self._roi_col_name], track_data_df)
                    if (len(track_data_df) == 0):
                        self.log.write("Note!  File '" + cur_dir + "/" + cur_file + "' contains 0 tracks after ROI filtering.\n")
                        self.log.flush()
                        self.set_rows_to_none_ss(index, '', self.max_tlag_step_size + 1, file_str, group_readable)
                        continue

                #check if we need to set the calibration for this file
                if(self.get_calibration_from_metadata):

                    if(index in self.calibration_from_metadata and self.calibration_from_metadata[index] != ''):
                        m_px=self.calibration_from_metadata[index][0]
                        exposure=self.calibration_from_metadata[index][1]
                        step_sizes=self.calibration_from_metadata[index][2]
                        msd_diff_obj.micron_per_px=m_px
                        if(len(step_sizes)>0):
                            msd_diff_obj.time_step=np.min(step_sizes)
                        else:
                            msd_diff_obj.time_step=exposure

                        # if we have varying step sizes, must filter tracks
                        if (len(np.unique(step_sizes)) > 0):  ## TODO fix this so it checks whether the largest diff. is > mindiff
                            track_data_df = filter_tracks(track_data_df, self.min_track_len_linfit, step_sizes, self.ts_resolution)
                            # save the new, filtered CSVs
                            track_data_df.to_csv(self.results_dir + '/' + cur_file[:-4] + "_filtered.csv")

                            if (len(track_data_df) == 0):
                                self.log.write("Note!  File '" + cur_dir + "/" + cur_file + "' contains 0 tracks after time step filtering.\n")
                                self.log.flush()
                                self.set_rows_to_none_ss(index, '', self.max_tlag_step_size + 1, file_str, group_readable)
                                continue

                track_data = track_data_df.to_numpy()

                # convert ROI area to microns, now that we have the correct scaling information
                if (self.limit_to_ROIs and roi_area != ''):
                    roi_area = roi_area * msd_diff_obj.micron_per_px

                # for this movie, calcuate step sizes and angles for each track
                msd_diff_obj.set_track_data(track_data)
                msd_diff_obj.step_sizes_and_angles()

                cur_data_step_sizes = msd_diff_obj.step_sizes
                #cur_data_angles = msd_diff_obj.angles
                ss_len=len(cur_data_step_sizes)
                #a_len=len(cur_data_angles)

                if(ss_len == 0):
                    self.log.write("Note!  File '" + cur_dir + "/" + cur_file +
                                   "' contains 0 tracks of minimum length for calculating step sizes/angles (" +
                                   str(msd_diff_obj.min_track_len_step_size) + ")\n")
                    self.log.flush()
                    self.set_rows_to_none_ss(index, roi_area, self.max_tlag_step_size+1, file_str, group_readable)
                    continue

                #fill step size data
                self.data_list_with_step_sizes_full.loc[full_data_ss_i:full_data_ss_i+ss_len-1,'id']=index
                for k in range(len(self.data_list.columns)):
                    self.data_list_with_step_sizes_full.iloc[full_data_ss_i:full_data_ss_i+ss_len,k+1]=self.data_list.loc[index][k]
                self.data_list_with_step_sizes_full.loc[full_data_ss_i:full_data_ss_i+ss_len-1,'group']=file_str
                self.data_list_with_step_sizes_full.loc[full_data_ss_i:full_data_ss_i+ss_len-1,'group_readable']=group_readable
                self.data_list_with_step_sizes_full.loc[full_data_ss_i:full_data_ss_i+ss_len-1,"0":str(len(msd_diff_obj.step_sizes[0])-1)]=msd_diff_obj.step_sizes

                for tlag_i in range(1,self.max_tlag_step_size+1,1):
                    ss_median = np.nanmedian(msd_diff_obj.step_sizes[tlag_i-1]) # FIX
                    ss_mean = np.nanmean(msd_diff_obj.step_sizes[tlag_i-1]) # FIX

                    self.data_list_with_step_sizes.at[index,'step_size_' + str(tlag_i) + '_median'] = ss_median
                    self.data_list_with_step_sizes.at[index,'step_size_' + str(tlag_i) + '_mean'] = ss_mean

                self.data_list_with_step_sizes.at[index, 'area'] = roi_area
                self.data_list_with_step_sizes.at[index, 'group'] = file_str
                self.data_list_with_step_sizes.at[index, 'group_readable'] = group_readable

                #fill angle data TODO test and uncomment
                # self.data_list_with_angles.loc[full_data_a_i:full_data_a_i+a_len-1,'id']=index
                # for k in range(len(self.data_list.columns)):
                #     self.data_list_with_angles.iloc[full_data_a_i:full_data_a_i+a_len,k+1]=self.data_list.loc[index][k]
                # self.data_list_with_angles.loc[full_data_a_i:full_data_a_i+a_len-1,'group']=file_str
                # self.data_list_with_angles.loc[full_data_a_i:full_data_a_i+a_len-1,'group_readable']=group_readable
                # self.data_list_with_angles.loc[full_data_a_i:full_data_a_i+a_len-1,"0":str(len(msd_diff_obj.angles[0])-1)]=msd_diff_obj.angles

                if (save_per_file_data):
                    msd_diff_obj.save_step_sizes(file_name=file_str + '_' + str(index) + "_step_sizes.txt")
                    msd_diff_obj.save_angles(file_name=file_str + '_' + str(index) + "_angles.txt")

                full_data_ss_i += len(cur_data_step_sizes)
                #full_data_a_i +=  len(cur_data_angles)

                self.log.write("Processed " + str(index) +" "+cur_file + "for step sizes and angles.\n")
                self.log.flush()

        self.data_list_with_step_sizes.to_csv(self.results_dir + '/' + "summary_step_sizes.txt", sep='\t')

        if ((self.get_calibration_from_metadata or self.limit_to_ROIs)):
            self.data_list_with_step_sizes_full=self.data_list_with_step_sizes_full.replace('', np.NaN)
            self.data_list_with_step_sizes_full.dropna(axis=1,how='all',inplace=True)
            self.data_list_with_step_sizes_full.dropna(axis=0,subset=['id',],inplace=True)

            #self.data_list_with_angles = self.data_list_with_angles.replace('', np.NaN) # TODO test and uncomment
            #self.data_list_with_angles.dropna(axis=1, how='all', inplace=True)

        self.data_list_with_step_sizes_full.to_csv(self.results_dir + '/' + "all_data_step_sizes.txt", sep='\t')
        #self.data_list_with_angles.to_csv(self.results_dir + '/' + "all_data_angles.txt", sep='\t') # TODO test and uncomment

    def set_rows_to_none(self, i, a, l, g, gr):
        self.data_list_with_results.at[i, 'D_median'] = np.nan
        self.data_list_with_results.at[i, 'D_mean'] = np.nan
        self.data_list_with_results.at[i, 'D_median_filtered'] = np.nan
        self.data_list_with_results.at[i, 'D_mean_filtered'] = np.nan
        self.data_list_with_results.at[i, 'num_tracks'] = l
        self.data_list_with_results.at[i, 'num_tracks_D'] = 0
        self.data_list_with_results.at[i, 'area'] = a
        self.data_list_with_results.at[i, 'group'] = g
        self.data_list_with_results.at[i, 'group_readable'] = gr

    def calculate_msd_and_diffusion(self, save_per_file_data=False):
        # calculates the msd and diffusion data for ALL groups

        group_list = self.groups

        # get total number of tracks for all groups/all files so I can make a large dataframe to fill
        full_length=0
        for group_i,group in enumerate(group_list):
            group_df = self.grouped_data_list.get_group(group)
            for index,data in group_df.iterrows():
                cur_dir = data[self._dir_col_name]
                cur_file = data[self._file_col_name]

                self.log.write("Reading track data file: "+cur_dir + '/' + cur_file+"\n")
                self.log.flush()
                track_data = self.read_track_data_file(cur_dir + '/' + cur_file)
                track_data = track_data.to_numpy()
                if(len(track_data) == 0):
                    continue
                #filter for tracks with min length (Diffusion data will only be returned for these tracks)
                hist, bin_edges=np.histogram(track_data[:,0], bins=range(1,int(np.max(track_data[:,0])+2),1))
                full_length += len(hist[hist >= self.min_track_len_linfit])

        # make a full dataframe containing all data, including all D values for all tracks etc.
        # NOTE: if track data must be filtered b/c of uneven time steps (done below), then this array may not be filled completely
        full_results1 = pd.DataFrame(np.empty((full_length, len(self.data_list.columns)), dtype=np.str),
                                     columns=list(self.data_list.columns))
        full_results1.insert(loc=0, column='id', value=0)
        full_results1['group']=''
        full_results1['group_readable']=''
        full_results2_cols1=['D_median','D_mean','D_median_filt','D_mean_filt']
        full_results2_cols2=['D','err','r_sq','rmse','track_len','D_track_len']
        cols_len=len(full_results2_cols1) + len(full_results2_cols2)
        full_results2 = pd.DataFrame(np.zeros((full_length, cols_len)), columns=full_results2_cols1+full_results2_cols2)
        self.data_list_with_results_full = pd.concat([full_results1,full_results2], axis=1)

        # make a dataframe containing only median and mean D values for each movie
        self.data_list_with_results = self.data_list.copy()
        self.data_list_with_results['D_median']=0.0
        self.data_list_with_results['D_mean']=0.0
        self.data_list_with_results['D_median_filtered'] = 0.0
        self.data_list_with_results['D_mean_filtered'] = 0.0
        self.data_list_with_results['num_tracks'] = 0
        self.data_list_with_results['num_tracks_D'] = 0
        self.data_list_with_results['area'] = ''
        self.data_list_with_results['group']=''
        self.data_list_with_results['group_readable'] = ''

        msd_diff_obj = self.make_msd_diff_object()

        full_data_i=0
        for group_i,group in enumerate(group_list):
            group_df = self.grouped_data_list.get_group(group)
            file_str = ""

            if (type(group) == type("")):
                file_str = str(group)
            else:
                for label_i, label in enumerate(group):
                    file_str += (str(label) + '_')
                file_str = file_str[:-1]

            group_readable = file_str
            if (self.group_str_to_readable and file_str in self.group_str_to_readable):
                group_readable = self.group_str_to_readable[file_str]

            # further group by csv file name (it may not be unique)
            files_list = np.unique(group_df[self._file_col_name])
            for cur_file in files_list:
                files_group = group_df[group_df[self._file_col_name]==cur_file]
                cur_fig=None
                cur_fig_ss=None
                cur_fig_roi=None
                cur_ax=None
                cur_ax_ss=None
                cur_ax_roi=None
                # set up the figure axes for making rainbow tracks
                if (self.make_rainbow_tracks):
                    common_index=files_group.index[0] # for the same file, the image dir is the same for all rows, just take first
                    if (common_index in self.valid_img_files and self.valid_img_files[common_index] != ''):
                        bk_img = io.imread(self.valid_img_files[common_index])

                        # plot figure to draw tracks by Deff with image in background
                        cur_fig = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=100)
                        cur_ax = cur_fig.add_subplot(1, 1, 1)
                        cur_ax.axis("off")
                        cur_ax.imshow(bk_img, cmap="gray")

                        # plot figure to draw tracks by ss with image in background
                        cur_fig_ss = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=100)
                        cur_ax_ss = cur_fig_ss.add_subplot(1, 1, 1)
                        cur_ax_ss.axis("off")
                        cur_ax_ss.imshow(bk_img, cmap="gray")

                        # plot figure to draw tracks by roi with image in background
                        cur_fig_roi = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=100)
                        cur_ax_roi = cur_fig_roi.add_subplot(1, 1, 1)
                        cur_ax_roi.axis("off")
                        cur_ax_roi.imshow(bk_img, cmap="gray")

                count=0
                roi_cmap = cm.get_cmap('jet', len(files_group))
                roi_colors = roi_cmap(range(1, len(files_group)+1))
                for index,data in files_group.iterrows():
                    cur_dir=data[self._dir_col_name]
                    cur_file=data[self._file_col_name]

                    print(cur_file, data[self._roi_col_name])

                    roi_area=''
                    track_data_df = self.read_track_data_file(cur_dir + '/' + cur_file)
                    if (len(track_data_df) == 0):
                        self.log.write("Note!  File '" + cur_dir + "/" + cur_file + "' contains 0 tracks.\n")
                        self.log.flush()
                        self.set_rows_to_none(index, '', 0, file_str, group_readable)
                        continue

                    if (self.limit_to_ROIs):
                        (track_data_df,roi_area) = self.filter_ROI(index, data[self._roi_col_name], track_data_df)

                        if (len(track_data_df) == 0):
                            self.log.write("Note!  File '" + cur_dir + "/" + cur_file + "' contains 0 tracks after ROI filtering.\n")
                            self.log.flush()
                            self.set_rows_to_none(index, '', 0, file_str, group_readable)
                            continue

                    # check if we need to set the calibration for this file
                    if (self.get_calibration_from_metadata):
                        if(index in self.calibration_from_metadata and self.calibration_from_metadata[index] != ''):
                            m_px = self.calibration_from_metadata[index][0]
                            exposure = self.calibration_from_metadata[index][1]
                            step_sizes = self.calibration_from_metadata[index][2]
                            msd_diff_obj.micron_per_px = m_px

                            if (len(step_sizes) > 0):
                                msd_diff_obj.time_step = np.min(step_sizes)
                            else:
                                msd_diff_obj.time_step = exposure

                            #if we have varying step sizes, must filter tracks
                            if (len(np.unique(step_sizes)) > 0):   ## TODO fix this so it checks whether the largest diff. is > mindiff
                                track_data_df = filter_tracks(track_data_df, self.min_track_len_linfit, step_sizes, self.ts_resolution) #0.005)
                                #save the new, filtered CSVs
                                track_data_df.to_csv(self.results_dir + '/' + cur_file[:-4] + "_filtered.csv")

                                if (len(track_data_df) == 0):
                                    self.log.write("Note!  File '" + cur_dir + "/" + cur_file + "' contains 0 tracks after time step filtering.\n")
                                    self.log.flush()
                                    self.set_rows_to_none(index, '', 0, file_str, group_readable)
                                    continue

                    track_data=track_data_df.to_numpy()

                    # convert ROI area to microns, now that we have the correct scaling information
                    if (self.limit_to_ROIs and roi_area != ''):
                        roi_area = roi_area * msd_diff_obj.micron_per_px

                    #for this movie, calculate msd and diffusion for each track
                    msd_diff_obj.set_track_data(track_data)
                    msd_diff_obj.msd_all_tracks()
                    msd_diff_obj.fit_msd()

                    # rainbow tracks
                    if(self.make_rainbow_tracks):
                        if(cur_ax != None):
                            msd_diff_obj.save_tracks_to_img(cur_ax, len_cutoff='none', remove_tracks=False,
                                                        min_Deff=self.min_D_rainbow_tracks,
                                                        max_Deff=self.max_D_rainbow_tracks, lw=0.1)
                        if(cur_ax_ss != None):
                            msd_diff_obj.save_tracks_to_img_ss(cur_ax_ss, min_ss=self.min_ss_rainbow_tracks,
                                                           max_ss=self.max_ss_rainbow_tracks, lw=0.1)
                        if (cur_ax_roi != None):
                            msd_diff_obj.save_tracks_to_img_clr(cur_ax_roi, lw=0.1, color=roi_colors[count])
                            count += 1

                    if (len(msd_diff_obj.D_linfits) == 0):
                        self.log.write("Note!  File '" + cur_dir + "/" + cur_file +
                                       "' contains 0 tracks of minimum length for calculating Deff (" +
                                       str(msd_diff_obj.min_track_len_linfit) + ")\n")
                        self.log.flush()
                        self.set_rows_to_none(index, roi_area, len(msd_diff_obj.track_lengths), file_str,
                                              group_readable)
                        continue

                    D_median = np.median(msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col])
                    D_mean = np.mean(msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col])

                    D_linfits_filtered = msd_diff_obj.D_linfits[np.where(
                        (msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col] <= self.max_D_cutoff) &
                        (msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_D_col] >= self.min_D_cutoff))]

                    if(len(D_linfits_filtered)==0):
                        D_median_filt = np.nan
                        D_mean_filt = np.nan
                    else:
                        D_median_filt = np.median(D_linfits_filtered[:, msd_diff_obj.D_lin_D_col])
                        D_mean_filt = np.mean(D_linfits_filtered[:, msd_diff_obj.D_lin_D_col])

                    cur_data = msd_diff_obj.D_linfits[:,1:] #don't need track id column
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'id'] = index
                    for k in range(len(self.data_list.columns)):
                        self.data_list_with_results_full.iloc[full_data_i:full_data_i+len(cur_data),k+1]=self.data_list.loc[index][k]
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'group']=file_str
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'group_readable']=group_readable
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'D_median']=D_median
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'D_mean']=D_mean
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'D_median_filt']=D_median_filt
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'D_mean_filt']=D_mean_filt

                    # ADD: D, err, r_sq, rmse, track_len, D_track_len, for each track
                    next_col = len(full_results1.columns) + len(full_results2_cols1)
                    self.data_list_with_results_full.iloc[full_data_i:full_data_i+len(cur_data),next_col:next_col+len(cur_data[0])]=cur_data

                    self.data_list_with_results.at[index, 'D_median'] = D_median
                    self.data_list_with_results.at[index, 'D_mean'] = D_mean
                    self.data_list_with_results.at[index, 'D_median_filtered'] = D_median_filt
                    self.data_list_with_results.at[index, 'D_mean_filtered'] = D_mean_filt
                    self.data_list_with_results.at[index, 'num_tracks'] = len(msd_diff_obj.track_lengths)
                    self.data_list_with_results.at[index, 'num_tracks_D'] = len(msd_diff_obj.D_linfits)
                    self.data_list_with_results.at[index, 'area'] = roi_area
                    self.data_list_with_results.at[index, 'group'] = file_str
                    self.data_list_with_results.at[index, 'group_readable']=group_readable

                    if(save_per_file_data):
                        msd_diff_obj.save_msd_data(file_name=file_str + '_' + str(index) + "_MSD.txt")
                        msd_diff_obj.save_fit_data(file_name=file_str + '_' + str(index) + "_Dlin.txt")

                    full_data_i += len(cur_data)
                    self.log.write("Processed "+str(index) +" "+cur_file+" for MSD and Diffusion coeff.\n")
                    self.log.flush()

                # ran through all the rows with same csv/image file -- now save the rainbow tracks
                if(self.make_rainbow_tracks):
                    # save figure of lines plotted on bk_img
                    cur_fig.tight_layout()
                    out_file = os.path.split(self.valid_img_files[common_index])[1][:-4] + '_tracks_Deff.tif'
                    cur_fig.savefig(self.results_dir + '/' + out_file, dpi=500)  # 1000
                    plt.close(cur_fig)

                    cur_fig_ss.tight_layout()
                    out_file = os.path.split(self.valid_img_files[common_index])[1][:-4] + '_tracks_ss.tif'
                    cur_fig_ss.savefig(self.results_dir + '/' + out_file, dpi=500)  # 1000
                    plt.close(cur_fig_ss)

                    cur_fig_roi.tight_layout()
                    out_file = os.path.split(self.valid_img_files[common_index])[1][:-4] + '_tracks_roi.tif'
                    cur_fig_roi.savefig(self.results_dir + '/' + out_file, dpi=500)  # 1000
                    plt.close(cur_fig_roi)

        self.data_list_with_results.to_csv(self.results_dir + '/' + "summary.txt", sep='\t')

        if((self.get_calibration_from_metadata or self.limit_to_ROIs) and full_length > full_data_i):
            #need to remove the extra rows of the df b/c some tracks were filtered
            to_drop = range(full_data_i,full_length,1)
            self.data_list_with_results_full.drop(to_drop, axis=0, inplace=True)

        self.data_list_with_results_full.to_csv(self.results_dir + '/' + "all_data.txt", sep='\t')

