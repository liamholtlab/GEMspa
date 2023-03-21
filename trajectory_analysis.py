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
from skimage import io, draw
from read_roi import read_roi_zip
from read_roi import read_roi_file
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pylab as pylab

def get_roi_name_list_from_mask(mask):

    num_objects = len(np.unique(mask)) - 1
    if(num_objects > 1):
        # we have a pre-labeled mask - read in the labels, these are each separate objects
        labeled = mask
    else:
        # we only have a black and white image - let python do the labeling
        mask = mask > 0
        sq=[[1,1,1],[1,1,1],[1,1,1]]
        labeled, num_objects = ndimage.label(mask, structure=sq)

    names = list(np.unique(labeled))
    names = sorted(names)
    if(0 in names):
        names.remove(0)

    return (names, labeled)

def get_roi_name_list(rois):
    name_list=[]
    for key in rois.keys():
        roi = rois[key]
        if ((roi['type'] == 'polygon') or
                (roi['type'] == 'traced' and 'x' in roi and 'y' in roi) or
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
    out_of_bounds=False
    for pos_i, pos in enumerate(track_data):
        if (pos[0] != id):
            if ((len(np.unique(track_labels_full[prev_pos:pos_i, 1])) == 1) and (track_labels_full[pos_i - 1][1] != 0)):
                valid_id_list.append(id)
            id = pos[0]
            prev_pos = pos_i
        if(pos[3]>=len(mask_image) or pos[2] >= len(mask_image[1])):
            #IndexError
            label=-1
            out_of_bounds=True
        else:
            label = mask_image[int(pos[3])][int(pos[2])]
            track_labels_full[pos_i][0] = pos[0]
            track_labels_full[pos_i][1] = label

    # check final track
    pos_i = pos_i + 1
    if ((len(np.unique(track_labels_full[prev_pos:pos_i, 1])) == 1) and (track_labels_full[pos_i - 1][1] != 0)):
        valid_id_list.append(id)

    valid_id_list = np.asarray(valid_id_list)
    return (valid_id_list, out_of_bounds)

def make_mask_from_roi(rois, roi_name, img_shape):
    # loop through ROIs, only set interior of selected ROI to 1
    final_img = np.zeros(img_shape, dtype='uint8')
    poly_error=False
    for key in rois.keys():
        if(key == roi_name):
            roi = rois[key]
            if (roi['type'] == 'polygon' or
                    (roi['type'] == 'freehand' and 'x' in roi and 'y' in roi) or
                    (roi['type'] == 'traced'   and 'x' in roi and 'y' in roi)):
                col_coords = roi['x']
                row_coords = roi['y']
                rr, cc = draw.polygon(row_coords, col_coords, shape=img_shape)
                final_img[rr, cc] = 1
            elif (roi['type'] == 'rectangle'):
                rr, cc = draw.rectangle((roi['top'], roi['left']), extent=(roi['height'], roi['width']), shape=img_shape)
                rr=rr.astype('int')
                cc = cc.astype('int')
                final_img[rr, cc] = 1
            elif (roi['type'] == 'oval'):
                rr, cc = draw.ellipse(roi['top'] + roi['height'] / 2, roi['left'] + roi['width'] / 2, roi['height'] / 2, roi['width'] / 2, shape=img_shape)
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
        if (len(images.metadata['experiment']['loops']) > 0):
            step=images.metadata['experiment']['loops'][0]['sampling_interval']

            step = np.round(step, 0)
            step = step / 1000

            #plt.plot(range(1,len(steps)+1), steps)
            #plt.clf()
        else:
            #print(f"Error reading exposure from nd2 movie! {file_name}")
            #print(images.metadata['experiment']['loops'])
            step=0

        steps = images.timesteps[1:] - images.timesteps[:-1]
        # round steps to the nearest ms
        steps = np.round(steps, 0)

        #convert steps from ms to s
        steps=steps/1000

        microns_per_pixel=images.metadata['pixel_microns']
        return [microns_per_pixel,step,steps] #np.min(steps),steps]

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
                 make_rainbow_tracks=True, limit_to_ROIs=False, measure_track_intensities=False,
                 img_file_prefix='DNA_',
                 log_file=''):

        params = {'legend.fontsize': 'x-large',
                  #'figure.figsize': (15, 5),
                  'axes.labelsize': 'x-large',
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 'x-large',
                  'ytick.labelsize': 'x-large'}
        pylab.rcParams.update(params)

        self.get_calibration_from_metadata=use_movie_metadata
        self.uneven_time_steps = uneven_time_steps
        self.make_rainbow_tracks = make_rainbow_tracks
        self.limit_to_ROIs = limit_to_ROIs
        self.measure_track_intensities=measure_track_intensities

        self.intensity_radius=3

        self.output_plain_rainbow_tracks_time=False
        self.combine_rois=False
        self.radius_of_gyration=False
        self.NGP=False

        self.output_filtered_tracks=True

        self.calibration_from_metadata={}
        self.valid_img_files = {}
        self.valid_roi_files = {}
        self.valid_movie_files = {}

        self.min_ss_rainbow_tracks=0
        self.max_ss_rainbow_tracks=1
        self.min_D_rainbow_tracks=0
        self.max_D_rainbow_tracks=2

        self.line_width_rainbow_tracks = 0.1
        self.time_coded_rainbow_tracks_by_frame = False # color by track start, by default
        self.rainbow_tracks_DPI=300

        self.img_file_prefix = img_file_prefix

        self.time_step = 0.010  # time between frames, in seconds
        self.micron_per_px = 0.11
        self.ts_resolution=0.005

        self.fit_msd_with_error_term=False

        # track (trajectory) files columns - user can adjust as needed
        self.traj_id_col = 'Trajectory'
        self.traj_frame_col = 'Frame'
        self.traj_x_col = 'x'
        self.traj_y_col = 'y'

        # this is the min. track length for individual tracks
        # any tracks >= this length will be fitted to the equations:
        # MSD(T) = 4DT to get eff-D
        # MSD(T) = 4DT^a (a is the anomolous exponent, T is the time-lag)
        # (*ignore the _linfit suffix here*)
        self.min_track_len_linfit = 11

        # this is the number of T (time-lag) to use to fit
        self.tlag_cutoff_linfit = 10

        # when calculating ensemble average, include tracks >= this length
        self.min_track_len_ensemble = 11

        # this is the number of T (time-lag) to use to fit, when fitting the ensemble average MSD
        self.tlag_cutoff_ensemble = 10

        self.min_track_len_step_size = 3
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
                (col == self._movie_file_col_name and (self.get_calibration_from_metadata or self.uneven_time_steps)) or
                (col == self._img_file_col_name and (self.make_rainbow_tracks or self.limit_to_ROIs)) ):
                    self.log.write(f"Error! Required column {col} not found in input file {self.data_file}\n")
                    self.error_in_data_file=True
                    break

        # read in the time step information from the movie files
        if(not self.error_in_data_file):

            if(self.get_calibration_from_metadata or self.uneven_time_steps):
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
                            if(self.combine_rois):
                                roi_name_list=[255,]
                                labeled_img = img_mask > 0
                                labeled_img = labeled_img.astype('uint8')*255
                            else:
                                roi_name_list, labeled_img = get_roi_name_list_from_mask(img_mask)
                            to_save=os.path.split(valid_roi_file)[1][:-4]+'_LABELS.tif'
                            io.imsave(self.results_dir + '/' + to_save, labeled_img)
                        else:
                            if(valid_roi_file.endswith('.zip')):
                                rois = read_roi_zip(valid_roi_file)
                            else:
                                rois = read_roi_file(valid_roi_file)
                            roi_name_list = get_roi_name_list(rois)
                            if(len(roi_name_list)<len(rois)):
                                num_invalid=len(rois)-len(roi_name_list)
                                print(f"Error: {num_invalid} ROI(s) were not read from file (invalid ROI type): {valid_roi_file}")
                                self.log.write(f"Error: {num_invalid} ROI(s) were not read from file (invalid ROI type): {valid_roi_file}.\n")

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

            if(self.make_rainbow_tracks or self.limit_to_ROIs):
                for row in self.data_list.iterrows():
                    ind = row[1][self._index_col_name]
                    img_dir = row[1][self._img_file_col_name]
                    csv_file = row[1][self._file_col_name]

                    if(self.make_rainbow_tracks or self.valid_roi_files[ind].endswith(".zip") or self.valid_roi_files[ind].endswith(".roi")):
                        # if ROI is imagej .roi or .zip, then must check and load image file even if not making rainbow tracks
                        img_file = img_dir + "/" + self.img_file_prefix + csv_file[5:-4]  # Drop "Traj_" at beginning, add prefix, and drop ".csv" at end
                        if (not img_file.endswith(".tif")):
                            img_file = img_file + ".tif"
                        if (os.path.isfile(img_file)):
                            self.valid_img_files[ind] = img_file
                        else:
                            self.valid_img_files[ind] = ''
                            self.log.write(f"Error! Image file not found: {img_file} for rainbow tracks/ROIs.\n")

            if(self.get_calibration_from_metadata or self.uneven_time_steps or self.measure_track_intensities):
                for row in self.data_list.iterrows():
                    ind = row[1][self._index_col_name]
                    csv_file = row[1][self._file_col_name]

                    movie_dir = row[1][self._movie_file_col_name]
                    if(type(movie_dir) != type("")):
                        self.valid_movie_files[ind] = ''
                        self.log.write(f"Error! Movie directory not set in the input file.\n")
                    else:
                        movie_file = movie_dir + "/" + csv_file[5:-4]  # Drop "Traj_" at beginning and ".csv" at end

                        # check first for .tif, then nd2
                        if (not movie_file.endswith(".tif")):
                            movie_file = movie_file + ".tif"
                        if (os.path.isfile(movie_file)):
                            self.valid_movie_files[ind]=movie_file
                            if (self.get_calibration_from_metadata or self.uneven_time_steps):
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
                            self.valid_movie_files[ind] = '' # no reading from nd2 for now.... TODO
                            if(self.measure_track_intensities):
                                self.log.write(f"Error! TIF Movie file not found: {movie_file} for measuring track intensities.\n")
                            if (self.get_calibration_from_metadata or self.uneven_time_steps):
                                movie_file = movie_dir + "/" + csv_file[5:-4] + ".nd2"
                                if (os.path.isfile(movie_file)):
                                    ret_val = read_movie_metadata_nd2(movie_file)
                                else:
                                    ret_val = None
                                    self.calibration_from_metadata[ind] = ''
                                    self.log.write(f"Error! Movie file not found: {movie_file}.  Falling back to default settings.\n")
                        if (self.get_calibration_from_metadata or self.uneven_time_steps):
                            if (ret_val):
                                if (not self.uneven_time_steps):
                                    ret_val[2] = []  # do not use full time step info / could be messed up anyway in case of tif file

                                self.calibration_from_metadata[ind] = ret_val
                                self.log.write(f"Movie file {movie_file}: microns-per-pixel={ret_val[0]}, exposure={ret_val[1]}\n")

                                if (ret_val[1] == 0):
                                    self.log.write(f"Movie file {movie_file}: exposure (time step) not read from file.  Falling back to default settings.\n")
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
        self.log.write("Run parameters:\n")
        self.log.write(f"Rainbow tracks: {self.make_rainbow_tracks}\n")
        self.log.write(f"Filter with ROI file: {self.limit_to_ROIs}\n")
        self.log.write(f"Read calibration from metadata: {self.get_calibration_from_metadata}\n")
        self.log.write(f"Filter for uneven time steps: {self.uneven_time_steps}\n")
        self.log.write(f"Save filtered trajectory files: {self.output_filtered_tracks}\n")
        self.log.write(f"Min time step resolution: {self.ts_resolution}\n")
        self.log.write(f"Time between frames (s): {self.time_step}\n")
        self.log.write(f"Scale (microns per px): {self.micron_per_px}\n")
        self.log.write(f"Fit (linear) MSD curve with error term: {self.fit_msd_with_error_term}\n")

        self.log.write(f"Min track length: {self.min_track_len_linfit}\n")
        self.log.write(f"Min track length (ensemble average): {self.min_track_len_ensemble}\n")

        self.log.write(f"Number of tau for fitting: {self.tlag_cutoff_linfit}\n")
        self.log.write(f"Number of tau for fitting, ensemble average: {self.tlag_cutoff_ensemble}\n")

        self.log.write(f"Min track length (step size/angles): {self.min_track_len_step_size}\n")
        self.log.write(f"Max t-lag (step size/angles): {self.max_tlag_step_size}\n")

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
        msd_diff_obj.fit_msd_with_error_term = self.fit_msd_with_error_term

        msd_diff_obj.min_track_len_linfit = self.min_track_len_linfit

        # set this to same as linfit - needs to be the SAME or output won't work right
        # b/c we output both for each track so need to have same number of tracks
        msd_diff_obj.min_track_len_loglogfit = self.min_track_len_linfit

        msd_diff_obj.min_track_len_ensemble = self.min_track_len_ensemble

        msd_diff_obj.tlag_cutoff_linfit = self.tlag_cutoff_linfit
        msd_diff_obj.tlag_cutoff_loglogfit = self.tlag_cutoff_linfit # set this to same as linfit

        msd_diff_obj.tlag_cutoff_linfit_ensemble = self.tlag_cutoff_ensemble
        msd_diff_obj.tlag_cutoff_loglogfit_ensemble = self.tlag_cutoff_ensemble

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

    def plot_cos_theta_by_group(self, label_order=[], plot_labels=[], max_tlag=-1, random_angles=True, legend_fs=10):
        self.cos_theta_by_group = pd.read_csv(self.results_dir + "/cos_theta_by_group.txt", index_col=0, sep='\t',low_memory=False)

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.cos_theta_by_group['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.cos_theta_by_group['group_readable'])
            labels.sort()

        if (plot_labels):
            self.cos_theta_by_group['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.cos_theta_by_group['group_readable'] = np.where(self.cos_theta_by_group['group'] == str(labels[i]),
                    plot_label,
                    self.cos_theta_by_group['group_readable'])
            labels = plot_labels

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        num_groups=len(np.unique(self.cos_theta_by_group['group']))

        # for group_i,group in enumerate(np.unique(self.cos_theta_by_group['group_readable'])):
        #     cur_data = self.cos_theta_by_group.loc[self.cos_theta_by_group['group_readable']==group]
        #     ax.plot(cur_data['tlag'], cur_data['cos_theta-mean'], linewidth=1.5, color='black')
        #     ax.plot(cur_data['tlag'], cur_data['cos_theta-mean']-cur_data['cos_theta-sem'], linewidth=1.5,
        #             color='black')
        #     ax.plot(cur_data['tlag'], cur_data['cos_theta-mean']+cur_data['cos_theta-sem'], linewidth=1.5,
        #             color='black')
        #     ax.fill_between(cur_data['tlag'], cur_data['cos_theta-mean']-cur_data['cos_theta-sem'], cur_data['cos_theta-mean']+cur_data['cos_theta-sem'],
        #                     color=mpl.cm.get_cmap('tab10').colors[group_i], alpha=0.5)


        sns.lineplot(x="tlag", y='cos_theta-mean', hue="group_readable", data=self.cos_theta_by_group,
                      palette='tab10', legend=False, linewidth=1.5, ax=ax)
        sns.scatterplot(x="tlag", y='cos_theta-mean', hue="group_readable", data=self.cos_theta_by_group,
                         palette='tab10', ax=ax)

        if(random_angles):
            x_val=[]
            y_val=[]
            for tlag in np.unique(self.cos_theta_by_group['tlag']):
                num_angles=self.cos_theta_by_group.loc[self.cos_theta_by_group['tlag']==tlag]['num_angles'].min()
                if(not np.isnan(num_angles)):
                    num_angles=int(num_angles)
                    rand_angles=np.random.uniform(low=0, high=180, size=(num_angles,))
                    rand_angles = np.cos(np.deg2rad(rand_angles))
                    x_val.append(tlag)
                    y_val.append(np.mean(rand_angles))

            plt.plot(x_val, y_val, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Random')
            plt.scatter(x_val, y_val, color='black', alpha=0.6, s=18)

        ax.set_xlabel(r'$\tau$ $(s)$')
        ax.set_ylabel(r'$<cos \theta >$')

        ax.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))

        if(max_tlag > 0):
            ax.set_xlim(0, max_tlag)
        fig.tight_layout()
        fig.savefig(self.results_dir + '/cos_theta_by_group.pdf')

        fig.clf()
        plt.close(fig)

    def plot_msd_ensemble_by_group(self, label_order=[], plot_labels=[], legend_fs=10):
        self.results_by_group = pd.read_csv(self.results_dir + "/group_summary.txt", index_col=0, sep='\t',low_memory=False)

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.results_by_group['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.results_by_group['group_readable'])
            labels.sort()

        if (plot_labels):
            self.results_by_group['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.results_by_group['group_readable'] = np.where(self.results_by_group['group'] == str(labels[i]),
                    plot_label,
                    self.results_by_group['group_readable'])
            labels = plot_labels

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        plot_cols=[]
        x_vals=[]
        for col in self.results_by_group.columns:
            if(col.startswith("MSD_ave_")):
                plot_cols.append(col)
                x_vals.append(float(col.lstrip("MSD_ave_")))
        x_vals=np.asarray(x_vals)
        for row in self.results_by_group.iterrows():
            group = row[1]['group_readable']
            alpha=row[1]['ensemble_loglog_aexp']
            deff=row[1]['ensemble_D']
            y_vals = np.asarray(row[1][plot_cols]).astype(float)

            # loglog plot and anomolous exp
            #ax.axhline(np.log(row[1]['ensemble_loglog_K']*4)) #np.exp(popt[1])/4
            ax.scatter(x_vals, y_vals, label="{}, a≈{}".format(group, round(alpha, 2)))

        ax.set_xlabel(r'$\tau$ $(s)$')
        ax.set_ylabel(r'$MSD_{ens}$ ($\mu m^{2}$)')

        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)

        ax.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(self.results_dir + '/MSD_ensemble_by_group.pdf')
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        fig.clf()
        plt.close(fig)

    def plot_alpha_D_heatmap(self, label_order=[], plot_labels=[], min_pts=5, xlims=[], ylims=[]):

        self.data_list_with_results_full = pd.read_csv(self.results_dir + "/all_data.txt", index_col=0, sep='\t',low_memory=False)

        self.data_list_with_results_full['group'] = self.data_list_with_results_full['group'].astype('str')
        self.data_list_with_results_full['group_readable'] = self.data_list_with_results_full['group_readable'].astype('str')

        d_label = round(self.time_step * 1000) * self.tlag_cutoff_linfit

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.data_list_with_results_full['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.data_list_with_results_full['group_readable'])
            labels.sort()

        if (plot_labels):
            self.data_list_with_results_full['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.data_list_with_results_full['group_readable'] = np.where(self.data_list_with_results_full['group'] == str(labels[i]),
                    plot_label,
                    self.data_list_with_results_full['group_readable'])
            labels = plot_labels

        num_groups=0
        for group in np.unique(self.data_list_with_results_full['group_readable']):
            group_data = self.data_list_with_results_full[self.data_list_with_results_full['group_readable'] == group]

            if(len(group_data)>=min_pts):
                num_groups+=1

        if(num_groups == 0):
            print("Error: No data")
            return ()

        fig,axs = plt.subplots(1,num_groups,figsize=(num_groups * 10, 10))
        group_i=0

        xlim_min_all=0
        xlim_max_all=0
        ylim_min_all=0
        ylim_max_all=0
        for group in np.unique(self.data_list_with_results_full['group_readable']):
            group_data = self.data_list_with_results_full[self.data_list_with_results_full['group_readable'] == group]

            if(len(group_data)>=min_pts):
                if(num_groups > 1):
                    cur_ax=axs[group_i]
                else:
                    cur_ax=axs

                to_plot1 = np.log(group_data['D'])
                to_plot2 = group_data['aexp']

                sns.kdeplot(x=to_plot1, y=to_plot2, color='black', linewidths=0.1, ax=cur_ax)
                sns.kdeplot(x=to_plot1, y=to_plot2, cmap='Blues', fill=True, thresh=0, ax=cur_ax, ) #levels=10)

                cur_ax.scatter(to_plot1, to_plot2, linewidths=0, s=10, marker='.', color='black')

                cur_ax.axvline(-1, linewidth=0.1, linestyle='--', color='black', alpha=0.5)
                cur_ax.axhline(1, linewidth=0.1, linestyle='--', color='black', alpha=0.5)

                cur_ax.set_title(group)

                cur_ax.set_xlabel(f"$log(D_{{{d_label}ms}})$")
                cur_ax.set_ylabel(r'$\alpha$')

                (xlim_min, xlim_max) = cur_ax.get_xlim()
                (ylim_min, ylim_max) = cur_ax.get_ylim()

                if(group_i == 0):
                    xlim_min_all=xlim_min
                    ylim_min_all=ylim_min
                    xlim_max_all=xlim_max
                    ylim_max_all=ylim_max
                else:
                    if(xlim_min<xlim_min_all):
                        xlim_min_all=xlim_min
                    if (ylim_min < ylim_min_all):
                        ylim_min_all = ylim_min
                    if(xlim_max>xlim_max_all):
                        xlim_max_all=xlim_max
                    if (ylim_max > ylim_max_all):
                        ylim_max_all = ylim_max

                group_i+=1
            else:
                print(f"Did not make alpha-D heatmap for group {group} since there were less than {min_pts} trajectories.")

        if(num_groups > 1):
            for ax in axs:
                if(len(xlims) == 2):
                    ax.set_xlim(xlims)
                else:
                    ax.set_xlim((xlim_min_all, xlim_max_all))
                if (len(ylims) == 2):
                    ax.set_ylim(ylims)
                else:
                    ax.set_ylim((ylim_min_all, ylim_max_all))
        else:
            if (len(xlims) == 2):
                axs.set_xlim(xlims)
            else:
                axs.set_xlim((xlim_min_all, xlim_max_all))
            if (len(ylims) == 2):
                axs.set_ylim(ylims)
            else:
                axs.set_ylim((ylim_min_all, ylim_max_all))

        fig.savefig(self.results_dir + '/alpha_D_heatmaps.pdf')
        fig.clf()
        plt.close(fig)

    def plot_distribution_Deff(self, label_order=[], plot_labels=[], plot_type='gkde', bin_size=0.02 ,
                               min_pts=20, make_legend=False, plot_inside_group=False, logscale=False,
                               legend_fs=10):
        self.data_list_with_results_full = pd.read_csv(self.results_dir + "/all_data.txt", index_col=0, sep='\t', low_memory=False)

        self.data_list_with_results_full['group'] = self.data_list_with_results_full['group'].astype('str')
        self.data_list_with_results_full['group_readable'] = self.data_list_with_results_full['group_readable'].astype('str')

        d_label=round(self.time_step*1000)*self.tlag_cutoff_linfit

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.data_list_with_results_full['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.data_list_with_results_full['group_readable'])
            labels.sort()

        if (plot_labels):
            self.data_list_with_results_full['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.data_list_with_results_full['group_readable'] = np.where(self.data_list_with_results_full['group'] == str(labels[i]),
                    plot_label,
                    self.data_list_with_results_full['group_readable'])
            labels = plot_labels


        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)

        for group in np.unique(self.data_list_with_results_full['group_readable']):
            group_data = self.data_list_with_results_full[self.data_list_with_results_full['group_readable'] == group]

            if(logscale):
                obs_dist = np.log(group_data['D'])
            else:
                obs_dist = group_data['D']
            if(len(obs_dist)>min_pts):
                if (plot_inside_group):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)

                plotting_ind = np.arange(obs_dist.min() - bin_size, obs_dist.max() + bin_size, bin_size)
                if(plot_type == 'gkde'):
                    gkde = stats.gaussian_kde(obs_dist)
                    plotting_kdepdf = gkde.evaluate(plotting_ind)
                    ax3.plot(plotting_ind, plotting_kdepdf, label=group)
                else:
                    sns.histplot(x=obs_dist, bins=plotting_ind, element="step", fill=False, stat="probability",
                                 label=group, ax=ax3)

                #filter by file name
                if(plot_inside_group):
                    for id in np.unique(group_data["id"]):
                        cur_data = group_data[group_data["id"]==id]
                        if (logscale):
                            obs_dist = np.log(cur_data['D'])
                        else:
                            obs_dist = cur_data['D']

                        if(len(obs_dist)>min_pts):
                            plotting_ind = np.arange(obs_dist.min() - bin_size, obs_dist.max() + bin_size, bin_size)
                            if (self._roi_col_name in cur_data.columns):
                                roi_str = '-' + str(cur_data[self._roi_col_name].iloc[0])
                            else:
                                roi_str = ''
                            if (plot_type == 'gkde'):
                                gkde = stats.gaussian_kde(obs_dist)
                                plotting_kdepdf = gkde.evaluate(plotting_ind)
                                ax.plot(plotting_ind, plotting_kdepdf, label=str(cur_data[self._file_col_name].iloc[0])+roi_str)
                            else:
                                sns.histplot(x=obs_dist, bins=plotting_ind, element="step", fill=False,
                                             stat="probability", label=str(cur_data[self._file_col_name].iloc[0])+roi_str, alpha=0.6)
                    if (logscale):
                        ax.set_xlabel(f"$log(D_{{{d_label} ms}}$"+r" $(\mu m^{2}/s))$")
                    else:
                        ax.set_xlabel(f"$D_{{{d_label} ms}}$"+r" $(\mu m^{2}/s)$")

                    if (plot_type == 'gkde'):
                        ax.set_ylabel("Density")
                    else:
                        ax.set_ylabel("Fraction")

                    if(make_legend):
                        ax.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))

                    fig.tight_layout()
                    fig.savefig(self.results_dir + "/all_" + str(group) + "_Deff_"+plot_type+".pdf")
                    fig.clf()
                    plt.close(fig)

        #ax3.set_xlim(ax3.get_xlim()[0], 2)
        if(logscale):
            ax3.set_xlabel(f"$log(D_{{{d_label} ms}}$"+r" $(\mu m^{2}/s)$")
        else:
            ax3.set_xlabel(f"$D_{{{d_label} ms}}$"+r" $(\mu m^{2}/s)$")

        if (plot_type == 'gkde'):
            ax3.set_ylabel("Density")
        else:
            ax3.set_ylabel("Fraction")
        ax3.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))

        fig3.tight_layout()
        if(logscale):
            fig3.savefig(self.results_dir + '/combined_allgroups_logDeff_' + plot_type + '.pdf')
        else:
            fig3.savefig(self.results_dir + '/combined_allgroups_Deff_' + plot_type + '.pdf')

        fig3.clf()
        plt.close(fig3)

    def plot_distribution_alpha(self, label_order=[], plot_labels=[], plot_type='gkde', bin_size=0.02 ,
                                min_pts=20, make_legend=False, plot_inside_group=False,
                                legend_fs=10):
        self.data_list_with_results_full = pd.read_csv(self.results_dir + "/all_data.txt", index_col=0, sep='\t', low_memory=False)

        self.data_list_with_results_full['group'] = self.data_list_with_results_full['group'].astype('str')
        self.data_list_with_results_full['group_readable'] = self.data_list_with_results_full['group_readable'].astype('str')

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.data_list_with_results_full['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.data_list_with_results_full['group_readable'])
            labels.sort()

        if (plot_labels):
            self.data_list_with_results_full['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.data_list_with_results_full['group_readable'] = np.where(self.data_list_with_results_full['group'] == str(labels[i]),
                    plot_label,
                    self.data_list_with_results_full['group_readable'])
            labels = plot_labels

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)

        for group in np.unique(self.data_list_with_results_full['group_readable']):
            group_data = self.data_list_with_results_full[self.data_list_with_results_full['group_readable'] == group]

            obs_dist = group_data['aexp']
            if(len(obs_dist)>min_pts):
                if (plot_inside_group):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)

                plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)
                if(plot_type == 'gkde'):
                    gkde = stats.gaussian_kde(obs_dist)
                    plotting_kdepdf = gkde.evaluate(plotting_ind)
                    ax3.plot(plotting_ind, plotting_kdepdf, label=group)
                else:
                    sns.histplot(x=obs_dist, bins=plotting_ind, element="step", fill=False, stat="probability",
                                 ax=ax3, label=group)

                #filter by file name
                if(plot_inside_group):
                    for id in np.unique(group_data["id"]):
                        cur_data = group_data[group_data["id"]==id]
                        obs_dist=cur_data['aexp']
                        if(len(obs_dist)>min_pts):
                            plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)
                            if (self._roi_col_name in cur_data.columns):
                                roi_str = '-' + str(cur_data[self._roi_col_name].iloc[0])
                            else:
                                roi_str = ''
                            if (plot_type == 'gkde'):
                                gkde = stats.gaussian_kde(obs_dist)
                                plotting_kdepdf = gkde.evaluate(plotting_ind)
                                ax.plot(plotting_ind, plotting_kdepdf, label=str(cur_data[self._file_col_name].iloc[0])+roi_str)
                            else:
                                sns.histplot(x=obs_dist, bins=plotting_ind, element="step", fill=False,
                                             stat="probability", label=str(cur_data[self._file_col_name].iloc[0])+roi_str, alpha=0.6)

                    ax.set_xlabel(r"$\alpha$")
                    if (plot_type == 'gkde'):
                        ax.set_ylabel("Density")
                    else:
                        ax.set_ylabel("Fraction")
                    if(make_legend):
                        ax.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))

                    fig.tight_layout()
                    fig.savefig(self.results_dir + "/all_" + str(group) + "_Deff_"+plot_type+".pdf")
                    fig.clf()
                    plt.close(fig)

        ax3.set_xlabel(r"$\alpha$")
        if (plot_type == 'gkde'):
            ax3.set_ylabel("Density")
        else:
            ax3.set_ylabel("Fraction")
        ax3.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))

        fig3.tight_layout()
        fig3.savefig(self.results_dir + '/combined_allgroups_alpha_' + plot_type + '.pdf')
        fig3.clf()
        plt.close(fig3)

    def plot_distribution_step_sizes(self, tlags=[1,2,3], plot_type='gkde', bin_size=0.05, min_pts=20,
                                     make_legend=False, plot_inside_group=False, legend_fs=10):
        self.data_list_with_step_sizes_full = pd.read_csv(self.results_dir + "/all_data_step_sizes.txt", index_col=0, sep='\t', low_memory=False)

        self.data_list_with_step_sizes_full = self.data_list_with_step_sizes_full.replace('', np.NaN)
        self.data_list_with_step_sizes_full.dropna(axis=0, subset=['group'], inplace=True)

        start_pos = self.data_list_with_step_sizes_full.columns.get_loc("0")
        stop_pos=len(self.data_list_with_step_sizes_full.columns) - start_pos - 1

        tlags_ = []
        for tlag in tlags:
            if(tlag in np.unique(self.data_list_with_step_sizes_full['tlag'])):
                tlags_.append(tlag)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)

        for group in np.unique(self.data_list_with_step_sizes_full['group_readable']):
            group_data = self.data_list_with_step_sizes_full[self.data_list_with_step_sizes_full['group_readable'] == group]

            for tlag in tlags_:
                cur_tlag_data = group_data[group_data['tlag'] == tlag]
                obs_dist = np.asarray(cur_tlag_data.loc[:, "0":str(stop_pos)]).flatten()
                obs_dist = obs_dist[np.logical_not(np.isnan(obs_dist))]
                if(len(obs_dist)>min_pts):
                    plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)

                    if (plot_inside_group):
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)

                    if(plot_type == 'gkde'):
                        gkde = stats.gaussian_kde(obs_dist)
                        plotting_kdepdf = gkde.evaluate(plotting_ind)
                        ax3.plot(plotting_ind, plotting_kdepdf, label=group)
                    else:
                        ax3.hist(obs_dist, bins=plotting_ind, histtype="step", density=True, label=group)

                    if(plot_inside_group):
                        for id in cur_tlag_data['id'].unique():
                            cur_kde_data = cur_tlag_data[cur_tlag_data['id'] == id]
                            obs_dist=np.asarray(cur_kde_data.loc[:,"0":str(stop_pos)].iloc[0].dropna())
                            if(len(obs_dist)>min_pts):
                                plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)
                                if (self._roi_col_name in cur_kde_data.columns):
                                    roi_str = '-' + str(cur_kde_data[self._roi_col_name].iloc[0])
                                else:
                                    roi_str = ''
                                if(plot_type == 'gkde'):
                                    gkde = stats.gaussian_kde(obs_dist)
                                    plotting_kdepdf=gkde.evaluate(plotting_ind)
                                    ax.plot(plotting_ind, plotting_kdepdf, label=str(cur_kde_data[self._file_col_name].iloc[0])+roi_str)
                                else:
                                    ax.hist(obs_dist, bins=plotting_ind, histtype="step", density=True,
                                            label=str(cur_kde_data[self._file_col_name].iloc[0])+roi_str, alpha=0.6)


                        ax.set_xlabel("microns")
                        ax.set_ylabel("frequency")
                        if(make_legend):
                            ax.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))
                        fig.tight_layout()
                        fig.savefig(self.results_dir + '/all_tlag'+str(tlag)+'_'+str(group)+'_steps_'+plot_type+'.pdf')
                        fig.clf()
                        plt.close(fig)

        ax3.set_xlabel("microns")
        ax3.set_ylabel("frequency")
        ax3.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))
        fig3.tight_layout()
        fig3.savefig(self.results_dir + '/combined_tlag' + str(tlag) + '_allgroups_steps_' + plot_type + '.pdf')
        fig3.clf()
        plt.close(fig3)

    def plot_distribution_angles(self, tlags=[1, 2, 3], plot_type='gkde', bin_size=18, min_pts=20,
                                 make_legend=False, plot_inside_group=False, legend_fs=10):
        self.data_list_with_angles = pd.read_csv(self.results_dir + "/all_data_angles.txt", index_col=0, sep='\t', low_memory=False)

        self.data_list_with_angles = self.data_list_with_angles.replace('', np.NaN)
        self.data_list_with_angles.dropna(axis=0,subset=['group'],inplace=True)

        start_pos = self.data_list_with_angles.columns.get_loc("0")
        stop_pos = len(self.data_list_with_angles.columns) - start_pos - 1

        tlags_ = []
        for tlag in tlags:
            if (tlag in np.unique(self.data_list_with_angles['tlag'])):
                tlags_.append(tlag)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)

        for group in np.unique(self.data_list_with_angles['group_readable']):
            group_data = self.data_list_with_angles[self.data_list_with_angles['group_readable'] == group]

            for tlag in tlags_:
                cur_tlag_data = group_data[group_data['tlag'] == tlag]
                obs_dist = np.asarray(cur_tlag_data.loc[:, "0":str(stop_pos)]).flatten()
                obs_dist = obs_dist[np.logical_not(np.isnan(obs_dist))]
                if(len(obs_dist) > min_pts):
                    plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)

                    if(plot_inside_group):
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)

                    if (plot_type == 'gkde'):
                        gkde = stats.gaussian_kde(obs_dist)
                        plotting_kdepdf = gkde.evaluate(plotting_ind)
                        ax3.plot(plotting_ind, plotting_kdepdf, label=group)
                    else:
                        ax3.hist(obs_dist, bins=plotting_ind, histtype="step", density=True, label=group)

                    if (plot_inside_group):
                        for id in cur_tlag_data['id'].unique():
                            cur_kde_data = cur_tlag_data[cur_tlag_data['id'] == id]
                            obs_dist = np.asarray(cur_kde_data.loc[:, "0":str(stop_pos)].iloc[0].dropna())
                            if(len(obs_dist)>min_pts):
                                plotting_ind = np.arange(0, obs_dist.max() + bin_size, bin_size)
                                if(self._roi_col_name in cur_kde_data.columns):
                                    roi_str='-'+str(cur_kde_data[self._roi_col_name].iloc[0])
                                else:
                                    roi_str=''
                                if (plot_type == 'gkde'):
                                    gkde = stats.gaussian_kde(obs_dist)
                                    plotting_kdepdf = gkde.evaluate(plotting_ind)
                                    ax.plot(plotting_ind, plotting_kdepdf, label=str(cur_kde_data[self._file_col_name].iloc[0])+roi_str)
                                else:
                                    ax.hist(obs_dist, bins=plotting_ind, histtype="step", density=True,
                                            label=str(cur_kde_data[self._file_col_name].iloc[0])+roi_str, alpha=0.6)

                        ax.set_xlabel("angles (degrees)")
                        ax.set_ylabel("frequency")
                        if(make_legend):
                            ax.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))

                        fig.tight_layout()
                        fig.savefig(self.results_dir + '/all_tlag' + str(tlag) + '_' + str(group) + '_angles_' + plot_type + '.pdf')
                        fig.clf()
                        plt.close(fig)

        ax3.set_xlabel("angle (degrees)")
        ax3.set_ylabel("frequency")
        ax3.legend(loc='center left', fontsize=legend_fs, bbox_to_anchor=(1, 0.5))
        fig3.tight_layout()
        fig3.savefig(self.results_dir + '/combined_tlag' + str(tlag) + '_allgroups_angles_' + plot_type + '.pdf')
        fig3.clf()
        plt.close(fig3)

    def make_by_cell_plot(self, label, label_order, show_legend=True, roi_matching='order'):
        #group is on the x-axis
        #separate plot for each combination of labels from all other columns
        #data is read in from msd/diff in the saved files
        # if 'roi' is a column, also sepearte by 'roi': match ROIs by ORDER or NAME
        # roi_matching == 'order' OR 'name'

        self.data_list_with_results_full = pd.read_csv(self.results_dir + "/all_data.txt",index_col=0,sep='\t')

        if('roi' in self.data_list_with_results_full.columns):
            hue_var='cell_roi'
            self.data_list_with_results_full['cell_roi']=''
            if(roi_matching=='order'):
                self.data_list_with_results_full['img_group']=self.data_list_with_results_full['group'].astype('str') + '_' + self.data_list_with_results_full['cell'].astype('str')
                for img_group in np.unique(self.data_list_with_results_full['img_group']):
                    cur_data=self.data_list_with_results_full[self.data_list_with_results_full['img_group']==img_group]
                    min_id = np.unique(cur_data['id']).min()
                    self.data_list_with_results_full['cell_roi']=np.where(self.data_list_with_results_full['img_group']==img_group,
                        self.data_list_with_results_full['cell'].astype('str') + '_' + (self.data_list_with_results_full['id']-min_id+1).astype('str'),
                        self.data_list_with_results_full['cell_roi'])
            else:
                pass
        else:
            hue_var='cell'

        label_columns = self.label_columns[:]
        label_columns.remove(label)
        if (len(label_columns) > 0):
            grouped_data_list = self.data_list_with_results_full.groupby(label_columns) # self.data_list
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
                id_list = group_df.id
                cur_data=self.data_list_with_results_full[self.data_list_with_results_full.id.isin(id_list)].copy()

                cur_data[label + '_tonum'] = -1
                for order_i, order_label in enumerate(label_order):
                    cur_data[label + '_tonum'] = np.where(cur_data[label].astype('str') == str(order_label), order_i,
                                                          cur_data[label + '_tonum'])
                cur_data[hue_var] = "cell " + cur_data[hue_var].astype('str')

                sns.lineplot(x=label+'_tonum', y="D", data=cur_data, hue=hue_var, estimator=np.median, ci=None, ax=ax)

                if(show_legend):
                    self.sort_legend(ax)
                else:
                    ax.get_legend().remove()

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
            cur_data[hue_var]="cell "+cur_data[hue_var].astype('str')

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
                            print("Error could not locate cell name in file name.")
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

    def make_plot_combined_data(self, label_order=[], plot_labels=[], xlabel='', ylabel=''):

        self.data_list_with_results_full = pd.read_csv(self.results_dir + '/'+"all_data.txt",index_col=0,sep='\t')

        d_label = round(self.time_step * 1000) * self.tlag_cutoff_linfit

        if(len(self.data_list_with_results_full) == 0):
            print("Error: No data.")
            return ()

        self.data_list_with_results_full['group'] = self.data_list_with_results_full['group'].astype('str')
        self.data_list_with_results_full['group_readable'] = self.data_list_with_results_full['group_readable'].astype('str')

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.data_list_with_results_full['group_readable'])):
                    label_order_.append(l)
            labels = label_order_
        else:
            labels = np.unique(self.data_list_with_results_full['group_readable'])
            labels.sort()

        if (plot_labels):
            self.data_list_with_results_full['group_readable'] = ''
            for i, plot_label in enumerate(plot_labels):
                self.data_list_with_results_full['group_readable'] = np.where(self.data_list_with_results_full['group'] == str(labels[i]),
                    plot_label,
                    self.data_list_with_results_full['group_readable'])
            labels = plot_labels

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        sns.pointplot(x="group_readable", y='D', data=self.data_list_with_results_full, estimator=np.median, order=labels, #showfliers=False, #fliersize=0,
                        capsize=0.02, join=False, ax=ax, color='black', scale=.8, errwidth=1.5)

        ax.set(xlabel=xlabel)
        if (ylabel != ''):
            ax.set(ylabel=ylabel)
        else:
            ax.set(ylabel=f"$D_{{{d_label} ms}}$"+r" $(\mu m^{2}/s)$")

        plt.xticks(rotation='vertical')
        plt.tight_layout()
        fig.savefig(self.results_dir + '/summary_combined_D.pdf')
        fig.clf()

    def make_plot(self, label_order=[], plot_labels=[], xlabel='', ylabel='', clrs=[], min_pts=10, dot_size=4, points_plot="swarm"):
        # label_order should match the group labels (i.e. group/group_readable)
        # it is just imposing an order for plotting

        # additionally to label_order, new names can be given using plot_labels
        # these must corresponding the the labels in label_order, position by position

        self.data_list_with_results = pd.read_csv(self.results_dir + '/' + "summary.txt", sep='\t')

        self.data_list_with_results['group']=self.data_list_with_results['group'].astype('str')
        self.data_list_with_results['group_readable'] = self.data_list_with_results['group_readable'].astype('str')

        d_label=round(self.time_step*1000)*self.tlag_cutoff_linfit


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

        data_to_plot=self.data_list_with_results[self.data_list_with_results['num_tracks_D']>min_pts]
        for y_col in ['D_median','D_median_filtered']:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if(clrs != []):
                sns.boxplot(x="group_readable", y=y_col, data=data_to_plot, order=labels, fliersize=0, ax=ax,
                            palette=clrs)
            else:
                sns.boxplot(x="group_readable", y=y_col, data=data_to_plot, order=labels, fliersize=0, ax=ax)

            if(points_plot == "swarm"):
                sns.swarmplot(x="group_readable", y=y_col, data=data_to_plot, order=labels, color=".25", size=dot_size, ax=ax)
            elif(points_plot == "strip"):
                sns.stripplot(x="group_readable", y=y_col, data=data_to_plot, order=labels, color=".25", size=dot_size,
                              jitter=True, ax=ax)
            else:
                pass # no plotting of individual points

            ax.set(xlabel=xlabel)
            if (ylabel != ''):
                ax.set(ylabel = ylabel)
            else:
                ax.set(ylabel = f"$median$ $D_{{{d_label} ms}}$"+r" $(\mu m^{2}/s)$")
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            fig.savefig(self.results_dir + '/summary_'+y_col+'.pdf')
            fig.clf()

    def make_plot_intensity(self, label_order=[], plot_labels=[], dot_size=8):
        self.data_list_with_results_full = pd.read_csv(self.results_dir + '/' + "all_data.txt", index_col=0, sep='\t')

        self.data_list_with_results_full['group'] = self.data_list_with_results_full['group'].astype('str')
        self.data_list_with_results_full['group_readable'] = self.data_list_with_results_full['group_readable'].astype('str')

        d_label = round(self.time_step * 1000) * self.tlag_cutoff_linfit

        if (label_order):
            label_order_ = []
            for l in label_order:
                if (l in list(self.data_list_with_results_full['group_readable'])):
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

        sns.scatterplot(x="D",y="int_mean",data=self.data_list_with_results_full,
                        hue="group_readable", #"file name", #"group_readable",
                        s=dot_size,
                        #legend=False,
                        ax=ax)
        l = ax.legend()
        l.set_title('')
        #ax.set_xlim(ax.get_xlim()[0],4)
        plt.xlabel(f"$D_{{{d_label} ms}}$"+r" $(\mu m^{2}/s)$")
        plt.ylabel('Ave Px Intensity')
        plt.tight_layout()
        fig.savefig(self.results_dir + f"/D_vs_intensity.pdf")
        fig.clf()

        # for file_ in self.data_list_with_results_full['file name'].unique():
        #     cur_data = self.data_list_with_results_full.loc[self.data_list_with_results_full['file name']==file_]
        #     print(stats.pearsonr(cur_data['D'],cur_data['int_mean']))

    def make_plot_roi_area(self, label_order=[], plot_labels=[], xlabel='', ylabel='', clrs=[], min_pts=10, dot_size=4):
        #make plot of ROI Area vs. med(Deff)
        self.data_list_with_results = pd.read_csv(self.results_dir + '/' + "summary.txt", sep='\t')

        self.data_list_with_results['group'] = self.data_list_with_results['group'].astype('str')
        self.data_list_with_results['group_readable'] = self.data_list_with_results['group_readable'].astype('str')

        d_label = round(self.time_step * 1000) * self.tlag_cutoff_linfit

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

        data_to_plot = self.data_list_with_results[self.data_list_with_results['num_tracks_D'] > min_pts].copy()
        data_to_plot['group']=data_to_plot['group_readable']
        for y_col in ['D_median','D_median_filtered']:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            if (clrs != []):
                sns.scatterplot(x="area", y=y_col, data=data_to_plot, hue='group',
                                ax=ax, palette=clrs, size=dot_size, hue_order=labels)
            else:
                sns.scatterplot(x="area", y=y_col, data=data_to_plot, hue='group', ax=ax, s=dot_size,
                                hue_order=labels)

            if (xlabel != ''):
                ax.set(xlabel=xlabel)
            else:
                ax.set(xlabel="roi area (microns)")

            if (ylabel != ''):
                ax.set(ylabel=ylabel)
            else:
                ax.set(ylabel=f"$median$ $D_{{{d_label} ms}}$"+r" $(\mu m^{2}/s)$")

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
        track_data_df = track_data_df.dropna()
        return track_data_df

    def filter_ROI(self, index, roi_name, df):
        roi_area=0
        if (index in self.valid_roi_files and self.valid_roi_files[index] != ''):
            roi_file = self.valid_roi_files[index]
            err = False
            if (roi_file.endswith('roi')):
                rois = read_roi_file(roi_file)
            elif (roi_file.endswith('zip')):
                rois = read_roi_zip(roi_file)
            elif (roi_file.endswith('_mask.tif')):
                mask = io.imread(roi_file)

                if (self.combine_rois):
                    mask = mask > 0
                    mask = mask.astype('uint8') * 255  # the label will be 255, this was set prior
                    num_objects = 1
                else:
                    num_objects = len(np.unique(mask)) - 1
                    if (num_objects > 1):
                        # we have a pre-labeled mask - read in the labels, these are each separate objects
                        labeled = mask
                    else:
                        # we only have a black and white image - let python do the labeling
                        mask = mask > 0
                        sq = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                        labeled, num_objects = ndimage.label(mask, structure=sq)

                    # make a mask with only the selected ROI set to "True"
                    mask = (labeled == roi_name)
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

            # limit the tracks to the ROI - returns track ids that are fully within mask
            if (not err):
                valid_id_list,err = limit_tracks_given_mask(mask, df.to_numpy())
                if(err):
                    self.log.write(f"Error!  Track positions are outside of TIF image bounds: {self.valid_img_files[index]}\n")
                    self.log.write(f"Check that the correct trajectory CSV file is associated with this ROI/TIF file.\n")
                    self.log.flush()
                df = df[df['Trajectory'].isin(valid_id_list)]
                roi_area=np.sum(mask.flatten())
        else:
            # no action here - error will already have been printed by class init function #
            pass
        return (df, roi_area)

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

        # make a dataframe containing the NGP for each tlag
        if(self.NGP):
            self.data_list_with_NGP = self.data_list.copy()
            for tlag_i in range(1, self.max_tlag_step_size + 1, 1):
                self.data_list_with_NGP['NGP_' + str(tlag_i)] = 0.0
            self.data_list_with_NGP['area'] = ''
            self.data_list_with_NGP['group'] = ''
            self.data_list_with_NGP['group_readable'] = ''

        # make a full dataframe containing all data - angles
        nrows =  self.max_tlag_step_size * len(self.data_list)
        ncols = len(self.data_list.columns) + max_tlag1_angle_num_steps
        colnames = list(self.data_list.columns)
        endpos = len(colnames)
        rest_cols = np.asarray(range(max_tlag1_angle_num_steps))
        rest_cols = rest_cols.astype('str')
        colnames.extend(rest_cols)
        self.data_list_with_angles = pd.DataFrame(np.empty((nrows, ncols), dtype=np.str), columns=colnames)
        self.data_list_with_angles.insert(loc=0, column='id', value=0)
        self.data_list_with_angles.insert(loc=endpos + 1, column='group', value='')
        self.data_list_with_angles.insert(loc=endpos + 2, column='group_readable', value='')
        self.data_list_with_angles.insert(loc=endpos + 3, column='tlag', value=0)
        self.data_list_with_angles['tlag'] = np.tile(range(1, self.max_tlag_step_size+1), len(self.data_list))

        #make summary data frame for the cosine theta of all angles per group
        colnames = ['group', 'group_readable', 'tlag',
                    'cos_theta-median', 'cos_theta-mean',
                    'cos_theta-std', 'cos_theta-sem', 'num_angles']

        self.cos_theta_by_group = pd.DataFrame(np.empty((len(group_list) * self.max_tlag_step_size, len(colnames)), dtype=np.str), columns=colnames)

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
                if(self.get_calibration_from_metadata or self.uneven_time_steps):

                    if(index in self.calibration_from_metadata and self.calibration_from_metadata[index] != ''):
                        m_px=self.calibration_from_metadata[index][0]
                        exposure=self.calibration_from_metadata[index][1]
                        step_sizes=self.calibration_from_metadata[index][2]
                        msd_diff_obj.micron_per_px=m_px
                        if(len(step_sizes)>0):
                            msd_diff_obj.time_step=np.min(step_sizes)
                        else:
                            if(exposure > 0):
                                msd_diff_obj.time_step=exposure

                        # if we have varying step sizes, must filter tracks
                        if(self.uneven_time_steps):
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

                if (self.NGP):
                    msd_diff_obj.non_gaussian_1d()

                cur_data_step_sizes = msd_diff_obj.step_sizes
                cur_data_angles = msd_diff_obj.angles
                ss_len=len(cur_data_step_sizes)
                a_len=len(cur_data_angles)

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
                    ss_data=msd_diff_obj.step_sizes[tlag_i-1][np.logical_not(np.isnan(msd_diff_obj.step_sizes[tlag_i-1]))]
                    if(len(ss_data)>0):
                        self.data_list_with_step_sizes.at[index,'step_size_' + str(tlag_i) + '_median'] = np.median(ss_data)
                        self.data_list_with_step_sizes.at[index,'step_size_' + str(tlag_i) + '_mean'] = np.mean(ss_data)
                    else:
                        self.data_list_with_step_sizes.at[index, 'step_size_' + str(tlag_i) + '_median']=np.nan
                        self.data_list_with_step_sizes.at[index, 'step_size_' + str(tlag_i) + '_mean']=np.nan
                self.data_list_with_step_sizes.at[index, 'area'] = roi_area
                self.data_list_with_step_sizes.at[index, 'group'] = file_str
                self.data_list_with_step_sizes.at[index, 'group_readable'] = group_readable

                #fill NGP data
                if(self.NGP):
                    for tlag_i in range(1,self.max_tlag_step_size+1,1):
                        self.data_list_with_NGP.at[index,'NGP_' + str(tlag_i)] = msd_diff_obj.ngp[tlag_i-1]
                    self.data_list_with_NGP.at[index, 'area'] = roi_area
                    self.data_list_with_NGP.at[index, 'group'] = file_str
                    self.data_list_with_NGP.at[index, 'group_readable'] = group_readable

                #fill angle data
                self.data_list_with_angles.loc[full_data_a_i:full_data_a_i+a_len-1,'id']=index
                for k in range(len(self.data_list.columns)):
                    self.data_list_with_angles.iloc[full_data_a_i:full_data_a_i+a_len,k+1]=self.data_list.loc[index][k]
                self.data_list_with_angles.loc[full_data_a_i:full_data_a_i+a_len-1,'group']=file_str
                self.data_list_with_angles.loc[full_data_a_i:full_data_a_i+a_len-1,'group_readable']=group_readable
                self.data_list_with_angles.loc[full_data_a_i:full_data_a_i+a_len-1,"0":str(len(msd_diff_obj.angles[0])-1)]=msd_diff_obj.angles

                if (save_per_file_data):
                    msd_diff_obj.save_step_sizes(file_name=file_str + '_' + str(index) + "_step_sizes.txt")
                    msd_diff_obj.save_angles(file_name=file_str + '_' + str(index) + "_angles.txt")

                full_data_ss_i += len(cur_data_step_sizes)
                full_data_a_i +=  len(cur_data_angles)

                self.log.write("Processed " + str(index) +" "+cur_file + "for step sizes and angles.\n")
                self.log.flush()

            # finished with all group calculations for current group, fill group data for cosine theta data frame
            cur_group_data = self.data_list_with_angles[self.data_list_with_angles['group'] == file_str]
            for tlag in range(1, self.max_tlag_step_size+1):
                ind=group_i*self.max_tlag_step_size
                self.cos_theta_by_group.at[ind+(tlag-1), 'group'] = file_str
                self.cos_theta_by_group.at[ind+(tlag-1), 'group_readable'] = group_readable
                self.cos_theta_by_group.at[ind+(tlag-1), 'tlag'] = str(tlag*self.time_step)


                cur_tlag_data = np.asarray(cur_group_data[cur_group_data['tlag'] == tlag].loc[:, "0":].replace('', np.NaN)).flatten().astype('float64')
                cur_tlag_data = cur_tlag_data[np.logical_not(np.isnan(cur_tlag_data))]
                if (len(cur_tlag_data) > 1):
                    # take the cosine of theta
                    angles = np.cos(np.deg2rad(cur_tlag_data))
                    self.cos_theta_by_group.at[ind+(tlag-1), 'cos_theta-mean'] = np.mean(angles)
                    self.cos_theta_by_group.at[ind+(tlag-1), 'cos_theta-median'] = np.median(angles)
                    self.cos_theta_by_group.at[ind+(tlag-1), 'cos_theta-std'] = np.std(angles)
                    self.cos_theta_by_group.at[ind+(tlag-1), 'cos_theta-sem'] = np.std(angles) / np.sqrt(len(angles))
                    self.cos_theta_by_group.at[ind+(tlag-1), 'num_angles'] = len(angles)

        self.data_list_with_step_sizes.to_csv(self.results_dir + '/' + "summary_step_sizes.txt", sep='\t')
        if(self.NGP):
            self.data_list_with_NGP.to_csv(self.results_dir + '/' + "NGP.txt", sep='\t')

        if (self.uneven_time_steps or self.limit_to_ROIs):
            self.data_list_with_step_sizes_full.replace('', np.NaN, inplace=True)
            self.data_list_with_step_sizes_full.dropna(axis=1,how='all',inplace=True)
            self.data_list_with_step_sizes_full.dropna(axis=0,subset=['id',],inplace=True)
            self.data_list_with_step_sizes_full.dropna(axis=0, subset=['group',], inplace=True)

            self.data_list_with_angles.replace('', np.NaN, inplace=True)
            self.data_list_with_angles.dropna(axis=1, how='all', inplace=True)
            self.data_list_with_angles.dropna(axis=0, subset=['id', ], inplace=True)
            self.data_list_with_angles.dropna(axis=0, subset=['group',], inplace=True)

        self.data_list_with_step_sizes_full.to_csv(self.results_dir + '/' + "all_data_step_sizes.txt", sep='\t')
        self.data_list_with_angles.to_csv(self.results_dir + '/' + "all_data_angles.txt", sep='\t')
        self.cos_theta_by_group.to_csv(self.results_dir + '/' + "cos_theta_by_group.txt", sep='\t')

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
        max_num_tracks=0
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

                num_tracks=len(np.unique(track_data[:,0]))
                if(num_tracks > max_num_tracks):
                    max_num_tracks=num_tracks

        # make a full dataframe containing all data, including all D values for all tracks etc.
        # NOTE: if track data must be filtered b/c of uneven time steps (done below), then this array may not be filled completely
        full_results1 = pd.DataFrame(np.empty((full_length, len(self.data_list.columns)), dtype=np.str),
                                     columns=list(self.data_list.columns))
        full_results1.insert(loc=0, column='id', value=0)
        full_results1['group']=''
        full_results1['group_readable']=''
        full_results2_cols1=['D_median','D_mean','D_median_filt','D_mean_filt','avg_velocity','int_mean','int_std']
        full_results2_cols2=['Trajectory','D','E','err','r_sq','rmse','track_len','D_max_tlag']
        full_results2_cols3=['K','aexp','aexp_r_sq','aexp_rmse']

        cols_len=len(full_results2_cols1) + len(full_results2_cols2) + len(full_results2_cols3)
        full_results2 = pd.DataFrame(np.zeros((full_length, cols_len)), columns=full_results2_cols1+full_results2_cols2+full_results2_cols3)
        self.data_list_with_results_full = pd.concat([full_results1,full_results2], axis=1)

        # make a dataframe for the radius of gyration, avg velocity, and track length.  for each, the distribution will be output in a row
        if(self.radius_of_gyration):
            colnames = list(self.data_list.columns)
            endpos = len(colnames)
            rest_cols = np.asarray(range(max_num_tracks))
            rest_cols = rest_cols.astype('str')
            colnames.extend(rest_cols)
            self.data_list_with_Rg = pd.DataFrame(np.empty((len(self.data_list)*3, len(self.data_list.columns) + max_num_tracks),
                                                           dtype=np.str), columns=colnames)
            self.data_list_with_Rg.insert(loc=0, column='id', value='')
            self.data_list_with_Rg.insert(loc=endpos + 1, column='group', value='')
            self.data_list_with_Rg.insert(loc=endpos + 2, column='group_readable', value='')
            self.data_list_with_Rg.insert(loc=endpos + 3, column='data', value='')

        # make a dataframe containing only median and mean D values for each movie
        self.data_list_with_results = self.data_list.copy()
        self.data_list_with_results['D_median']=0.0
        self.data_list_with_results['D_mean']=0.0
        self.data_list_with_results['D_median_filtered'] = 0.0
        self.data_list_with_results['D_mean_filtered'] = 0.0
        self.data_list_with_results['num_tracks'] = 0
        self.data_list_with_results['num_tracks_D'] = 0
        self.data_list_with_results['area'] = ''
        self.data_list_with_results['ensemble_D'] = ''
        self.data_list_with_results['ensemble_E'] = ''
        self.data_list_with_results['ensemble_r_sq'] = ''
        self.data_list_with_results['ensemble_loglog_K'] = ''
        self.data_list_with_results['ensemble_loglog_aexp'] = ''
        self.data_list_with_results['ensemble_loglog_r_sq'] = ''
        self.data_list_with_results['group']=''
        self.data_list_with_results['group_readable'] = ''

        # make a dataframe containing summary values by group
        colnames=['group', 'group_readable',
                  'ensemble_D', 'ensemble_E', 'ensemble_r_sq',
                  'ensemble_loglog_K', 'ensemble_loglog_aexp', 'ensemble_loglog_r_sq',
                  'D_group_median', 'D_group_mean', 'D_group_std', 'D_group_sem',
                  'aexp_group_median', 'aexp_group_mean', 'aexp_group_std', 'aexp_group_sem',
                  'group_num_tracks', 'ensemble_num_tracks']
        self.results_by_group = pd.DataFrame(np.empty((len(group_list), len(colnames)), dtype=np.str), columns=colnames)

        msd_diff_obj = self.make_msd_diff_object()

        full_data_i=0
        Rg_i=0

        for group_i,group in enumerate(group_list):
            track_data_df_all = pd.DataFrame()  # this will hold all trajectories for the group

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
                cur_fig_time = None
                cur_fig_time_plain = None
                #cur_fig_roi=None
                cur_ax=None
                cur_ax_ss=None
                cur_ax_time = None
                cur_ax_time_plain = None
                #cur_ax_roi=None
                # set up the figure axes for making rainbow tracks
                if (self.make_rainbow_tracks):
                    common_index=files_group.index[0] # for the same file, the image dir is the same for all rows, just take first
                    rt_image_found=True
                    if (common_index in self.valid_img_files and self.valid_img_files[common_index] != ''):
                        bk_img = io.imread(self.valid_img_files[common_index])

                        # plot figure to draw tracks by Deff with image in background
                        cur_fig = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=self.rainbow_tracks_DPI)
                        cur_ax = cur_fig.add_subplot(1, 1, 1)
                        cur_ax.axis("off")
                        cur_ax.imshow(bk_img, cmap="gray")

                        # plot figure to draw tracks by ss with image in background
                        cur_fig_ss = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=self.rainbow_tracks_DPI)
                        cur_ax_ss = cur_fig_ss.add_subplot(1, 1, 1)
                        cur_ax_ss.axis("off")
                        cur_ax_ss.imshow(bk_img, cmap="gray")

                        # plot figure to draw tracks by time with image in background
                        cur_fig_time = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=self.rainbow_tracks_DPI)
                        cur_ax_time = cur_fig_time.add_subplot(1, 1, 1)
                        cur_ax_time.axis("off")
                        cur_ax_time.imshow(bk_img, cmap="gray")

                        # plot figure to draw tracks by time with NO image in background, as pdf
                        # x and y needs reversing
                        if(self.output_plain_rainbow_tracks_time):
                            cur_fig_time_plain = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=self.rainbow_tracks_DPI)
                            cur_ax_time_plain = cur_fig_time_plain.add_subplot(1, 1, 1)
                            cur_ax_time_plain.axis("off")

                        # plot figure to draw tracks by roi with image in background
                        # cur_fig_roi = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=100)
                        # cur_ax_roi = cur_fig_roi.add_subplot(1, 1, 1)
                        # cur_ax_roi.axis("off")
                        # cur_ax_roi.imshow(bk_img, cmap="gray")
                    else:
                        rt_image_found=False

                #count=0
                #roi_cmap = cm.get_cmap('jet', len(files_group))
                #roi_colors = roi_cmap(range(1, len(files_group)+1))
                for index,data in files_group.iterrows():
                    cur_dir=data[self._dir_col_name]
                    cur_file=data[self._file_col_name]

                    print(cur_file, data[self._roi_col_name])

                    if(self.measure_track_intensities and index in self.valid_movie_files and self.valid_movie_files[index]!=''):
                        cur_tif_movie = io.imread(self.valid_movie_files[index])

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
                    if (self.get_calibration_from_metadata or self.uneven_time_steps):
                        if(index in self.calibration_from_metadata and self.calibration_from_metadata[index] != ''):
                            m_px = self.calibration_from_metadata[index][0]
                            exposure = self.calibration_from_metadata[index][1]
                            step_sizes = self.calibration_from_metadata[index][2]
                            msd_diff_obj.micron_per_px = m_px

                            if (len(step_sizes) > 0):
                                msd_diff_obj.time_step = np.min(step_sizes)
                            else:
                                if (exposure > 0):
                                    msd_diff_obj.time_step = exposure

                            #if we have varying step sizes, must filter tracks
                            if (self.uneven_time_steps):
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
                    if(self.output_filtered_tracks and (self.limit_to_ROIs or self.uneven_time_steps)):
                        #track data was filtered
                        #  output the track data to a file - label with csv file name and roi if present
                        if (self.limit_to_ROIs and data[self._roi_col_name]):
                            file_name_to_save = cur_file[:-4] + '_' + str(data[self._roi_col_name]) + "-filtered.csv"
                        else:
                            file_name_to_save = cur_file[:-4] + "-filtered.csv"

                        track_data_new_df=pd.DataFrame(track_data, columns=['Trajectory','Frame','x','y'])
                        track_data_new_df.sort_index(inplace=True) # I dont think this is needed...
                        track_data_new_df.to_csv(self.results_dir + '/' + file_name_to_save, index=False)

                    msd_diff_obj.set_track_data(track_data)
                    msd_diff_obj.msd_all_tracks()
                    msd_diff_obj.fit_msd()
                    msd_diff_obj.fit_msd_alpha()
                    msd_diff_obj.calculate_ensemble_average()
                    msd_diff_obj.fit_msd_ensemble()
                    msd_diff_obj.fit_msd_ensemble_alpha()
                    if(self.radius_of_gyration):
                        msd_diff_obj.radius_of_gyration()
                    msd_diff_obj.average_velocity()

                    if(self.measure_track_intensities and index in self.valid_movie_files and self.valid_movie_files[index]!=''):
                        # remove tracks by length
                        msd_diff_obj.fill_track_intensities(cur_tif_movie, self.intensity_radius)

                    # accumulate the trajectories into one large list for the entire group
                    # in order to calculate the full time+ensemble average MSD for the group and fit
                    if (len(track_data_df_all) == 0):
                        next_start = 0
                    else:
                        next_start = np.max(track_data_df_all['Trajectory'])
                    track_data_df['Trajectory'] = track_data_df['Trajectory'] + next_start
                    track_data_df_all = pd.concat([track_data_df_all, track_data_df], axis=0, ignore_index=True)

                    # rainbow tracks
                    if(self.make_rainbow_tracks): # lw is line width in the matplotlib function - convert from pixel size - using 96 PPI (???)
                        if(cur_ax != None):
                            msd_diff_obj.save_tracks_to_img(cur_ax, len_cutoff='none', remove_tracks=False,
                                                        min_Deff=self.min_D_rainbow_tracks,
                                                        max_Deff=self.max_D_rainbow_tracks, lw=self.line_width_rainbow_tracks)
                        if(cur_ax_ss != None):
                            msd_diff_obj.save_tracks_to_img_ss(cur_ax_ss, min_ss=self.min_ss_rainbow_tracks,
                                                           max_ss=self.max_ss_rainbow_tracks, lw=self.line_width_rainbow_tracks)

                        if (cur_ax_time != None):
                            if(self.time_coded_rainbow_tracks_by_frame):
                                msd_diff_obj.save_tracks_to_img_time(cur_ax_time, relative_to='frame', lw=self.line_width_rainbow_tracks)
                            else:
                                msd_diff_obj.save_tracks_to_img_time(cur_ax_time, relative_to='track', lw=self.line_width_rainbow_tracks)

                        if (self.output_plain_rainbow_tracks_time and cur_ax_time_plain != None):
                            if(self.time_coded_rainbow_tracks_by_frame):
                                msd_diff_obj.save_tracks_to_img_time(cur_ax_time_plain, relative_to='frame', lw=self.line_width_rainbow_tracks, reverse_coords=True,
                                                                     xlim=bk_img.shape[1], ylim=bk_img.shape[0])
                            else:
                                msd_diff_obj.save_tracks_to_img_time(cur_ax_time_plain, relative_to='track', lw=self.line_width_rainbow_tracks, reverse_coords=True,
                                                                     xlim=bk_img.shape[1], ylim=bk_img.shape[0])

                        # if (cur_ax_roi != None):
                        #     msd_diff_obj.save_tracks_to_img_clr(cur_ax_roi, lw=0.1, color=roi_colors[count])
                        #     count += 1

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

                    # Fill data array with eff-D
                    cur_data = msd_diff_obj.D_linfits[:,:]
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'id'] = index
                    for k in range(len(self.data_list.columns)):
                        self.data_list_with_results_full.iloc[full_data_i:full_data_i+len(cur_data),k+1]=self.data_list.loc[index][k]
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'group']=file_str
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'group_readable']=group_readable
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'D_median']=D_median
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'D_mean']=D_mean
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'D_median_filt']=D_median_filt
                    self.data_list_with_results_full.loc[full_data_i:full_data_i+len(cur_data)-1,'D_mean_filt']=D_mean_filt

                    # output avg_velocity but only for the tracks were used in Deff calculation
                    self.data_list_with_results_full.loc[full_data_i:full_data_i + len(cur_data)-1, 'avg_velocity'] = (
                        msd_diff_obj.avg_velocity[np.isin(msd_diff_obj.avg_velocity[:,0],
                                                          msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_id_col])][:,1])

                    if(self.measure_track_intensities and index in self.valid_movie_files and self.valid_movie_files[index]!=''):
                        self.data_list_with_results_full.loc[full_data_i:full_data_i + len(cur_data) - 1,'int_mean'] = (
                            msd_diff_obj.track_intensities[np.isin(msd_diff_obj.track_intensities[:, 0],
                                                                   msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_id_col])][:, 1])
                        self.data_list_with_results_full.loc[full_data_i:full_data_i + len(cur_data) - 1,'int_std'] = (
                            msd_diff_obj.track_intensities[np.isin(msd_diff_obj.track_intensities[:, 0],
                                                                   msd_diff_obj.D_linfits[:, msd_diff_obj.D_lin_id_col])][:, 2])

                    next_col = len(full_results1.columns) + len(full_results2_cols1)
                    self.data_list_with_results_full.iloc[full_data_i:full_data_i+len(cur_data), next_col:next_col+len(cur_data[0])] = cur_data

                    # add in the alpha information for each track
                    next_col = len(full_results1.columns) + len(full_results2_cols1) + len(full_results2_cols2)
                    self.data_list_with_results_full.iloc[full_data_i:full_data_i + len(cur_data),
                        next_col:next_col + 1] = msd_diff_obj.D_loglogfits[:,msd_diff_obj.D_loglog_K_col]
                    self.data_list_with_results_full.iloc[full_data_i:full_data_i + len(cur_data),
                        next_col+1:next_col + 2] = msd_diff_obj.D_loglogfits[:, msd_diff_obj.D_loglog_alpha_col]
                    self.data_list_with_results_full.iloc[full_data_i:full_data_i + len(cur_data),
                        next_col+2:next_col + 3] = msd_diff_obj.D_loglogfits[:, msd_diff_obj.D_loglog_rsq_col]
                    self.data_list_with_results_full.iloc[full_data_i:full_data_i + len(cur_data),
                        next_col+3:next_col + 4] = msd_diff_obj.D_loglogfits[:, msd_diff_obj.D_loglog_rmse_col]

                    # fill Rg data and track len data
                    if (self.radius_of_gyration):
                        self.data_list_with_Rg.loc[Rg_i:Rg_i+2, 'id'] = index
                        for k in range(len(self.data_list.columns)):
                            self.data_list_with_Rg.iloc[Rg_i:Rg_i+3, k + 1] = self.data_list.loc[index][k]
                        self.data_list_with_Rg.loc[Rg_i:Rg_i+2,'group'] = file_str
                        self.data_list_with_Rg.loc[Rg_i:Rg_i+2,'group_readable'] = group_readable
                        self.data_list_with_Rg.loc[Rg_i,'data'] = 'track_len'
                        self.data_list_with_Rg.loc[Rg_i+1,'data'] = 'Rg'
                        self.data_list_with_Rg.loc[Rg_i+2, 'data'] = 'avg_velocity'
                        self.data_list_with_Rg.loc[Rg_i, "0":str(len(msd_diff_obj.r_of_g) - 1)] = msd_diff_obj.track_lengths[:,1]
                        self.data_list_with_Rg.loc[Rg_i+1,"0":str(len(msd_diff_obj.r_of_g) - 1)] = msd_diff_obj.r_of_g[:,1]
                        self.data_list_with_Rg.loc[Rg_i+2,"0":str(len(msd_diff_obj.r_of_g) - 1)] = msd_diff_obj.avg_velocity[:,1]

                    # fill summary data array
                    self.data_list_with_results.at[index, 'D_median'] = D_median
                    self.data_list_with_results.at[index, 'D_mean'] = D_mean
                    self.data_list_with_results.at[index, 'D_median_filtered'] = D_median_filt
                    self.data_list_with_results.at[index, 'D_mean_filtered'] = D_mean_filt
                    self.data_list_with_results.at[index, 'num_tracks'] = len(msd_diff_obj.track_lengths)
                    self.data_list_with_results.at[index, 'num_tracks_D'] = len(msd_diff_obj.D_linfits)
                    self.data_list_with_results.at[index, 'area'] = roi_area
                    self.data_list_with_results.at[index, 'group'] = file_str
                    self.data_list_with_results.at[index, 'group_readable']=group_readable
                    self.data_list_with_results.at[index, 'ensemble_D']=msd_diff_obj.ensemble_fit_D
                    self.data_list_with_results.at[index, 'ensemble_E'] = msd_diff_obj.ensemble_fit_E
                    self.data_list_with_results.at[index, 'ensemble_r_sq'] = msd_diff_obj.ensemble_fit_rsq
                    self.data_list_with_results.at[index, 'ensemble_loglog_K'] = msd_diff_obj.anomolous_fit_K
                    self.data_list_with_results.at[index, 'ensemble_loglog_aexp'] = msd_diff_obj.anomolous_fit_alpha
                    self.data_list_with_results.at[index, 'ensemble_loglog_r_sq'] = msd_diff_obj.anomolous_fit_rsq

                    if(save_per_file_data):
                        msd_diff_obj.save_msd_data(file_name=file_str + '_' + str(index) + "_MSD.txt")
                        msd_diff_obj.save_fit_data(file_name=file_str + '_' + str(index) + "_Dlin.txt")

                    full_data_i += len(cur_data)
                    Rg_i += 3
                    self.log.write("Processed "+str(index) +" "+cur_file+" for MSD and Diffusion coeff.\n")
                    self.log.flush()

                # ran through all the rows with same csv/image file -- now save the rainbow tracks
                if(self.make_rainbow_tracks and rt_image_found):
                    # save figure of lines plotted on bk_img
                    cur_fig.tight_layout()
                    out_file = os.path.split(self.valid_img_files[common_index])[1][:-4] + '_tracks_Deff.tif'
                    cur_fig.savefig(self.results_dir + '/' + out_file, dpi=self.rainbow_tracks_DPI)
                    plt.close(cur_fig)

                    cur_fig_ss.tight_layout()
                    out_file = os.path.split(self.valid_img_files[common_index])[1][:-4] + '_tracks_ss.tif'
                    cur_fig_ss.savefig(self.results_dir + '/' + out_file, dpi=self.rainbow_tracks_DPI)
                    plt.close(cur_fig_ss)

                    cur_fig_time.tight_layout()
                    out_file = os.path.split(self.valid_img_files[common_index])[1][:-4] + '_tracks_time.tif'
                    cur_fig_time.savefig(self.results_dir + '/' + out_file, dpi=self.rainbow_tracks_DPI)
                    plt.close(cur_fig_time)

                    if(self.output_plain_rainbow_tracks_time):
                        cur_fig_time_plain.tight_layout()
                        out_file = os.path.split(self.valid_img_files[common_index])[1][:-4] + '_tracks_time.pdf'
                        cur_fig_time_plain.savefig(self.results_dir + '/' + out_file, format="pdf", dpi=self.rainbow_tracks_DPI)
                        plt.close(cur_fig_time_plain)

                    # cur_fig_roi.tight_layout()
                    # out_file = os.path.split(self.valid_img_files[common_index])[1][:-4] + '_tracks_roi.tif'
                    # cur_fig_roi.savefig(self.results_dir + '/' + out_file, dpi=self.rainbow_tracks_DPI)
                    # plt.close(cur_fig_roi)

            # Calculate the time-ensemble MSD curve for this group, and do the fitting to get D, alpha
            msd_diff_obj.set_track_data(track_data_df_all.to_numpy())
            msd_diff_obj.msd_all_tracks() # this is repetitive but would be confusing otherwise, keep this way for now
            num_tracks_ens=msd_diff_obj.calculate_ensemble_average()
            msd_diff_obj.fit_msd_ensemble()
            msd_diff_obj.fit_msd_ensemble_alpha()

            # output the fitting results
            self.results_by_group.at[group_i,'group']=file_str
            self.results_by_group.at[group_i,'group_readable']=group_readable
            self.results_by_group.at[group_i,'ensemble_D'] = msd_diff_obj.ensemble_fit_D
            self.results_by_group.at[group_i, 'ensemble_E'] = msd_diff_obj.ensemble_fit_E
            self.results_by_group.at[group_i,'ensemble_r_sq'] = msd_diff_obj.ensemble_fit_rsq
            self.results_by_group.at[group_i,'ensemble_loglog_K'] = msd_diff_obj.anomolous_fit_K
            self.results_by_group.at[group_i,'ensemble_loglog_aexp'] = msd_diff_obj.anomolous_fit_alpha
            self.results_by_group.at[group_i,'ensemble_loglog_r_sq'] = msd_diff_obj.anomolous_fit_rsq
            self.results_by_group.at[group_i,'ensemble_num_tracks'] = num_tracks_ens

            # output the tau vs. MSD points for ensemble average
            for tau in range(1, self.tlag_cutoff_ensemble+1):
                if(len(msd_diff_obj.ensemble_average) >= tau):
                    self.results_by_group.at[group_i, "MSD_ave_"+str(tau*self.time_step)] = msd_diff_obj.ensemble_average[tau-1][1]
                    self.results_by_group.at[group_i, "MSD_std_"+str(tau*self.time_step)] = msd_diff_obj.ensemble_average[tau-1][2]
                else:
                    self.results_by_group.at[group_i, "MSD_ave_" + str(tau * self.time_step)] = np.nan
                    self.results_by_group.at[group_i, "MSD_std_" + str(tau * self.time_step)] = np.nan

        if((self.uneven_time_steps or self.limit_to_ROIs) and full_length > full_data_i):
            #need to remove the extra rows of the df b/c some tracks were filtered
            to_drop = range(full_data_i,full_length,1)
            self.data_list_with_results_full.drop(to_drop, axis=0, inplace=True)

            if (self.radius_of_gyration):
                self.data_list_with_Rg = self.data_list_with_Rg.replace('', np.NaN)
                self.data_list_with_Rg.dropna(axis=1, how='all', inplace=True)

        # navigate through the groups, calculating med(D), etc for each group
        # add to the group data frame
        for row in self.results_by_group.iterrows():
            cur_group = row[1]['group']
            cur_group_data = self.data_list_with_results_full[self.data_list_with_results_full['group'] == cur_group]['D']
            if(len(cur_group_data)>1):

                self.results_by_group.loc[row[0],'D_group_median']=np.median(cur_group_data)
                self.results_by_group.loc[row[0], 'D_group_mean'] = np.mean(cur_group_data)
                self.results_by_group.loc[row[0], 'D_group_std'] = np.std(cur_group_data)
                self.results_by_group.loc[row[0], 'D_group_sem'] = np.std(cur_group_data)/np.sqrt(len(cur_group_data))

                cur_group_data = self.data_list_with_results_full[self.data_list_with_results_full['group'] == cur_group]['aexp']

                self.results_by_group.loc[row[0], 'aexp_group_median'] = np.median(cur_group_data)
                self.results_by_group.loc[row[0], 'aexp_group_mean'] = np.mean(cur_group_data)
                self.results_by_group.loc[row[0], 'aexp_group_std'] = np.std(cur_group_data)
                self.results_by_group.loc[row[0], 'aexp_group_sem'] = np.std(cur_group_data) / np.sqrt(len(cur_group_data))

            self.results_by_group.loc[row[0], 'group_num_tracks'] = len(cur_group_data)

        self.data_list_with_results.to_csv(self.results_dir + '/' + "summary.txt", sep='\t')
        self.data_list_with_results_full.to_csv(self.results_dir + '/' + "all_data.txt", sep='\t')
        self.results_by_group.to_csv(self.results_dir + '/' + "group_summary.txt", sep='\t')
        if (self.radius_of_gyration):
            self.data_list_with_Rg.to_csv(self.results_dir + '/' + "all_data_track_len_and_Rg.txt", sep='\t')

