from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
import os
from skimage import img_as_ubyte, io
from numpy import linalg as LA
from scipy.stats import kurtosis
from skimage import draw

def reshape_to_rgb(grey_img):
    # makes single color channel image into rgb
    ret_img = np.zeros(shape=[grey_img.shape[0], grey_img.shape[1], 3], dtype='uint8')
    grey_img_ = img_as_ubyte(grey_img)

    ret_img[:, :, 0] = grey_img_
    ret_img[:, :, 1] = grey_img_
    ret_img[:, :, 2] = grey_img_
    return ret_img

class msd_diffusion:

    def __init__(self):
        self.tracks = np.asarray([])
        self.msd_tracks = np.asarray([])
        self.D_linfits = np.asarray([])
        self.D_linfits_E = np.asarray([])
        self.track_lengths = np.asarray([])
        self.track_frames = np.asarray([])
        self.track_step_sizes=np.asarray([])
        self.save_dir = '.'

        self.time_step = 0.010 #time between frames, in seconds
        self.micron_per_px = 0.11

        self.min_track_len_linfit=11  # min. 11 track length gives at least 10 tlag values
        self.min_track_len_loglogfit=11
        self.min_track_len_ensemble=11

        self.min_track_len_step_size = 3
        self.max_tlag_step_size=10

        self.tlag_cutoff_linfit=10
        self.tlag_cutoff_loglogfit=10
        self.tlag_cutoff_linfit_ensemble=10
        self.tlag_cutoff_loglogfit_ensemble=10

        self.perc_tlag_linfit=25
        self.use_perc_tlag_linfit=False

        self.initial_guess_linfit=0.2
        self.initial_guess_aexp=1

        self.fit_msd_with_error_term=False
        self.fit_msd_with_no_error_term=True

        self.tracks_num_cols=5
        self.tracks_id_col=0
        self.tracks_frame_col=1
        self.tracks_x_col=2
        self.tracks_y_col=3
        self.tracks_step_size_col=4

        self.msd_num_cols=8
        self.msd_id_col = 0
        self.msd_t_col = 1
        self.msd_frame_col = 2
        self.msd_x_col = 3
        self.msd_y_col = 4
        self.msd_msd_col = 5
        self.msd_std_col = 6
        self.msd_len_col = 7

        self.D_lin_num_cols=7
        self.D_lin_id_col = 0
        self.D_lin_D_col = 1
        self.D_lin_err_col = 2
        self.D_lin_rsq_col = 3
        self.D_lin_rmse_col = 4
        self.D_lin_len_col = 5
        self.D_lin_fitlen_col = 6

        self.D_lin_E_num_cols = 8
        self.D_lin_E_id_col = 0
        self.D_lin_E_D_col = 1
        self.D_lin_E_E_col = 2
        self.D_lin_E_err_col = 3
        self.D_lin_E_rsq_col = 4
        self.D_lin_E_rmse_col = 5
        self.D_lin_E_len_col = 6
        self.D_lin_E_fitlen_col = 7

        self.D_loglog_num_cols=9
        self.D_loglog_id_col=0
        self.D_loglog_K_col = 1
        self.D_loglog_alpha_col = 2
        self.D_loglog_errK_col = 3
        self.D_loglog_erralpha_col = 4
        self.D_loglog_rsq_col = 5
        self.D_loglog_rmse_col = 6
        self.D_loglog_len_col = 7
        self.D_loglog_fitlen_col = 8

        # self.alpha_app_num_cols = 7
        # self.alpha_app_id_col = 0
        # self.alpha_app_param_col = 1
        # self.alpha_app_errparam_col = 2
        # self.alpha_app_rsq_col = 3
        # self.alpha_app_rmse_col = 4
        # self.alpha_app_len_col = 5
        # self.alpha_app_fitlen_col = 6

        self.ensemble_t_col=0
        self.ensemble_msd_col=1
        self.ensemble_std_col=2
        self.ensemble_n_col=3


    def set_track_data(self, track_data):
        self.tracks=track_data
        self.track_lengths = np.asarray([])
        self.track_frames = np.asarray([])
        self.track_step_sizes = np.asarray([])
        self.msd_tracks = np.asarray([])
        self.D_linfits = np.asarray([])
        self.D_linfits_E = np.asarray([])
        self.D_loglogfits = np.asarray([])
        self.alpha_appfits = np.asarray([])
        self.r_of_g = np.asarray([])
        self.avg_velocity = np.asarray([])
        self.r_of_g_full = np.asarray([])
        self.ngp = np.asarray([])
        self.kurt = np.asarray([])
        self.ensemble_average = np.asarray([])
        self.track_intensities = np.asarray([])
        self.fill_track_lengths()
        self.fill_track_sizes()

    def fill_track_intensities(self, tif_movie, r, filter=True):
        # sets the average track intensity for each track
        ids = np.unique(self.tracks[:, self.tracks_id_col])

        if (filter):  # remove tracks that are too short, less MSD to calculate, faster
            min_track_length = np.min([self.min_track_len_linfit, self.min_track_len_loglogfit])

            valid_track_lens = self.track_lengths[np.where(self.track_lengths[:, 1] >= min_track_length)]
            if (len(valid_track_lens) == 0):
                self.track_intensities = np.asarray([])
                return ()
            ids = valid_track_lens[:, 0]

        self.track_intensities = np.zeros((len(ids), 3), )
        for i,id in enumerate(ids):
            cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]

            intens_list = []
            for j in range(len(cur_track)):
                rr, cc = draw.disk((int(cur_track[j][self.tracks_y_col]), int(cur_track[j][self.tracks_x_col])),
                                   r, shape=tif_movie[0].shape)
                #if(np.max(rr) > tif_movie[0].shape[0] or np.max(cc) > tif_movie[0].shape[1]):
                #    print("ERROR")
                intens_list.append(np.mean(tif_movie[int(cur_track[j][self.tracks_frame_col])][rr, cc]))

            self.track_intensities[i][0] = cur_track[0][self.tracks_id_col]
            self.track_intensities[i][1] = np.mean(intens_list)
            self.track_intensities[i][2] = np.std(intens_list)

    def vel_2d(self, x, y):
        dists = np.sqrt(np.square(x[1:] - x[:-1]) + np.square(y[1:] - y[:-1]))
        vels=dists/self.time_step
        return np.mean(vels)

    def msd2d(self, x, y):

        shifts = np.arange(1, len(x), 1)
        MSD = np.zeros(shifts.size)
        MSD_std = np.zeros(shifts.size)

        for i, shift in enumerate(shifts):
            sum_diffs_sq = np.square(x[shift:] - x[:-shift]) + np.square(y[shift:] - y[:-shift])
            MSD[i] = np.mean(sum_diffs_sq)
            MSD_std[i] = np.std(sum_diffs_sq)

        return MSD, MSD_std

    def D_time_nofit(self, tlag=2, min_track_len=3):
        # for each track, get the MSD at given tlag - this is time average MSD
        # for each track, do MSD calculation

        # remove tracks that are too short: < tlag+1 or user selected
        if(min_track_len < tlag+1):
            min_track_len=tlag+1
        ids = self.track_lengths[np.where(self.track_lengths[:, 1] >= min_track_len)][:, 0]

        if (len(ids) == 0):
            return ()

        self.D_inst = np.zeros((len(ids), 2), ) # 2 cols: track_id, D-value = MSD-at-tlag/(4*tlag)
        for i,id in enumerate(ids):
            cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]
            x=cur_track[:, self.tracks_x_col] * self.micron_per_px
            y=cur_track[:, self.tracks_y_col] * self.micron_per_px

            sum_diffs_sq = np.square(x[tlag:] - x[:-tlag]) + np.square(y[tlag:] - y[:-tlag]) # adjust to avoid correlation?
            self.D_inst[i, 0] = id
            self.D_inst[i,1] = np.mean(sum_diffs_sq)/(4*tlag*self.time_step)

    def D_ens_nofit(self, tlag=2, min_track_len=3):
        # for each frame, n, starting at frame 1 + tlag
        # get all tracks that have values for frames from frame (n-tlag) to tlag
        # calculate the distance between x,y at (n-tlag) and x,y at tlag
        # average over all tracks - this is one data point (ID == the frame ending point)

        # remove tracks that are too short: < tlag+1 or user selected
        if (min_track_len < tlag + 1):
            min_track_len = tlag + 1
        ids = self.track_lengths[np.where(self.track_lengths[:, 1] >= min_track_len)][:, 0]
        if (len(ids) == 0):
            return ()

        valid_tracks = self.tracks[np.isin(self.tracks[:, self.tracks_id_col], ids)]

        first_frame=int(np.min(self.tracks[:,self.tracks_frame_col]))
        last_frame = int(np.max(self.tracks[:, self.tracks_frame_col]))
        self.D_inst_ens = np.zeros(shape=((last_frame-tlag)+1, 2))
        for i,frame0 in enumerate(range(first_frame, (last_frame-tlag)+1, 1)):  # ( or should we go up by tlag each time to avoid correlation? )
            # pull tracks from frame to frame+tlag
            d_arr=[]
            frame0_track_rows = valid_tracks[valid_tracks[:,self.tracks_frame_col]==frame0]
            for frame0_track_row in frame0_track_rows:
                frame1_track_row=valid_tracks[(valid_tracks[:,self.tracks_frame_col]==(frame0+tlag)) &
                                              (valid_tracks[:,self.tracks_id_col]==frame0_track_row[self.tracks_id_col])]
                if(len(frame1_track_row)>0):
                    frame1_track_row=frame1_track_row[0]

                    x1 = frame0_track_row[self.tracks_x_col] * self.micron_per_px
                    y1 = frame0_track_row[self.tracks_y_col] * self.micron_per_px

                    x2 = frame1_track_row[self.tracks_x_col] * self.micron_per_px
                    y2 = frame1_track_row[self.tracks_y_col] * self.micron_per_px

                    d_arr.append(np.square(x2 - x1) + np.square(y2 - y1))

            self.D_inst_ens[i, 0] = frame0+tlag
            if(len(d_arr)>0):
                self.D_inst_ens[i, 1] = np.mean(d_arr)/(4*tlag*self.time_step)
            else:
                print(f"No trajectories found for frame0={frame0}")
                self.D_inst_ens[i, 1] = -1

    def fill_track_lengths(self):
        # fill track length array with the track lengths
        ids = np.unique(self.tracks[:, self.tracks_id_col])
        self.track_lengths = np.zeros((len(ids), 2))
        self.track_frames = np.zeros((len(ids), 3))
        for i,id in enumerate(ids):
            cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]
            self.track_lengths[i,0] = id
            self.track_lengths[i,1] = len(cur_track)

            self.track_frames[i,0] = id
            self.track_frames[i,1] = cur_track[:, self.tracks_frame_col].min()
            self.track_frames[i,2] = cur_track[:, self.tracks_frame_col].max()

    def fill_track_sizes(self):
        # add column to tracks array containing the step size for each step of each track (distance between points)
        # step size in *microns*
        self.tracks = np.append(self.tracks, np.zeros((len(self.tracks),1)), axis=1)
        ids = np.unique(self.tracks[:, self.tracks_id_col])
        ss_i=0
        for i,id in enumerate(ids):
            cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]
            ss_i+=1
            for j in range(1,len(cur_track),1):
                d = np.sqrt((cur_track[j, self.tracks_x_col] - cur_track[j-1, self.tracks_x_col]) ** 2 +
                        (cur_track[j, self.tracks_y_col] - cur_track[j-1, self.tracks_y_col]) ** 2)
                self.tracks[ss_i,self.tracks_step_size_col] = d * self.micron_per_px
                ss_i+=1

    def step_sizes_and_angles(self):
        # calculates step sizes and angles for tracks with min. length that is given for Linear fit
        # steps sizes in *microns*
        ids = self.track_lengths[self.track_lengths[:,1] >= self.min_track_len_step_size][:,0]
        if(len(ids) == 0):
            self.step_sizes=np.asarray([])
            self.angles=np.asarray([])
            return ()

        print("SS/Angles number of tracks:", len(ids))
        track_lens=self.track_lengths[self.track_lengths[:,1] >= self.min_track_len_step_size][:,1]

        #rows correspond to t-lag==1,2,3,4,5, columns list the step sizes
        tlag1_dim_steps = int(np.sum(track_lens - 1))
        self.step_sizes = np.empty((self.max_tlag_step_size, tlag1_dim_steps,))
        self.step_sizes.fill(np.nan)

        #displacements in x and y directions
        self.deltaX = np.empty((self.max_tlag_step_size, tlag1_dim_steps,))
        self.deltaX.fill(np.nan)
        self.deltaY = np.empty((self.max_tlag_step_size, tlag1_dim_steps,))
        self.deltaY.fill(np.nan)

        tlag1_dim_angles = int(np.sum(track_lens - 2))
        self.angles = np.empty((self.max_tlag_step_size, tlag1_dim_angles,))
        self.angles.fill(np.nan)

        start_arr=np.zeros((self.max_tlag_step_size,), dtype='int')
        angle_start_arr = np.zeros((self.max_tlag_step_size,), dtype='int')
        for id_i,id in enumerate(ids):
            cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]
            num_shifts = min(self.max_tlag_step_size,len(cur_track)-1)
            max_num_angle_tlags= min(self.max_tlag_step_size, int((len(cur_track) - 1) / 2))
            x = cur_track[:, self.tracks_x_col]
            y = cur_track[:, self.tracks_y_col]
            tlags = np.arange(1, num_shifts+1, 1)
            for i, tlag in enumerate(tlags):
                x_shifts = x[tlag:] - x[:-tlag]
                y_shifts = y[tlag:] - y[:-tlag]

                sum_diffs_sq = np.square(x[tlag:] - x[:-tlag]) + np.square(y[tlag:] - y[:-tlag])
                self.step_sizes[i][start_arr[i]:start_arr[i] + len(sum_diffs_sq)] = np.sqrt(sum_diffs_sq) * self.micron_per_px
                self.deltaX[i][start_arr[i]:start_arr[i] + len(sum_diffs_sq)] = x_shifts * self.micron_per_px
                self.deltaY[i][start_arr[i]:start_arr[i] + len(sum_diffs_sq)] = y_shifts * self.micron_per_px
                start_arr[i] += len(sum_diffs_sq)

                if(tlag <= max_num_angle_tlags): # angles for this tlag
                    # relative angle: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3856831/
                    vecs = np.column_stack((x_shifts, y_shifts))
                    theta=np.zeros(len(list(range(0, len(vecs)-tlag, tlag))))
                    for theta_i,vec_i in enumerate(range(0, len(vecs)-tlag, tlag)):
                        if(np.linalg.norm(vecs[vec_i]) == 0 or np.linalg.norm(vecs[vec_i+tlag]) == 0):
                            print("norm of vec is 0: id=", id)
                            #print(np.rad2deg(np.arccos(np.dot(vecs[vec_i],vecs[vec_i+tlag]) / (np.linalg.norm(vecs[vec_i]) * np.linalg.norm(vecs[vec_i+tlag])))))
                        theta[theta_i] = np.nan_to_num(np.rad2deg(
                            np.arccos(np.dot(vecs[vec_i],vecs[vec_i+tlag]) / (np.linalg.norm(vecs[vec_i]) * np.linalg.norm(vecs[vec_i+tlag])))))
                    self.angles[i][angle_start_arr[i]:angle_start_arr[i]+len(theta)] = theta

                    angle_start_arr[i] += len(theta)

    def msd_all_tracks(self, filter=True):
        # for each track, do MSD calculation

        ids = np.unique(self.tracks[:, self.tracks_id_col])
        print("Total number of tracks:", len(ids))

        if(filter): # remove tracks that are too short, less MSD to calculate, faster
            min_track_length = np.min([self.min_track_len_linfit,self.min_track_len_loglogfit,self.min_track_len_ensemble])

            valid_track_lens = self.track_lengths[np.where(self.track_lengths[:, 1] >= min_track_length)]
            if (len(valid_track_lens) == 0):
                self.msd_tracks = np.asarray([])
                return ()
            ids=valid_track_lens[:,0]
            full_len=int(np.sum(valid_track_lens[:,1]))
        else:
            full_len=len(self.tracks)

        self.msd_tracks = np.zeros((full_len, self.msd_num_cols), )
        i=0
        for id in ids:
            cur_track = self.tracks[np.where(self.tracks[:,self.tracks_id_col]==id)]
            n_MSD=len(cur_track)
            cur_MSD, cur_MSD_std = self.msd2d(cur_track[:, self.tracks_x_col] * self.micron_per_px,
                                                              cur_track[:, self.tracks_y_col] * self.micron_per_px)
            self.msd_tracks[i:i+n_MSD,self.msd_id_col] = id
            self.msd_tracks[i:i+n_MSD,self.msd_t_col] = np.arange(0,n_MSD,1)*self.time_step
            self.msd_tracks[i:i + n_MSD, self.msd_frame_col] = cur_track[:, self.tracks_frame_col]
            self.msd_tracks[i:i + n_MSD, self.msd_x_col] = cur_track[:, self.tracks_x_col]
            self.msd_tracks[i:i + n_MSD, self.msd_y_col] = cur_track[:, self.tracks_y_col]
            self.msd_tracks[i+1:i+n_MSD,self.msd_msd_col] = cur_MSD
            self.msd_tracks[i+1:i+n_MSD,self.msd_std_col] = cur_MSD_std
            self.msd_tracks[i:i+n_MSD,self.msd_len_col] = n_MSD-1
            i += n_MSD

    def radius_of_gyration_full(self, ):
        # for each track, do r. of g. calculation
        # calculation is stored for each n (n = # of steps) to observe how it evolves over time (see Elliot, et al)
        # https://doi.org/10.1039/c0cp01805h
        if(len(self.tracks)>0):

            ids = np.unique(self.tracks[:, self.tracks_id_col])
            self.r_of_g_full = np.zeros((len(self.tracks), 3), )
            i = 0
            for id in ids:
                cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]
                n=len(cur_track)
                self.r_of_g_full[i, 0] = id
                self.r_of_g_full[i, 1] = 0
                self.r_of_g_full[i, 2] = 0 # need atleast 2 points to get an r-of-g...set value for 1st position to 0
                i+=1
                for j in range(2,n+1,1):
                    T=np.cov(cur_track[:j,self.tracks_x_col],
                             cur_track[:j,self.tracks_y_col])*(j-1)/j
                    w, v = LA.eig(T)
                    self.r_of_g_full[i,0]=id
                    self.r_of_g_full[i,1]=(j-1) * self.time_step
                    self.r_of_g_full[i,2]=np.sqrt(np.sum(w)) * self.micron_per_px #eigenvalues (w) are squared radii of gyration
                    i+=1
        else:
            self.r_of_g_full=np.asarray([])

    def average_velocity(self, len_cutoff=0):
        # for each track, calculate average velocity
        # if len_cutoff > 1, then velocity will be calculated only for tracks >= len_cutoff
        # otherwise, will be calculated for all tracks

        if(len(self.tracks)>0):
            if(len_cutoff>1):
                valid_track_ids = self.track_lengths[np.where(self.track_lengths[:, 1] >= len_cutoff)][:,0]
            else:
                valid_track_ids = np.unique(self.tracks[:, self.tracks_id_col])

            if(len(valid_track_ids)>0):
                self.avg_velocity = np.zeros((len(valid_track_ids), 2), )
                i = 0
                for id in valid_track_ids:
                    cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]

                    cur_avg = self.vel_2d(cur_track[:, self.tracks_x_col] * self.micron_per_px,
                                          cur_track[:, self.tracks_y_col] * self.micron_per_px)

                    self.avg_velocity[i, 0] = id
                    self.avg_velocity[i, 1] = cur_avg

                    i += 1
            else:
                self.avg_velocity = np.asarray([])
        else:
            self.avg_velocity=np.asarray([])

    def radius_of_gyration(self, len_cutoff=0):
        # for each track, do r. of g. calculation (calculated only for a single length)
        # if len_cutoff > 1, then r.of.g. will be calculated at len_cutoff and only for tracks >= len_cutoff
        # otherwise, r.of.g. will be calculated for all tracks at their full length
        # https://doi.org/10.1039/c0cp01805h

        if(len(self.tracks)>0):
            if(len_cutoff>1):
                valid_track_ids = self.track_lengths[np.where(self.track_lengths[:, 1] >= len_cutoff)][:,0]
            else:
                valid_track_ids = np.unique(self.tracks[:, self.tracks_id_col])

            if(len(valid_track_ids)>0):
                self.r_of_g = np.zeros((len(valid_track_ids), 2), )
                i = 0
                for id in valid_track_ids:
                    cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]
                    if(len_cutoff > 1):
                        n=len_cutoff
                    else:
                        n=len(cur_track)
                    T = np.cov(cur_track[:n, self.tracks_x_col], cur_track[:n, self.tracks_y_col]) * (n - 1) / n  # np.cov divides by (n-1) but i want to divide by n

                    w, v = LA.eig(T)
                    self.r_of_g[i, 0] = id
                    self.r_of_g[i, 1] = np.sqrt(np.sum(w)) * self.micron_per_px  # eigenvalues (w) are squared radii of gyration
                    i += 1
            else:
                self.r_of_g = np.asarray([])
        else:
            self.r_of_g=np.asarray([])


    def non_gaussian_1d(self): # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5300785 & https://doi.org/10.1016/j.devcel.2020.07.020
        #related to 4th order moment (kurtosis), for gaussian distribution NGP = 0

        if (len(self.deltaX) > 0):
            self.ngp = np.zeros(len(self.deltaX))
            self.kurt = np.zeros(len(self.deltaX))
            for i in range(len(self.deltaX)):
                cur_arrX = self.deltaX[i, :][np.logical_not(np.isnan(self.deltaX[i, :]))]
                cur_arrY = self.deltaY[i, :][np.logical_not(np.isnan(self.deltaY[i, :]))]
                cur_arr = np.concatenate((cur_arrX,cur_arrY))
                self.ngp[i] = (np.mean(cur_arr**4) / (3 * np.mean(cur_arr**2) ** 2)) - 1
                self.kurt[i]=kurtosis(cur_arr)
        else:
            self.ngp = np.asarray([])
            self.kurt = np.asarray([])

            # self.ngpX = np.zeros(len(self.deltaX))
            # for i in range(len(self.deltaX)):
            #     cur_arr = self.deltaX[i, :][np.logical_not(np.isnan(self.deltaX[i, :]))]
            #     self.ngpX[i] = (np.mean(cur_arr**4) / (3 * np.mean(cur_arr**2) ** 2)) - 1
            #
            # self.ngpY = np.zeros(len(self.deltaY))
            # for i in range(len(self.deltaY)):
            #     cur_arr = self.deltaY[i, :][np.logical_not(np.isnan(self.deltaY[i, :]))]
            #     self.ngpY[i] = (np.mean(cur_arr**4) / (3 * np.mean(cur_arr**2) ** 2)) - 1


    def calculate_ensemble_average(self, limit_by_fitting=True):
        # average the MSD at each t-lag

        ensemble_average = []
        len_uniq=0
        if (len(self.msd_tracks) > 0):
            # filter tracks by length
            valid_tracks = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_len_col] >= (self.min_track_len_ensemble - 1))]
            if(len(valid_tracks)>0):

                if(limit_by_fitting):
                    max_tlag = np.max([self.tlag_cutoff_linfit_ensemble, self.tlag_cutoff_loglogfit_ensemble])
                    max_tlag = np.min([max_tlag, int(np.max(valid_tracks[:, self.msd_len_col]))])
                else:
                    max_tlag = int(np.max(valid_tracks[:, self.msd_len_col]))
                for tlag in range(1,max_tlag+1,1):
                    # gather all data for current tlag
                    tlag_time=tlag*self.time_step
                    cur_tlag_MSDs=valid_tracks[valid_tracks[:, self.msd_t_col] == tlag_time][:,self.msd_msd_col]

                    ensemble_average.append([tlag_time, np.mean(cur_tlag_MSDs), np.std(cur_tlag_MSDs), len(cur_tlag_MSDs)])
                len_uniq=len(np.unique(valid_tracks[:, self.msd_id_col]))

        self.ensemble_average=np.asarray(ensemble_average)
        return len_uniq

    def fit_msd_ensemble_alpha(self):
        #MSD(t) = L * t ^ a
        #L = 4K
        #log(MSD) = a * log(t) + log(L)
        #FIT to straight line

        # uses ensemble average to fit for anomolous exponent
        if (len(self.ensemble_average) == 0):
            self.anomolous_fit_rsq = 0
            self.anomolous_fit_rmse = 0
            self.anomolous_fit_K = 0
            self.anomolous_fit_alpha = 0
            self.anomolous_fit_errs = 0
            return ()

        stop = self.tlag_cutoff_loglogfit_ensemble

        def linear_fn(x, m, b):
            return m * x + b
        linear_fn_v = np.vectorize(linear_fn)

        popt, pcov = curve_fit(linear_fn, np.log(self.ensemble_average[:stop, self.ensemble_t_col]),
                               np.nan_to_num(np.log(self.ensemble_average[:stop, self.ensemble_msd_col])),
                               p0=[self.initial_guess_aexp, np.log(4*self.initial_guess_linfit)])

        residuals = np.nan_to_num(np.log(self.ensemble_average[:stop,
                                         self.ensemble_msd_col])) - linear_fn_v(np.log(self.ensemble_average[:stop,
                                                                                       self.ensemble_t_col]),
                                                                                popt[0], popt[1])
        ss_res = np.sum(residuals ** 2)
        rmse = np.mean(residuals ** 2) ** 0.5
        ss_tot = np.sum((np.nan_to_num(np.log(self.ensemble_average[:stop,
                                              self.ensemble_msd_col]))-np.mean(np.nan_to_num(np.log(self.ensemble_average[:stop,
                                                                                                    self.ensemble_msd_col]))))**2)

        r_squared = max(0, 1 - (ss_res / ss_tot))

        self.anomolous_fit_rsq = r_squared
        self.anomolous_fit_rmse = rmse
        self.anomolous_fit_K = np.exp(popt[1])/4
        self.anomolous_fit_alpha = popt[0]
        self.anomolous_fit_errs = np.sqrt(np.diag(pcov))  # one standard deviation errors on the parameters

    def fit_msd_ensemble(self):
        # uses ensemble average to fit for Deff
        # only fits up to the cutoff, self.tlag_cutoff_linfit

        if (len(self.ensemble_average) == 0):
            self.ensemble_fit_rsq = 0
            self.ensemble_fit_rmse = 0
            self.ensemble_fit_D = 0
            self.ensemble_fit_err = 0
            return ()

        stop = self.tlag_cutoff_linfit_ensemble

        if (self.fit_msd_with_error_term):
            def linear_fn(x, a, c):
                return 4 * a * x + c

            linear_fn_v = np.vectorize(linear_fn)
            popt, pcov = curve_fit(linear_fn,
                                   self.ensemble_average[:stop,self.ensemble_t_col],
                                   self.ensemble_average[:stop,self.ensemble_msd_col],
                                   p0=[self.initial_guess_linfit, 0])
            residuals = self.ensemble_average[:stop,
                        self.ensemble_msd_col]-linear_fn_v(self.ensemble_average[:stop,
                                                           self.ensemble_t_col], popt[0], popt[1])
            ss_res = np.sum(residuals ** 2)
            rmse = np.mean(residuals ** 2) ** 0.5
            ss_tot = np.sum((self.ensemble_average[:stop,
                             self.ensemble_msd_col] - np.mean(self.ensemble_average[:stop,
                                                              self.ensemble_msd_col])) ** 2)
            r_squared = max(0, 1 - (ss_res / ss_tot))
            perr = np.sqrt(np.diag(pcov))  # one standard deviation error on the parameters

            self.ensemble_fit_E_rsq = r_squared
            self.ensemble_fit_E_rmse = rmse
            self.ensemble_fit_E_D = popt[0]
            self.ensemble_fit_E = popt[1]
            self.ensemble_fit_E_Derr = perr[0]

        if(self.fit_msd_with_no_error_term):
            def linear_fn(x, a):
                return 4 * a * x

            linear_fn_v = np.vectorize(linear_fn)
            popt, pcov = curve_fit(linear_fn,
                                   self.ensemble_average[:stop, self.ensemble_t_col],
                                   self.ensemble_average[:stop, self.ensemble_msd_col],
                                   p0=[self.initial_guess_linfit])
            residuals = self.ensemble_average[:stop,
                        self.ensemble_msd_col]-linear_fn_v(self.ensemble_average[:stop,
                                                           self.ensemble_t_col], popt[0])
            ss_res = np.sum(residuals ** 2)
            rmse = np.mean(residuals ** 2) ** 0.5
            ss_tot = np.sum((self.ensemble_average[:stop,
                             self.ensemble_msd_col] - np.mean(self.ensemble_average[:stop,
                                                              self.ensemble_msd_col])) ** 2)
            r_squared = max(0, 1 - (ss_res / ss_tot))
            perr = np.sqrt(np.diag(pcov))  # one standard deviation error on the parameters

            self.ensemble_fit_rsq = r_squared
            self.ensemble_fit_rmse = rmse
            self.ensemble_fit_D = popt[0]
            self.ensemble_fit_err = perr[0]







    def plot_msd_ensemble(self, file_name="msd_ensemble.pdf", ymax=-1, xmax=-1, fit_line=False):

        def power_law_fn(x, a, D):
            return 4*D*x**a

        power_law_fn_v = np.vectorize(power_law_fn)

        # loglog plot and anomolous exp
        plt.scatter(self.ensemble_average[:, 0], self.ensemble_average[:, 1], s=2, marker='o')

        if (fit_line):  # np.exp(popt[1])/4
            y_vals_from_fit = power_law_fn_v(self.ensemble_average[:, 0], self.anomolous_fit_alpha, self.anomolous_fit_K)
            plt.plot(self.ensemble_average[:, 0], y_vals_from_fit, linestyle='dashed', linewidth=0.5, color='red', alpha=0.75)
            #plt.axvline(self.tlag_cutoff_loglogfit_ensemble * self.time_step)
            plt.title('D = ' + str(np.round(self.anomolous_fit_K, 4)) + ', a = ' + str(np.round(self.anomolous_fit_alpha, 4)))

        if (ymax > 0):
            plt.ylim(np.min(y_vals_from_fit), ymax)
        if (xmax > 0):
            plt.xlim(self.ensemble_average[0, 0], xmax)
        plt.xlabel('tau (s)')
        plt.ylabel('MSD (micron^2)')

        (file_root, ext) = os.path.splitext(file_name)
        plt.savefig(self.save_dir + '/' + file_root + '-loglog' + ext)
        plt.clf()

    def plot_msd_ensemble2(self, file_name="msd_ensemble.pdf", ymax=-1, xmax=-1, fit_line=False):

        def linear_fn(x, a):
            return 4 * a * x
        linear_fn_v = np.vectorize(linear_fn)

        def linear_fn2(x, m, b):
            return m * x + b
        linear_fn_v2 = np.vectorize(linear_fn2)

        #linear scale and eff-D
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.plot(self.ensemble_average[:, 0], self.ensemble_average[:, 1], color='grey')
        plt.xscale('linear')
        plt.yscale('linear')
        if(fit_line):
            y_vals_from_fit = linear_fn_v(self.ensemble_average[:, 0], self.ensemble_fit_D)
            plt.plot(self.ensemble_average[:, 0], y_vals_from_fit, color='red')
            plt.axvline(self.tlag_cutoff_linfit_ensemble*self.time_step)
            plt.title('D-eff (ens)=' + str(np.round(self.ensemble_fit_D, 2)))

        lower_bound = self.ensemble_average[:, 1] - self.ensemble_average[:, 2]
        lower_bound[lower_bound < 0] = 0
        plt.fill_between(self.ensemble_average[:, 0],
                         lower_bound,
                         self.ensemble_average[:, 1] + self.ensemble_average[:, 2],
                         color='gray', alpha=0.8)
        if (ymax > 0):
            plt.ylim(0, ymax)
        if (xmax > 0):
            plt.xlim(0, xmax)
        plt.xlabel('t-lag (s)')
        plt.ylabel('MSD (microns^2)')
        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()

        # loglog plot and anomolous exp
        plt.plot(np.log10(self.ensemble_average[:, 0]), np.log10(self.ensemble_average[:, 1]), color='grey')

        if(fit_line): # np.exp(popt[1])/4
            y_vals_from_fit = linear_fn_v2(np.log10(self.ensemble_average[:, 0]), self.anomolous_fit_alpha, np.log10(4*self.anomolous_fit_K))
            plt.plot(np.log10(self.ensemble_average[:, 0]), y_vals_from_fit, color='red')
            plt.axvline(np.log10(self.tlag_cutoff_loglogfit_ensemble*self.time_step))
            plt.title('K, alpha = ' + str(np.round(self.anomolous_fit_K, 2)) + ', ' + str(np.round(self.anomolous_fit_alpha, 2)))

        if (ymax > 0):
            plt.ylim(np.min(y_vals_from_fit), np.log10(ymax))
        if (xmax > 0):
            plt.xlim(np.log10(self.ensemble_average[0, 0]), np.log10(xmax))
        plt.xlabel('log10[t-lag (s)]')
        plt.ylabel('log10[MSD (microns^2)]')

        (file_root, ext) = os.path.splitext(file_name)
        plt.savefig(self.save_dir + '/' + file_root + '-loglog' + ext)
        plt.clf()

    def fit_msd(self):
        # fit MSD curve to get Diffusion coefficient
        # filter for tracks > min-length

        if(len(self.msd_tracks) == 0):
            self.D_linfits = np.asarray([])
            self.D_linfits_E = np.asarray([])
            return ()

        #msd_len is one less than the track len
        valid_tracks = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_len_col] >= (self.min_track_len_linfit-1))]

        if(len(valid_tracks) == 0):
            self.D_linfits = np.asarray([])
            self.D_linfits_E = np.asarray([])
            return ()

        ids = np.unique(valid_tracks[:,self.msd_id_col])
        print("Deff number of tracks:", len(ids))

        if (self.fit_msd_with_error_term):
            self.D_linfits_E = np.zeros((len(ids), self.D_lin_E_num_cols,))
        if (self.fit_msd_with_no_error_term):
            self.D_linfits = np.zeros((len(ids), self.D_lin_num_cols,))


        for i,id in enumerate(ids):
            cur_track = valid_tracks[np.where(valid_tracks[:, self.msd_id_col] == id)]
            if(self.use_perc_tlag_linfit):
                stop = int(cur_track[0][self.msd_len_col] * (self.perc_tlag_linfit/100))+1
            else:
                stop = self.tlag_cutoff_linfit+1

            if(self.fit_msd_with_error_term):
                def linear_fn(x, a, c):
                    return 4 * a * x + c
                linear_fn_v = np.vectorize(linear_fn)
                popt, pcov = curve_fit(linear_fn, cur_track[1:stop, self.msd_t_col], cur_track[1:stop, self.msd_msd_col],
                                       p0=[self.initial_guess_linfit, 0])
                residuals = cur_track[1:stop, self.msd_msd_col] - linear_fn_v(cur_track[1:stop, self.msd_t_col],
                                                                              popt[0],
                                                                              popt[1])
                ss_res = np.sum(residuals ** 2)
                rmse = np.mean(residuals ** 2) ** 0.5
                ss_tot = np.sum(
                    (cur_track[1:stop, self.msd_msd_col] - np.mean(cur_track[1:stop, self.msd_msd_col])) ** 2)
                r_squared = max(0, 1 - (ss_res / ss_tot))
                perr = np.sqrt(np.diag(pcov))

                self.D_linfits_E[i][self.D_lin_E_id_col] = id
                self.D_linfits_E[i][self.D_lin_E_D_col] = popt[0]
                self.D_linfits_E[i][self.D_lin_E_E_col] = popt[1]
                self.D_linfits_E[i][self.D_lin_E_err_col] = perr[0]
                self.D_linfits_E[i][self.D_lin_E_rsq_col] = r_squared
                self.D_linfits_E[i][self.D_lin_E_rmse_col] = rmse
                self.D_linfits_E[i][self.D_lin_E_len_col] = cur_track[0][self.msd_len_col]
                self.D_linfits_E[i][self.D_lin_E_fitlen_col] = stop - 1

            if(self.fit_msd_with_no_error_term):
                def linear_fn(x, a):
                    return 4 * a * x
                linear_fn_v = np.vectorize(linear_fn)
                popt, pcov = curve_fit(linear_fn, cur_track[1:stop,self.msd_t_col], cur_track[1:stop,self.msd_msd_col],
                                       p0=[self.initial_guess_linfit])
                residuals = cur_track[1:stop,self.msd_msd_col] - linear_fn_v(cur_track[1:stop,self.msd_t_col],
                                                                             popt[0])

                ss_res = np.sum(residuals ** 2)
                rmse = np.mean(residuals ** 2) ** 0.5
                ss_tot = np.sum((cur_track[1:stop, self.msd_msd_col] - np.mean(cur_track[1:stop, self.msd_msd_col])) ** 2)
                r_squared = max(0, 1 - (ss_res / ss_tot))
                perr = np.sqrt(np.diag(pcov))

                self.D_linfits[i][self.D_lin_id_col]=id
                self.D_linfits[i][self.D_lin_D_col]=popt[0]
                self.D_linfits[i][self.D_lin_err_col]=perr[0]
                self.D_linfits[i][self.D_lin_rsq_col] = r_squared
                self.D_linfits[i][self.D_lin_rmse_col] = rmse
                self.D_linfits[i][self.D_lin_len_col] = cur_track[0][self.msd_len_col]
                self.D_linfits[i][self.D_lin_fitlen_col] = stop-1

    def fit_msd_alpha(self):
        # fit MSD curve to get anomolous exponent, fit of each track
        # filter for tracks > min-length

        # MSD(t) = L * t ^ a
        # L = 4K
        # log(MSD) = a * log(t) + log(L)
        # FIT to straight line

        if(len(self.msd_tracks) == 0):
            self.D_loglogfits = np.asarray([])
            return ()

        #msd_len is one less than the track len
        valid_tracks = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_len_col] >= (self.min_track_len_loglogfit-1))]

        if (len(valid_tracks) == 0):
            self.D_loglogfits=np.asarray([])
            return ()

        ids = np.unique(valid_tracks[:,self.msd_id_col])
        print("Anom exp number of tracks:", len(ids))
        self.D_loglogfits = np.zeros((len(ids),self.D_loglog_num_cols,))
        for i,id in enumerate(ids):
            cur_track = valid_tracks[np.where(valid_tracks[:, self.msd_id_col] == id)]
            stop = self.tlag_cutoff_loglogfit + 1

            def linear_fn(x, m, b):
                return m * x + b
            linear_fn_v = np.vectorize(linear_fn)

            popt, pcov = curve_fit(linear_fn, np.log(cur_track[1:stop,self.msd_t_col]), np.nan_to_num(np.log(cur_track[1:stop,self.msd_msd_col])),
                                   p0=[self.initial_guess_aexp,np.log(4*self.initial_guess_linfit),])
            residuals = np.nan_to_num(np.log(cur_track[1:stop,self.msd_msd_col])) - linear_fn_v(np.log(cur_track[1:stop,self.msd_t_col]), popt[0],popt[1])
            ss_res = np.sum(residuals ** 2)
            rmse = np.mean(residuals**2)**0.5
            ss_tot = np.sum( ( np.nan_to_num(np.log(cur_track[1:stop,self.msd_msd_col])) - np.mean( np.nan_to_num(np.log(cur_track[1:stop,self.msd_msd_col])) ) ) ** 2 )

            r_squared = max(0, 1 - (ss_res / ss_tot))

            self.D_loglogfits[i][self.D_loglog_id_col]=id
            self.D_loglogfits[i][self.D_loglog_K_col] = np.exp(popt[1]) / 4
            self.D_loglogfits[i][self.D_loglog_alpha_col] = popt[0]
            self.D_loglogfits[i][self.D_loglog_errK_col] = np.sqrt(np.diag(pcov))[1]   #   OR should I take np.exp( ) / 4
            self.D_loglogfits[i][self.D_loglog_erralpha_col] = np.sqrt(np.diag(pcov))[0]
            self.D_loglogfits[i][self.D_loglog_rsq_col] = r_squared
            self.D_loglogfits[i][self.D_loglog_rmse_col] = rmse
            self.D_loglogfits[i][self.D_loglog_len_col] = cur_track[0][self.msd_len_col]
            self.D_loglogfits[i][self.D_loglog_fitlen_col] = stop - 1

    # def fit_der_loglog_msd(self):
    #     # fit local derivative of log-log MSD curve to get alpha-apparent=1/(1+c/tau), where c=(2*sigma^2)/(4D)
    #     # As in https://pubmed.ncbi.nlm.nih.gov/12324428/
    #     # Martin, et al: Apparent subdiffusion inherent to single particle tracking, Biophys J 2002
    #
    #     if (len(self.msd_tracks) == 0):
    #         self.alpha_appfits = np.asarray([])
    #         return ()
    #
    #     # filter for tracks > min-length; msd_len is one less than the track len
    #     valid_tracks = self.msd_tracks[np.where(self.msd_tracks[:,
    #                                             self.msd_len_col] >= (self.min_track_len_loglogfit - 1))]
    #
    #     if (len(valid_tracks) == 0):
    #         self.alpha_appfits = np.asarray([])
    #         return ()
    #
    #     ids = np.unique(valid_tracks[:, self.msd_id_col])
    #     self.alpha_appfits = np.zeros((len(ids), self.alpha_app_num_cols,))
    #     for i, id in enumerate(ids):
    #         cur_track = valid_tracks[np.where(valid_tracks[:, self.msd_id_col] == id)]
    #         stop = self.tlag_cutoff_loglogfit + 1
    #
    #         def alpha_app_fn(x, c):
    #             return 1/(1+(c/x))
    #         alpha_app_fn_v = np.vectorize(alpha_app_fn)
    #
    #         # local derivative of log-log MSD
    #         alpha_app = np.diff(np.nan_to_num(np.log(cur_track[1:stop,
    #                                                  self.msd_msd_col])))/np.diff(np.log(cur_track[1:stop,
    #                                                                                      self.msd_t_col]))
    #
    #         # fit curve, local derivative as a function of tau
    #         # TODO: NOTE: Should "weight the one-parameter fit inversely proportional to the density of the data"
    #         popt, pcov = curve_fit(alpha_app_fn, np.log(cur_track[2:stop, self.msd_t_col]), alpha_app, p0=[1])
    #         residuals = alpha_app - alpha_app_fn_v(np.log(cur_track[2:stop, self.msd_t_col]), popt[0])
    #         ss_res = np.sum(residuals ** 2)
    #         rmse = np.mean(residuals ** 2) ** 0.5
    #         ss_tot = np.sum((alpha_app - np.mean(alpha_app)) ** 2)
    #         r_squared = max(0, 1 - (ss_res / ss_tot))
    #
    #         self.alpha_appfits[i][self.alpha_app_id_col] = id
    #         self.alpha_appfits[i][self.alpha_app_param_col] = popt[0]
    #         self.alpha_appfits[i][self.alpha_app_errparam_col] = np.sqrt(np.diag(pcov))[0]
    #         self.alpha_appfits[i][self.alpha_app_rsq_col] = r_squared
    #         self.alpha_appfits[i][self.alpha_app_rmse_col] = rmse
    #         self.alpha_appfits[i][self.alpha_app_len_col] = cur_track[0][self.msd_len_col]
    #         self.alpha_appfits[i][self.alpha_app_fitlen_col] = stop - 2
    #
    # def fit_der_loglog_msd_ensemble(self):
    #     # fit local derivative of log-log MSD curve to get alpha-apparent=1/(1+c/tau), where c=(2*sigma^2)/(4D)
    #     # As in https://pubmed.ncbi.nlm.nih.gov/12324428/
    #     # Martin, et al: Apparent subdiffusion inherent to single particle tracking, Biophys J 2002
    #
    #     # uses ensemble average to fit
    #     if (len(self.ensemble_average) == 0):
    #         self.alpha_app_param = 0
    #         self.alpha_app_errparam = 0
    #         self.alpha_app_rsq = 0
    #         self.alpha_app_rmse = 0
    #         return ()
    #
    #     stop = self.tlag_cutoff_loglogfit_ensemble
    #
    #     def alpha_app_fn(x, c):
    #         return 1/(1+(c/x))
    #     alpha_app_fn_v = np.vectorize(alpha_app_fn)
    #
    #     # local derivative of log-log MSD
    #     alpha_app = np.diff(np.nan_to_num(np.log(self.ensemble_average[:stop, 1])))/np.diff(
    #         np.log(self.ensemble_average[:stop, 0]))
    #
    #     # fit curve, local derivative as a function of tau
    #     # TODO: NOTE: Should "weight the one-parameter fit inversely proportional to the density of the data"
    #     popt, pcov = curve_fit(alpha_app_fn, np.log(self.ensemble_average[1:stop, 0]), alpha_app, p0=[1])
    #     residuals = alpha_app - alpha_app_fn_v(np.log(self.ensemble_average[1:stop, 0]), popt[0])
    #     ss_res = np.sum(residuals ** 2)
    #     rmse = np.mean(residuals ** 2) ** 0.5
    #     ss_tot = np.sum((alpha_app - np.mean(alpha_app)) ** 2)
    #     r_squared = max(0, 1 - (ss_res / ss_tot))
    #
    #     self.alpha_app_param = popt[0]
    #     self.alpha_app_errparam = np.sqrt(np.diag(pcov))[0]
    #     self.alpha_app_rsq = r_squared
    #     self.alpha_app_rmse = rmse


    def save_fit_data(self, file_name="fit_results.txt"):
        df = pd.DataFrame(self.D_linfits)
        df.rename(columns={self.D_lin_id_col: 'id', self.D_lin_D_col: 'D', self.D_lin_err_col: 'err',
                           self.D_lin_rsq_col: 'r_sq', self.D_lin_rmse_col: 'rmse', self.D_lin_len_col: 'track_len',
                           self.D_lin_fitlen_col: 'D_track_len'}, inplace=True)
        df.to_csv(self.save_dir + '/' + file_name, sep='\t', index=False)
        return df

    def save_alpha_fit_data(self, file_name="alpha_fit_results.txt"):
        df = pd.DataFrame(self.D_loglogfits)
        df.rename(columns={self.D_loglog_id_col: 'id', self.D_loglog_K_col: 'K', self.D_loglog_alpha_col: 'alpha',
                           self.D_loglog_errK_col: 'err_K', self.D_loglog_erralpha_col: 'err_alpha', self.D_loglog_rsq_col: 'r_sq',
                           self.D_loglog_rmse_col: 'rmse', self.D_loglog_len_col: 'track_len', self.D_loglog_fitlen_col: 'fit_track_len'}, inplace=True)
        df.to_csv(self.save_dir + '/' + file_name, sep='\t', index=False)
        return df

    def save_msd_data(self, file_name="msd_results.txt"):
        df = pd.DataFrame(self.msd_tracks)
        df.rename(columns={self.msd_id_col: 'id', self.msd_t_col: 't', self.msd_frame_col: 'frame',
                           self.msd_x_col: 'x', self.msd_y_col: 'y', self.msd_msd_col: 'MSD',
                           self.msd_std_col: 'MSD_stdev', self.msd_len_col: 'MSD_len'}, inplace=True)
        df.to_csv(self.save_dir + '/' + file_name, sep='\t', index=False)
        return df

    def plot_msd_curves(self, file_name="msd_all.pdf", max_tracks=50, ymax=-1, xmax=-1,
                        min_track_len=0, fit_line=False, scale='linear'):

        def linear_fn(x, a):
            return 4 * a * x

        linear_fn_v = np.vectorize(linear_fn)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        valid_tracks = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_len_col] >= (min_track_len - 1))]
        ids = np.unique(valid_tracks[:, self.msd_id_col])
        x_range=[]
        for i, id in enumerate(ids):
            cur_track = valid_tracks[np.where(valid_tracks[:, self.msd_id_col] == id)]
            #if(self.D_linfits[i][3]<0.75):
            plt.plot(cur_track[1:,self.msd_t_col], cur_track[1:,self.msd_msd_col], linewidth=".5")
            plt.scatter(cur_track[1:, self.msd_t_col], cur_track[1:, self.msd_msd_col], s=0.5)
            if(len(cur_track[1:,self.msd_t_col]) > len(x_range)):
                x_range=cur_track[1:,self.msd_t_col]

            if (max_tracks != 0 and i+1 >= max_tracks):
                break

        if(fit_line):
            y_vals_from_fit=linear_fn_v(x_range, np.median(self.D_linfits[:,self.D_lin_D_col]))
            plt.plot(x_range, y_vals_from_fit, color='black')
            plt.title('Deff(med)=' + str(np.round(np.median(self.D_linfits[:,self.D_lin_D_col]), 2)) +
                      ' (mean)='+str(np.round(np.mean(self.D_linfits[:,self.D_lin_D_col]),2)))

        if(scale == 'linear'):
            if (ymax > 0):
                plt.ylim(0, ymax)
        else:
            if (ymax > 0):
                plt.ylim(.001, ymax)
        if (xmax > 0):
            plt.xlim(self.time_step, xmax)
        else:
            plt.xlim(self.time_step, plt.xlim()[1])
        plt.xscale(scale)
        plt.yscale(scale)
        plt.xlabel('tau (s)')
        plt.ylabel('MSD (microns^2)')

        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()
        plt.close()

    def plot_sample_tracks0(self, file_name="sample_tracks.pdf", max_tracks=12, min_track_len=11):

        def linear_fn(x, a):
            return 4 * a * x
        linear_fn_v = np.vectorize(linear_fn)

        def linear_fn2(x, m, b):
            return m * x + b
        linear_fn_v2 = np.vectorize(linear_fn2)

        track_ids = np.unique(self.tracks[:,self.tracks_id_col])
        i=0
        fig, axs = plt.subplots(max_tracks, 4,figsize=(20,max_tracks*4))

        for id in track_ids:
            cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]
            if(len(cur_track)>min_track_len):
                #plot
                x_sub=np.min(cur_track[:,self.tracks_x_col]) * self.micron_per_px
                y_sub = np.min(cur_track[:, self.tracks_y_col]) * self.micron_per_px

                axs[i, 0].plot(cur_track[:,self.tracks_x_col] * self.micron_per_px-x_sub,cur_track[:,self.tracks_y_col] * self.micron_per_px-y_sub)
                axs[i, 0].plot(cur_track[0,self.tracks_x_col] * self.micron_per_px-x_sub,cur_track[0,self.tracks_y_col] * self.micron_per_px-y_sub,'.',color='green')
                axs[i, 0].plot(cur_track[-1, self.tracks_x_col] * self.micron_per_px-x_sub, cur_track[-1, self.tracks_y_col] * self.micron_per_px-y_sub, '.', color='red')

                if(len(self.r_of_g_full) > 0):
                    cur_r_of_g = self.r_of_g_full[np.where(self.r_of_g_full[:, 0] == id)]
                    axs[i, 1].plot(cur_r_of_g[1:, 1], cur_r_of_g[1:, 2])
                    axs[i, 1].set_title("Rg(N)="+str(np.round(cur_r_of_g[-1, 2],2)))
                    axs[i, 1].set_ylabel('Rg (microns)')

                if(len(self.msd_tracks)>0):
                    cur_msd = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_id_col] == id)]
                    axs[i, 2].plot(cur_msd[1:self.tlag_cutoff_linfit+1,self.msd_t_col],
                                   cur_msd[1:self.tlag_cutoff_linfit+1,self.msd_msd_col])

                    if (len(self.D_linfits) > 0):
                        cur_D_linfits = self.D_linfits[np.where(self.D_linfits[:, self.D_lin_id_col] == id)]
                        if (len(cur_D_linfits) > 0):
                            cur_D = cur_D_linfits[0, self.D_lin_D_col]
                            cur_r2 = cur_D_linfits[0, self.D_lin_rsq_col]
                            y_vals_from_fit = linear_fn_v(cur_msd[1:self.tlag_cutoff_linfit+1,self.msd_t_col], cur_D)
                            axs[i, 2].plot(cur_msd[1:self.tlag_cutoff_linfit+1,self.msd_t_col], y_vals_from_fit, color='black')
                            axs[i, 2].set_title(f"Deff={np.round(cur_D, 2)}, r2={np.round(cur_r2, 2)}")
                            axs[i, 2].set_ylabel('MSD (microns^2)')

                    if (len(self.D_loglogfits) > 0):
                        cur_D_loglogfits = self.D_loglogfits[np.where(self.D_loglogfits[:, self.D_loglog_id_col] == id)]
                        if (len(cur_D_loglogfits) > 0):
                            axs[i, 3].plot(np.log10(cur_msd[1:self.tlag_cutoff_loglogfit+1, self.msd_t_col]),
                                           np.log10(cur_msd[1:self.tlag_cutoff_loglogfit+1, self.msd_msd_col]))

                            cur_K = cur_D_loglogfits[0, self.D_loglog_K_col]
                            cur_alpha = cur_D_loglogfits[0, self.D_loglog_alpha_col]
                            cur_r2 = cur_D_loglogfits[0, self.D_loglog_rsq_col]
                            y_vals_from_fit = linear_fn_v2(np.log10(cur_msd[1:self.tlag_cutoff_loglogfit+1,self.msd_t_col]), cur_alpha, np.log10(4*cur_K))
                            axs[i, 3].plot(np.log10(cur_msd[1:self.tlag_cutoff_loglogfit+1,self.msd_t_col]), y_vals_from_fit, color='black')
                            axs[i, 3].set_title(f"K={np.round(cur_K, 2)}, alpha={np.round(cur_alpha, 2)}, r2={np.round(cur_r2, 2)}")
                            axs[i, 3].set_ylabel('log10[MSD (microns^2)]')

                i += 1
                if(i == max_tracks):
                    axs[i-1, 1].set_xlabel('t (s)')
                    axs[i-1, 2].set_xlabel('t-lag (s)')
                    axs[i-1, 3].set_xlabel('log10[t-lag (s)]')
                    break

        # for ax in axs.flat:
        #     ax.set(xlabel='x', ylabel='y')
        # for ax in axs.flat:
        #     ax.label_outer()
        plt.tight_layout()
        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()

    def plot_sample_tracks(self, file_name="sample_tracks.pdf", max_tracks=12, min_track_len=11):

        def linear_fn(x, a):
            return 4 * a * x
        linear_fn_v = np.vectorize(linear_fn)

        def linear_fn2(x, a, b):
            return 4*b*x**a
        linear_fn_v2 = np.vectorize(linear_fn2)

        track_ids = np.unique(self.tracks[:,self.tracks_id_col])
        i=0
        fig, axs = plt.subplots(max_tracks, 4,figsize=(20,max_tracks*4))

        for id in track_ids:
            cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]

            # cur_D_loglogfits = self.D_loglogfits[np.where(self.D_loglogfits[:, self.D_loglog_id_col] == id)]
            # if (len(cur_D_loglogfits) > 0):
            #     the_alpha = cur_D_loglogfits[0, self.D_loglog_alpha_col]
            # else:
            #     the_alpha = 0

            if(len(cur_track)>min_track_len): #and int(id)==8267): #and int(id) in [5,13,14,39,291,550,970,1042,1203,1770,1784,1793,1794,]): #the_alpha > 2): #< 0.8): # > 2):
                #plot
                x_sub=np.min(cur_track[:,self.tracks_x_col]) * self.micron_per_px
                y_sub = np.min(cur_track[:, self.tracks_y_col]) * self.micron_per_px

                axs[i, 0].plot(cur_track[:,self.tracks_x_col] * self.micron_per_px-x_sub,cur_track[:,self.tracks_y_col] * self.micron_per_px-y_sub)
                axs[i, 0].plot(cur_track[0,self.tracks_x_col] * self.micron_per_px-x_sub,cur_track[0,self.tracks_y_col] * self.micron_per_px-y_sub,'.',color='green')
                axs[i, 0].plot(cur_track[-1, self.tracks_x_col] * self.micron_per_px-x_sub, cur_track[-1, self.tracks_y_col] * self.micron_per_px-y_sub, '.', color='red')
                axs[i, 0].set_title(f"track: {id}")
                if(len(self.r_of_g_full) > 0):
                    cur_r_of_g = self.r_of_g_full[np.where(self.r_of_g_full[:, 0] == id)]
                    axs[i, 1].plot(cur_r_of_g[1:, 1], cur_r_of_g[1:, 2])
                    axs[i, 1].set_title("Rg(N)="+str(np.round(cur_r_of_g[-1, 2],2)))
                    axs[i, 1].set_ylabel('Rg (microns)')

                if(len(self.msd_tracks)>0):
                    cur_msd = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_id_col] == id)]
                    axs[i, 2].plot(cur_msd[1:self.tlag_cutoff_linfit+1,self.msd_t_col],
                                   cur_msd[1:self.tlag_cutoff_linfit+1,self.msd_msd_col])

                    if (len(self.D_linfits) > 0):
                        cur_D_linfits = self.D_linfits[np.where(self.D_linfits[:, self.D_lin_id_col] == id)]
                        if (len(cur_D_linfits) > 0):
                            cur_D = cur_D_linfits[0, self.D_lin_D_col]
                            cur_r2 = cur_D_linfits[0, self.D_lin_rsq_col]
                            y_vals_from_fit = linear_fn_v(cur_msd[1:self.tlag_cutoff_linfit+1,self.msd_t_col], cur_D)
                            axs[i, 2].plot(cur_msd[1:self.tlag_cutoff_linfit+1,self.msd_t_col], y_vals_from_fit, color='black')
                            axs[i, 2].set_title(f"Deff={np.round(cur_D, 4)}, r2={np.round(cur_r2, 2)}")
                            axs[i, 2].set_ylabel('MSD (microns^2)')

                    if (len(self.D_loglogfits) > 0): # TODO try fitting with K-parameter but directly, not loglog and see if get same result
                        cur_D_loglogfits = self.D_loglogfits[np.where(self.D_loglogfits[:, self.D_loglog_id_col] == id)]
                        if (len(cur_D_loglogfits) > 0):
                            axs[i, 3].plot(cur_msd[1:self.tlag_cutoff_loglogfit+1, self.msd_t_col],
                                           cur_msd[1:self.tlag_cutoff_loglogfit+1, self.msd_msd_col])

                            cur_K = cur_D_loglogfits[0, self.D_loglog_K_col]
                            cur_alpha = cur_D_loglogfits[0, self.D_loglog_alpha_col]
                            cur_r2 = cur_D_loglogfits[0, self.D_loglog_rsq_col]
                            y_vals_from_fit = linear_fn_v2(cur_msd[1:self.tlag_cutoff_loglogfit+1,self.msd_t_col], cur_alpha,cur_K)
                            axs[i, 3].plot(cur_msd[1:self.tlag_cutoff_loglogfit+1,self.msd_t_col], y_vals_from_fit, color='black')
                            axs[i, 3].set_title(f"K={np.round(cur_K, 4)}, alpha={np.round(cur_alpha, 4)}, r2={np.round(cur_r2, 2)}")
                            axs[i, 3].set_ylabel('MSD (microns^2)')

                i += 1
                if(i == max_tracks):
                    axs[i-1, 1].set_xlabel('t (s)')
                    axs[i-1, 2].set_xlabel('t-lag (s)')
                    axs[i-1, 3].set_xlabel('t-lag (s)')
                    break

        # for ax in axs.flat:
        #     ax.set(xlabel='x', ylabel='y')
        # for ax in axs.flat:
        #     ax.label_outer()

        plt.tight_layout()
        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()

    def save_step_sizes(self, file_name="step_sizes.txt"):
        df = pd.DataFrame(self.step_sizes)
        df.insert(0, 't', range(1,len(self.step_sizes)+1))
        df.to_csv(self.save_dir + '/' + file_name, sep='\t', index=False)
        return df

    def save_angles(self, file_name="relative_angles.txt"):
        df = pd.DataFrame(self.angles)
        df.insert(0, 't', range(1, len(self.angles) + 1))
        df.to_csv(self.save_dir + '/' + file_name, sep='\t', index=False)
        return df

    def save_track_length_hist(self, file_name="track_length_histogram.pdf"):
        #to_plot = self.track_lengths[np.where(self.track_lengths[:,1] >= (self.min_track_len_linfit-1))][:,1]
        to_plot = self.track_lengths[:, 1]
        plt.hist(to_plot, bins=np.arange(0,np.max(to_plot),1))
        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()

    def save_step_size_hist(self, file_name="step_sizes.pdf", tlag=1):

        to_plot = self.step_sizes[tlag-1,:]
        to_plot = to_plot[np.logical_not(np.isnan(to_plot))]
        plt.hist(to_plot, bins=np.arange(0,np.max(to_plot),0.1))
        plt.xlabel("microns")
        plt.ylabel("count")
        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()

    def save_angle_hist(self, file_name="angles.pdf", tlag=1):
        to_plot = self.angles[tlag - 1, :]
        to_plot = to_plot[np.logical_not(np.isnan(to_plot))]

        #histogram
        plt.hist(to_plot, bins=np.arange(0, 180+5, 5))
        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()

        #KDE
        gkde = stats.gaussian_kde(to_plot)
        ind = np.arange(0, 180+0.1, 0.1)
        kdepdf = gkde.evaluate(ind)
        plt.plot(ind, kdepdf)
        ext=os.path.splitext(file_name)[1]
        plt.savefig(self.save_dir + '/' + file_name[:-(len(ext))]+"_kde"+ext)
        plt.clf()

    def save_D_histogram(self, file_name="Deff.pdf"):

        df = pd.DataFrame(self.D_linfits)
        df.rename(columns={self.D_lin_id_col: 'id', self.D_lin_D_col: 'D', self.D_lin_err_col: 'err',
                           self.D_lin_rsq_col: 'r_sq', self.D_lin_rmse_col: 'rmse', self.D_lin_len_col: 'track_len',
                           self.D_lin_fitlen_col: 'D_track_len'}, inplace=True)


        to_plot = self.D_linfits[:,self.D_lin_D_col]

        # histogram
        plt.hist(to_plot, bins=np.arange(0, 10, 0.1))
        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()

        # KDE
        gkde = stats.gaussian_kde(to_plot)
        ind = np.arange(0, 10, 0.1)
        kdepdf = gkde.evaluate(ind)
        plt.plot(ind, kdepdf)
        ext = os.path.splitext(file_name)[1]
        plt.savefig(self.save_dir + '/' + file_name[:-(len(ext))] + "_kde" + ext)
        plt.clf()

    def save_tracks_to_img_ss(self, ax, min_ss=0, max_ss=5, lw=0.1):
        # min_ss/max_ss are in microns (not pixels)

        ids = np.unique(self.tracks[:, self.tracks_id_col])
        for id in ids:
            cur_track = self.tracks[self.tracks[:, self.tracks_id_col] == id]
            if (len(cur_track) >= self.min_track_len_step_size):

                for step_i in range(1,len(cur_track),1):
                    cur_ss=cur_track[step_i,self.tracks_step_size_col]
                    if (cur_ss < min_ss):
                        cur_ss = min_ss
                    if (cur_ss > max_ss):
                        cur_ss = max_ss

                    show_color=cur_ss/max_ss

                    ax.plot([cur_track[step_i-1,self.tracks_x_col],cur_track[step_i,self.tracks_x_col]],
                            [cur_track[step_i-1,self.tracks_y_col],cur_track[step_i,self.tracks_y_col]],
                            '-', color=cm.jet(show_color), linewidth=lw)

    def save_tracks_to_img_time(self, ax, relative_to='track', lw=0.1, reverse_coords=False, xlim=0, ylim=0):
    # coloring based on frame number of each track over time
    # uses min track len step size to limit tracks
        if(len(self.track_lengths)>0):
            # first, get the min frame and max frame over all tracks that we will plot
            ids=self.track_lengths[self.track_lengths[:,1]>=self.min_track_len_step_size][:,0]
            if(len(ids)>0):
                if(relative_to=='frame'):
                    valid_tracks = self.tracks[np.isin(self.tracks[:,self.tracks_id_col], ids)]
                    min_frame=valid_tracks[:,self.tracks_frame_col].min()
                    max_frame=valid_tracks[:,self.tracks_frame_col].max()

                    for id in ids:
                        cur_track = self.tracks[self.tracks[:, self.tracks_id_col] == id]

                        for step_i in range(1,len(cur_track),1):
                            cur_frame=cur_track[step_i,self.tracks_frame_col]
                            if (cur_frame < min_frame):
                                cur_frame = min_frame
                            if (cur_frame > max_frame):
                                cur_frame = max_frame

                            show_color=cur_frame/max_frame

                            if(reverse_coords):
                                ax.plot(
                                    [cur_track[step_i - 1, self.tracks_y_col], cur_track[step_i, self.tracks_y_col]],
                                    [cur_track[step_i - 1, self.tracks_x_col], cur_track[step_i, self.tracks_x_col]],
                                    '-', color=cm.jet(show_color), linewidth=lw)
                            else:
                                ax.plot(
                                    [cur_track[step_i-1,self.tracks_x_col],cur_track[step_i,self.tracks_x_col]],
                                    [cur_track[step_i-1,self.tracks_y_col],cur_track[step_i,self.tracks_y_col]],
                                    '-', color=cm.jet(show_color), linewidth=lw)
                else:
                    for id in ids:
                        cur_track = self.tracks[self.tracks[:, self.tracks_id_col] == id]
                        max_step = len(cur_track)
                        for step_i in range(1,max_step,1):

                            show_color=step_i/max_step

                            if(reverse_coords):
                                ax.plot(
                                    [cur_track[step_i - 1, self.tracks_y_col], cur_track[step_i, self.tracks_y_col]],
                                    [cur_track[step_i - 1, self.tracks_x_col], cur_track[step_i, self.tracks_x_col]],
                                    '-', color=cm.jet(show_color), linewidth=lw)

                            else:
                                ax.plot(
                                    [cur_track[step_i - 1, self.tracks_x_col], cur_track[step_i, self.tracks_x_col]],
                                    [cur_track[step_i - 1, self.tracks_y_col], cur_track[step_i, self.tracks_y_col]],
                                    '-', color=cm.jet(show_color), linewidth=lw)
                        # if (reverse_coords):
                        #     ax.text(cur_track[0, self.tracks_y_col],
                        #         cur_track[0, self.tracks_x_col], str(id), color='red')
                        # else:
                        #     pass


    def save_tracks_to_img_clr(self, ax, lw=0.1, color='blue'):
        ids = np.unique(self.tracks[:, self.tracks_id_col])
        for id in ids:
            cur_track = self.tracks[self.tracks[:, self.tracks_id_col] == id]
            if (len(cur_track) >= self.min_track_len_step_size):
                ax.plot(cur_track[:,self.tracks_x_col],cur_track[:,self.tracks_y_col],'-',color=color,linewidth=lw)

    def save_tracks_to_img(self, ax, len_cutoff='none',
                           remove_tracks=False, error_term=False,
                           min_Deff=0.01, max_Deff=2, lw=0.1):
        if(len_cutoff == 'default'):
            len_cutoff=self.min_track_len_linfit

        ids=np.unique(self.tracks[:,self.tracks_id_col])
        for id in ids:
            cur_track=self.tracks[self.tracks[:,self.tracks_id_col]==id]
            if(len(cur_track) >= self.min_track_len_linfit):
                if(len_cutoff != 'none'):
                    x_vals=cur_track[0:len_cutoff,self.tracks_x_col]
                    y_vals=cur_track[0:len_cutoff,self.tracks_y_col]
                else:
                    x_vals = cur_track[:, self.tracks_x_col]
                    y_vals = cur_track[:, self.tracks_y_col]

                if(error_term):
                    D = self.D_linfits_E[self.D_linfits_E[:, self.D_lin_E_id_col] == id][0, self.D_lin_E_D_col]
                else:
                    D = self.D_linfits[self.D_linfits[:,self.D_lin_id_col]==id][0,self.D_lin_D_col]

                if(remove_tracks and (D < min_Deff or D > max_Deff)):
                    continue
                else:
                    if (D < min_Deff):
                        D = min_Deff
                    if (D > max_Deff):
                        D = max_Deff

                    show_color=D/max_Deff
                    ax.plot(x_vals, y_vals, '-', color=cm.jet(show_color), linewidth=lw)

    def rainbow_tracks(self, img_file, output_file, len_cutoff='none', remove_tracks=False, min_Deff=0.01, max_Deff=2, lw=0.1):
        # given img file, plots tracks on img
        bk_img = io.imread(img_file)

        # plot figure to draw tracks with image in background
        fig = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.imshow(bk_img, cmap="gray")

        self.save_tracks_to_img(ax, len_cutoff=len_cutoff, remove_tracks=remove_tracks, min_Deff=min_Deff, max_Deff=max_Deff, lw=lw)

        # save figure of lines plotted on bk_img
        fig.tight_layout()
        fig.savefig(output_file, dpi=500)  # 1000
        plt.close(fig)

    def rainbow_tracks_ss(self, img_file, output_file, min_ss=0, max_ss=5, lw=0.1):
        # min_ss/max_ss are in microns (not pixels)
        # given img file, plots tracks on img
        bk_img = io.imread(img_file)

        # plot figure to draw tracks with image in background
        fig = plt.figure(figsize=(bk_img.shape[1] / 100, bk_img.shape[0] / 100), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.imshow(bk_img, cmap="gray")

        self.save_tracks_to_img_ss(ax, min_ss=min_ss, max_ss=max_ss, lw=lw)

        # save figure of lines plotted on bk_img
        fig.tight_layout()
        fig.savefig(output_file, dpi=500)  # 1000
        plt.close(fig)

