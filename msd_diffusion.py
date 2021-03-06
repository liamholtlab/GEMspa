from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
import os
from skimage import img_as_ubyte, io

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
        self.track_lengths = np.asarray([])
        self.track_step_sizes=np.asarray([])
        self.save_dir = '.'

        self.time_step = 0.010 #time between frames, in seconds
        self.micron_per_px = 0.11
        self.min_track_len_linfit=11
        self.min_track_len_step_size = 3
        self.max_tlag_step_size=3
        self.track_len_cutoff_linfit=11
        self.perc_tlag_linfit=25
        self.use_perc_tlag_linfit=False
        self.initial_guess_linfit=0.2
        self.initial_guess_aexp=1

        self.tracks_num_cols=4
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
        self.D_lin_Dlen_col = 6

    def set_track_data(self, track_data):
        self.tracks=track_data
        self.msd_tracks = np.asarray([])
        self.D_linfits = np.asarray([])
        self.fill_track_lengths()
        self.fill_track_sizes()
        self.ensemble_average = np.asarray([])

    def msd2d(self, x, y):

        shifts = np.arange(1, len(x), 1)
        MSD = np.zeros(shifts.size)
        MSD_std = np.zeros(shifts.size)

        for i, shift in enumerate(shifts):
            sum_diffs_sq = np.square(x[shift:] - x[:-shift]) + np.square(y[shift:] - y[:-shift])
            MSD[i] = np.mean(sum_diffs_sq)
            MSD_std[i] = np.std(sum_diffs_sq)

        return MSD, MSD_std

    def fill_track_lengths(self):
        # fill track length array with the track lengths
        ids = np.unique(self.tracks[:, self.tracks_id_col])
        self.track_lengths = np.zeros((len(ids), 2))
        for i,id in enumerate(ids):
            cur_track = self.tracks[np.where(self.tracks[:, self.tracks_id_col] == id)]
            self.track_lengths[i,0] = id
            self.track_lengths[i,1] = len(cur_track)

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
            return

        print("SS/Angles number of tracks:", len(ids))
        track_lens=self.track_lengths[self.track_lengths[:,1] >= self.min_track_len_step_size][:,1]

        #rows correspond to t-lag==1,2,3,4,5, columns list the step sizes
        tlag1_dim_steps = int(np.sum(track_lens - 1))
        self.step_sizes = np.empty((self.max_tlag_step_size, tlag1_dim_steps,))
        self.step_sizes.fill(np.nan)

        tlag1_dim_angles = int(np.sum(track_lens - 2))
        self.angles = np.empty((self.max_tlag_step_size, tlag1_dim_angles,))
        self.angles.fill(np.nan)

        #used for angle autocorrelation
        #self.angles_tlag1 = np.empty((len(ids),max_track_len-2))
        #self.angles_tlag1.fill(np.nan)

        start_arr=np.zeros((self.max_tlag_step_size,), dtype='int')
        angle_start_arr = np.zeros((self.max_tlag_step_size,), dtype='int')
        for id_i,id in enumerate(ids):
            #if(id_i==13):
            #    print("")
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
                start_arr[i] += len(sum_diffs_sq)

                if(tlag <= max_num_angle_tlags): # angles for this tlag
                    # relative angle: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3856831/
                    vecs = np.column_stack((x_shifts, y_shifts))
                    theta=np.zeros(len(list(range(0, len(vecs)-tlag, tlag))))
                    for theta_i,vec_i in enumerate(range(0, len(vecs)-tlag, tlag)): # check range with longer track
                        if(np.linalg.norm(vecs[vec_i]) == 0 or np.linalg.norm(vecs[vec_i+tlag]) == 0):
                            print("norm of vec is 0: id=", id)
                        theta[theta_i] = np.rad2deg(np.arccos(np.dot(vecs[vec_i],vecs[vec_i+tlag]) / (np.linalg.norm(vecs[vec_i]) * np.linalg.norm(vecs[vec_i+tlag]))))
                    self.angles[i][angle_start_arr[i]:angle_start_arr[i]+len(theta)] = theta
                    #if(i==0):
                    #   self.angles_tlag1[id_i][:len(theta)]=theta
                    angle_start_arr[i] += len(theta)

    def msd_all_tracks(self):
        # for each track, do MSD calculation
        ids = np.unique(self.tracks[:,self.tracks_id_col])
        print("MSD number of tracks:", len(ids))
        self.msd_tracks = np.zeros((len(self.tracks),self.msd_num_cols), )
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

    def calculate_ensemble_average(self):
        # average the MSD at each t-lag, weighted by number of points in the MSD average
        # did not use weighted mean, instead cut off tracks < 11 in length, and
        # also only continue with larger tlags while the number of points to average >= 5
        valid_tracks = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_len_col] >= (self.min_track_len_linfit - 1))]
        min_MSD_values_for_mean=10
        max_tlag=int(np.max(valid_tracks[:,self.msd_len_col]))
        ensemble_average=[]
        for tlag in range(1,max_tlag+1,1):
            # gather all data for current tlag
            tlag_time=tlag*self.time_step
            cur_tlag_MSDs=valid_tracks[valid_tracks[:, self.msd_t_col] == tlag_time][:,self.msd_msd_col]

            #cur_tlag_weights=valid_tracks[valid_tracks[:, self.msd_t_col] == tlag_time][:,self.msd_len_col]
            #cur_tlag_weights=cur_tlag_weights-tlag # not using the weights for now...
            if(len(cur_tlag_MSDs)>=min_MSD_values_for_mean):
                ensemble_average.append([tlag_time, np.mean(cur_tlag_MSDs), np.std(cur_tlag_MSDs)])
            else:
                break
        self.ensemble_average=np.asarray(ensemble_average)

    # def calculate_ensemble_average(self):
    #     # average the MSD at each t-lag, weighted by number of points in the MSD average
    #     # did not use weighted mean, instead cut off tracks < 11 in length, and
    #     # also only continue with larger tlags while the number of points to average >= 5
    #     min_MSD_values_for_mean = 2
    #     max_tlag=int(np.max(self.msd_tracks[:,self.msd_len_col]))
    #     ensemble_average=[]
    #     for tlag in range(1,max_tlag+1,1):
    #         # gather all data for current tlag
    #         tlag_time=tlag*self.time_step
    #         cur_tlag_MSDs=self.msd_tracks[self.msd_tracks[:, self.msd_t_col] == tlag_time][:,self.msd_msd_col]
    #
    #         cur_tlag_weights=self.msd_tracks[self.msd_tracks[:, self.msd_t_col] == tlag_time][:,self.msd_len_col]
    #         cur_tlag_weights=cur_tlag_weights-(tlag-1)
    #         if(np.min(cur_tlag_weights) <= 0):
    #             print("ERROR")
    #         if (len(cur_tlag_MSDs) >= min_MSD_values_for_mean):
    #             wm=np.average(cur_tlag_MSDs, weights=cur_tlag_weights)
    #             variance = np.average((cur_tlag_MSDs - wm) ** 2, weights=cur_tlag_weights)
    #             ensemble_average.append([tlag_time, wm, np.sqrt(variance)])
    #         else:
    #             break
    #
    #     self.ensemble_average=np.asarray(ensemble_average)

    # def fit_msd_ensemble_alpha(self):
    #     # uses ensemble average to fit for anomolous exponent
    #     if (len(self.ensemble_average) == 0):
    #         return ()
    #
    #     def powerlaw_fn(x, a, b):
    #         return 4 * a * (x ** b)
    #
    #     powerlaw_fn_v = np.vectorize(powerlaw_fn)
    #
    #     ##### Correct way
    #     popt, pcov = curve_fit(powerlaw_fn, self.ensemble_average[:, 0], self.ensemble_average[:, 1],
    #                            p0=[self.initial_guess_linfit, self.initial_guess_aexp])
    #     residuals = self.ensemble_average[:, 1] - powerlaw_fn_v(self.ensemble_average[:, 0], popt[0], popt[1])
    #     ss_res = np.sum(residuals ** 2)
    #     rmse = np.mean(residuals ** 2) ** 0.5
    #     ss_tot = np.sum((self.ensemble_average[:, 1] - np.mean(self.ensemble_average[:, 1])) ** 2)
    #
    #     r_squared = 1 - (ss_res / ss_tot)
    #
    #     self.anomolous_fit_rsq = r_squared
    #     self.anomolous_fit_rmse = rmse
    #     self.anomolous_fit_K = popt[0]
    #     self.anomolous_fit_alpha = popt[1]
    #     self.anomolous_fit_errs = np.sqrt(np.diag(pcov))  # one standard deviation errors on the parameters

    def fit_msd_ensemble_alpha(self):
        #MSD(t) = L * t ^ a
        #L = 4K
        #log(MSD) = a * log(t) + log(L)
        #FIT to straight line

        # uses ensemble average to fit for anomolous exponent
        if (len(self.ensemble_average) == 0):
            return ()

        def linear_fn(x, m, b):
            return m * x + b

        linear_fn_v = np.vectorize(linear_fn)

        popt, pcov = curve_fit(linear_fn, np.log(self.ensemble_average[:, 0]), np.log(self.ensemble_average[:, 1]),
                               p0=[self.initial_guess_aexp, np.log(4*self.initial_guess_linfit)])

        residuals = self.ensemble_average[:, 1] - linear_fn_v(self.ensemble_average[:, 0], popt[0], popt[1])
        ss_res = np.sum(residuals ** 2)
        rmse = np.mean(residuals ** 2) ** 0.5
        ss_tot = np.sum((self.ensemble_average[:, 1] - np.mean(self.ensemble_average[:, 1])) ** 2)

        r_squared = 1 - (ss_res / ss_tot)

        self.anomolous_fit_rsq = r_squared
        self.anomolous_fit_rmse = rmse
        self.anomolous_fit_K = np.exp(popt[1])/4
        self.anomolous_fit_alpha = popt[0]
        self.anomolous_fit_errs = np.sqrt(np.diag(pcov))  # one standard deviation errors on the parameters

    def fit_msd_ensemble(self):
        # uses ensemble average to fit for Deff
        if (len(self.ensemble_average) == 0):
            return ()

        def linear_fn(x, a):
            return 4 * a * x

        linear_fn_v = np.vectorize(linear_fn)

        popt, pcov = curve_fit(linear_fn, self.ensemble_average[:,0], self.ensemble_average[:,1],
                               p0=[self.initial_guess_linfit,])
        residuals = self.ensemble_average[:,1] - linear_fn_v(self.ensemble_average[:,0], popt[0])
        ss_res = np.sum(residuals ** 2)
        rmse = np.mean(residuals ** 2) ** 0.5
        ss_tot = np.sum((self.ensemble_average[:,1] - np.mean(self.ensemble_average[:,1]))**2)

        r_squared = 1 - (ss_res / ss_tot)

        self.ensemble_fit_rsq = r_squared
        self.ensemble_fit_rmse = rmse
        self.ensemble_fit_D=popt[0]
        self.ensemble_fit_err=np.sqrt(np.diag(pcov)) #one standard deviation error on the parameter

    def plot_msd_ensemble(self, file_name="msd_ensemble.pdf", fit_line=False):

        def linear_fn(x, a):
            return 4 * a * x
        linear_fn_v = np.vectorize(linear_fn)

        def linear_fn2(x, m, b):
            return m * x + b

        linear_fn_v2 = np.vectorize(linear_fn2)

        #linear scale
        plt.plot(self.ensemble_average[:, 0], self.ensemble_average[:, 1], color='black')
        plt.xlabel('t-lag (s)')
        plt.ylabel('MSD (microns^2)')
        plt.xscale('linear')
        plt.yscale('linear')
        if(fit_line):
            y_vals_from_fit = linear_fn_v(self.ensemble_average[:, 0], msd_diff.ensemble_fit_D)
            plt.plot(self.ensemble_average[:, 0], y_vals_from_fit, color='red')

        lower_bound = self.ensemble_average[:, 1] - self.ensemble_average[:, 2]
        lower_bound[lower_bound < 0] = 0
        plt.fill_between(self.ensemble_average[:, 0],
                         lower_bound,
                         self.ensemble_average[:, 1] + self.ensemble_average[:, 2], color='gray', alpha=0.8)
        plt.savefig(self.save_dir + '/' + file_name)
        plt.clf()

        # loglog plot
        plt.plot(np.log(self.ensemble_average[:, 0]), np.log(self.ensemble_average[:, 1]), color='black')

        if(fit_line): # np.exp(popt[1])/4
            y_vals_from_fit = linear_fn_v2(np.log(self.ensemble_average[:, 0]), msd_diff.anomolous_fit_alpha, np.log(4*self.anomolous_fit_K))
            plt.plot(np.log(self.ensemble_average[:, 0]), y_vals_from_fit, color='red')

            #y_vals_from_fit = linear_fn_v2(np.log(self.ensemble_average[:, 0]), .90, np.log(4 * .40))
            #plt.plot(np.log(self.ensemble_average[:, 0]), y_vals_from_fit, color='pink')


        plt.xlabel('log[t-lag (s)]')
        plt.ylabel('log[MSD (microns^2)]')

        (file_root, ext) = os.path.splitext(file_name)
        plt.savefig(self.save_dir + '/' + file_root + '-loglog' + ext)
        plt.clf()

    def fit_msd(self, type="brownian"):
        # fit MSD curve to get Diffusion coefficient
        # filter for tracks > min-length

        if(len(self.msd_tracks) == 0):
            return ()

        if (type == 'brownian'):
            #msd_len is one less than the track len
            valid_tracks = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_len_col] >= (self.min_track_len_linfit-1))]

            ids = np.unique(valid_tracks[:,self.msd_id_col])
            print("Deff number of tracks:", len(ids))
            self.D_linfits = np.zeros((len(ids),self.D_lin_num_cols,))
            for i,id in enumerate(ids):
                cur_track = valid_tracks[np.where(valid_tracks[:, self.msd_id_col] == id)]
                if(self.use_perc_tlag_linfit):
                    stop = int(cur_track[0][self.msd_len_col] * (self.perc_tlag_linfit/100))+1
                else:
                    stop = self.track_len_cutoff_linfit

                def linear_fn(x, a):
                    return 4 * a * x
                linear_fn_v = np.vectorize(linear_fn)

                ##### Correct way
                popt, pcov = curve_fit(linear_fn, cur_track[1:stop,self.msd_t_col], cur_track[1:stop,self.msd_msd_col],
                                       p0=[self.initial_guess_linfit,])
                residuals = cur_track[1:stop,self.msd_msd_col] - linear_fn_v(cur_track[1:stop,self.msd_t_col], popt[0])
                ss_res = np.sum(residuals ** 2)
                rmse = np.mean(residuals**2)**0.5
                ss_tot = np.sum((cur_track[1:stop,self.msd_msd_col] - np.mean(cur_track[1:stop,self.msd_msd_col])) ** 2)
                #####

                ##### Matches matlab script results
                # popt, pcov = curve_fit(linear_fn, cur_track[0:stop-1, self.msd_t_col],cur_track[1:stop, self.msd_msd_col],
                #                        p0=[self.initial_guess_linfit, ])
                # residuals = cur_track[1:stop, self.msd_msd_col] - linear_fn_v(cur_track[0:stop-1, self.msd_t_col],popt[0])
                # ss_res = np.sum(residuals ** 2)
                # rmse = np.mean(residuals ** 2) ** 0.5
                # ss_tot = np.sum((cur_track[1:stop, self.msd_msd_col] - np.mean(cur_track[1:stop, self.msd_msd_col])) ** 2)
                #####

                r_squared = 1 - (ss_res / ss_tot)

                D = popt[0]
                perr=np.sqrt(np.diag(pcov))

                self.D_linfits[i][self.D_lin_id_col]=id
                self.D_linfits[i][self.D_lin_D_col]=D
                self.D_linfits[i][self.D_lin_err_col]=perr
                self.D_linfits[i][self.D_lin_rsq_col] = r_squared
                self.D_linfits[i][self.D_lin_rmse_col] = rmse
                self.D_linfits[i][self.D_lin_len_col] = cur_track[0][self.msd_len_col]
                self.D_linfits[i][self.D_lin_Dlen_col] = stop-1
        else:
            # no other type supported currently
            print("Error: fit_msd: unknown type of fit.")

    def save_fit_data(self, file_name="fit_results.txt"):
        df = pd.DataFrame(self.D_linfits)
        df.rename(columns={self.D_lin_id_col: 'id', self.D_lin_D_col: 'D', self.D_lin_err_col: 'err',
                           self.D_lin_rsq_col: 'r_sq', self.D_lin_rmse_col: 'rmse', self.D_lin_len_col: 'track_len',
                           self.D_lin_Dlen_col: 'D_track_len'}, inplace=True)
        df.to_csv(self.save_dir + '/' + file_name, sep='\t', index=False)
        return df

    def save_msd_data(self, file_name="msd_results.txt"):
        df = pd.DataFrame(self.msd_tracks)
        df.rename(columns={self.msd_id_col: 'id', self.msd_t_col: 't', self.msd_frame_col: 'frame',
                           self.msd_x_col: 'x', self.msd_y_col: 'y', self.msd_msd_col: 'MSD',
                           self.msd_std_col: 'MSD_stdev', self.msd_len_col: 'MSD_len'}, inplace=True)
        df.to_csv(self.save_dir + '/' + file_name, sep='\t', index=False)
        return df

    def plot_msd_curves(self,file_name="msd_all.pdf", max_tracks=50, ymax=-1, xmax=-1, min_track_len=0, fit_line=False):

        def linear_fn(x, a):
            return 4 * a * x

        linear_fn_v = np.vectorize(linear_fn)

        valid_tracks = self.msd_tracks[np.where(self.msd_tracks[:, self.msd_len_col] >= (min_track_len - 1))]
        ids = np.unique(valid_tracks[:, self.msd_id_col])
        x_range=[]
        for i, id in enumerate(ids):
            cur_track = valid_tracks[np.where(valid_tracks[:, self.msd_id_col] == id)]
            plt.plot(cur_track[:,self.msd_t_col], cur_track[:,self.msd_msd_col], linewidth=".5")
            if(len(cur_track[:,self.msd_t_col]) > len(x_range)):
                x_range=cur_track[:,self.msd_t_col]

            if (max_tracks != 0 and i+1 >= max_tracks):
                break

        if(fit_line):
            y_vals_from_fit=linear_fn_v(x_range, np.median(self.D_linfits[:,self.D_lin_D_col]))
            plt.plot(x_range, y_vals_from_fit, color='black')
        plt.xlabel('t-lag (s)')
        plt.ylabel('MSD (microns^2)')
        if(ymax>0):
            plt.ylim(0,ymax)
        if (xmax > 0):
            plt.xlim(0, xmax)
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
        to_plot = self.track_lengths[np.where(self.track_lengths[:,1] >= (self.track_len_cutoff_linfit-1))][:,1]
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
                           self.D_lin_Dlen_col: 'D_track_len'}, inplace=True)


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

    def save_tracks_to_img_clr(self, ax, lw=0.1, color='blue'):
        ids = np.unique(self.tracks[:, self.tracks_id_col])
        for id in ids:
            cur_track = self.tracks[self.tracks[:, self.tracks_id_col] == id]
            if (len(cur_track) >= self.min_track_len_step_size):
                ax.plot(cur_track[:,self.tracks_x_col],cur_track[:,self.tracks_y_col],'-',color=color,linewidth=lw)

    def save_tracks_to_img(self, ax, len_cutoff='none', remove_tracks=False, min_Deff=0.01, max_Deff=2, lw=0.1):
        if(len_cutoff == 'default'):
            len_cutoff=self.track_len_cutoff_linfit

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


    # def save_angle_autocorr(self, file_name="angles_autocorr.pdf"):
    #     from statsmodels.graphics.tsaplots import plot_acf
    #     from statsmodels.tsa.stattools import acf
    #
    #     mean_acf_y_vals=np.zeros(len(self.angles_tlag1[0]))
    #     for i in range(len(self.angles_tlag1)):
    #         mean_acf_y_vals += acf(self.angles_tlag1[i])
    #     mean_acf_y_vals /= len(self.angles_tlag1)
    #
    #     plt.scatter(range(0, len(mean_acf_y_vals)), mean_acf_y_vals, marker='.', color="blue")
    #     plt.bar(x=range(0,len(mean_acf_y_vals)), height=mean_acf_y_vals, width=0.05, color="black")
    #     plt.plot([-0.5,len(mean_acf_y_vals)],[0,0],color='blue')
    #     plt.xlim(-0.5,len(mean_acf_y_vals))
    #     plt.ylim(-1,1.25)
    #     plt.savefig(self.save_dir + '/' + file_name)
    #     plt.clf()
    #
    #     # plot_acf(self.angles_tlag1[0])
    #     # plt.savefig(self.save_dir + '/0_' + file_name)
    #     # plt.clf()


#
# test1=False
test2=False
# test3=False
#
# from nd2reader import ND2Reader
# def read_movie_metadata(file_name):
#     try:
#         with ND2Reader(file_name) as images:
#             steps = images.timesteps[1:] - images.timesteps[:-1]
#             steps = np.round(steps, 0)
#             #convert steps from ms to s
#             steps=steps/1000
#             microns_per_pixel=images.metadata['pixel_microns']
#             return (microns_per_pixel,steps)
#     except FileNotFoundError as e:
#         return None
#
# def filter_tracks(track_data, min_len, time_steps, min_resolution):
#     correct_ts = np.min(time_steps)
#     traj_ids = np.unique(track_data['Trajectory'])
#     track_data_cp=track_data.copy()
#     track_data_cp['Remove']=True
#     for traj_id in traj_ids:
#         cur_track = track_data[track_data['Trajectory']==traj_id].copy()
#         if(len(cur_track) >= min_len):
#             new_seg=True
#             longest_seg_start = cur_track.index[0]
#             longest_seg_len = 0
#             for row_i, row in enumerate(cur_track.iterrows()):
#                 if(row_i > 0):
#                     if(new_seg):
#                         cur_seg_start=row[0]-1
#                         cur_seg_len=0
#                         new_seg=False
#
#                     fr=row[1]['Frame']
#                     cur_ts = time_steps[int(fr-1)]
#                     if((cur_ts - correct_ts) <= min_resolution):
#                         #keep going
#                         cur_seg_len+=1
#                     else:
#                         if(cur_seg_len > longest_seg_len):
#                             longest_seg_len=cur_seg_len
#                             longest_seg_start = cur_seg_start
#                             new_seg=True
#             if(not new_seg):
#                 if (cur_seg_len > longest_seg_len):
#                     longest_seg_len = cur_seg_len
#                     longest_seg_start = cur_seg_start
#
#             if(longest_seg_len >= min_len):
#                 # add track segment to new DF - check # 10
#                 track_data_cp.loc[longest_seg_start:longest_seg_start+longest_seg_len,'Remove']=False
#
#     track_data_cp = track_data_cp[track_data_cp['Remove']==False]
#     track_data_cp.index=range(len(track_data_cp))
#     track_data_cp = track_data_cp.drop('Remove', axis=1)
#     return track_data_cp
#
# if(test3):
#     dir_results = "/Users/sarahkeegan/Dropbox/mac_files/holtlab/data_and_results/Ying GEM/For MOISAIC analysis/testing_results"
#     csv_file="/Users/sarahkeegan/Dropbox/mac_files/holtlab/data_and_results/Ying GEM/For MOISAIC analysis/Traj_rim15 -C stavation 4h-1.nd2_crop.csv"
#     movie_file="/Users/sarahkeegan/Dropbox/mac_files/holtlab/data_and_results/Ying GEM/rim15 -C stavation 4h-1.nd2"
#
#     ret = read_movie_metadata(movie_file)
#
#     track_data_df = pd.read_csv(csv_file)
#     track_data_df = track_data_df[['Trajectory', 'Frame', 'x', 'y']]
#
#     if(len(np.unique(ret[1])) > 0):
#         track_data_filtered = filter_tracks(track_data_df, 11, ret[1], 0.005)
#         track_data_filtered.to_csv(dir_results + "/" + os.path.split(csv_file)[1][:-4] + "_filtered.csv")
#         track_data = track_data_filtered.to_numpy()
#     else:
#         track_data = track_data_df.to_numpy()
#
#     msd_diff_obj = msd_diffusion()
#     msd_diff_obj.save_dir = dir_results
#
#     print("Base time step:", np.min(ret[1]))
#     print("Micron per pixel:", ret[0])
#     msd_diff_obj.time_step = np.min(ret[1])
#     msd_diff_obj.micron_per_px = ret[0]
#
#     msd_diff_obj.set_track_data(track_data)
#     msd_diff_obj.step_sizes_and_angles()
#
#     msd_diff_obj.msd_all_tracks()
#     msd_diff_obj.fit_msd()
#
#     msd_diff_obj.save_msd_data(file_name=os.path.split(csv_file)[1][:-4] + "_MSD.txt")
#     df=msd_diff_obj.save_fit_data(file_name=os.path.split(csv_file)[1][:-4] + "_Dlin.txt")
#     msd_diff_obj.save_step_sizes(file_name=os.path.split(csv_file)[1][:-4] + "_step_sizes.txt")
#
#     print(np.median(df['D']))
#
# if(test1):
#     dir_="/Users/sarahkeegan/Dropbox/mac_files/holtlab/data_and_results/GEMs/for_presentation"
#
#     #file_="Traj_GBC1_3W_0min_GEM_6_cyto.tif.csv"
#     #suffix = '3W_0hr'  # med(D) = 0.54
#     #ym = -1
#
#     #file_ = "Traj_GBC1_3W_5_hour_GEM_4_cyto.tif.csv"
#     #suffix='3W_5hr' # med(D) = 0.04
#     #ym = -1
#
#     file_="GBC1_4_cyto.csv"
#     suffix = 'CTRL' # med(D) = 0.18
#     ym = 0.4
#
#     track_data_df = pd.read_csv(dir_ + '/' + file_)
#     track_data_df = track_data_df[['Trajectory', 'Frame', 'x', 'y']]
#     track_data = track_data_df.to_numpy()
#
#     msd_diff = msd_diffusion()
#     msd_diff.set_track_data(track_data)
#     msd_diff.step_sizes_and_angles()
#
#     msd_diff.msd_all_tracks()
#     msd_diff.fit_msd()
#
#     msd_diff.save_dir = dir_ + '/results_'+suffix
#
#     msd_diff.save_angles()
#     msd_diff.save_step_sizes()
#
#     msd_diff.save_msd_data()
#     df=msd_diff.save_fit_data()
#     print(np.median(df['D']))
#     msd_diff.save_track_length_hist()
#     msd_diff.save_step_size_hist("step_sizes1.pdf",1)
#
#
#     msd_diff.save_step_size_hist("step_sizes2.pdf", 2)
#     msd_diff.save_step_size_hist("step_sizes3.pdf", 3)
#     # msd_diff.save_step_size_hist("step_sizes4.pdf", 4)
#     # msd_diff.save_step_size_hist("step_sizes5.pdf", 5)
#
#     #msd_diff.save_angle_hist("angles1.pdf", 1)
#     #msd_diff.save_angle_hist("angles2.pdf", 2)
#     #msd_diff.save_angle_hist("angles3.pdf", 3)
#
#     #msd_diff.save_angle_hist("angles4.pdf", 4)
#     #msd_diff.save_angle_hist("angles5.pdf", 5)
#
#     msd_diff.plot_msd_curves(ymax=ym, max_tracks=30)
#
#     msd_diff.save_D_histogram("Deff.pdf")
#
if(test2):
    dir_="/Volumes/Seagate Backup Plus Drive/holtlab/data_and_results/tamas-20201218_HeLa_hPNE_nucPfV_NPM1_clones/tracks/"
    file_name='Traj_02_WT_hPNE_nucPfV_010_01.csv'

    track_data_df = pd.read_csv(dir_ + '/' + file_name)
    track_data_df = track_data_df[['Trajectory', 'Frame', 'x', 'y']]
    track_data = track_data_df.to_numpy()

    msd_diff = msd_diffusion()
    msd_diff.micron_per_px=0.1527
    msd_diff.time_step=0.010
    msd_diff.set_track_data(track_data)
    msd_diff.save_dir = dir_ + '/results'

    # msd_diff.step_sizes_and_angles()
    # msd_diff.save_angle_hist(tlag=1)
    # msd_diff.save_angle_hist(tlag=2)
    # msd_diff.save_angle_hist(tlag=3)

    msd_diff.msd_all_tracks()
    msd_diff.calculate_ensemble_average()


    msd_diff.fit_msd()
    msd_diff.plot_msd_curves(min_track_len=11, ymax=1, xmax=0.3, max_tracks=0, fit_line=True)

    msd_diff.fit_msd_ensemble()
    print(msd_diff.ensemble_fit_rsq)
    print(msd_diff.ensemble_fit_rmse)
    print(msd_diff.ensemble_fit_D)
    print(msd_diff.ensemble_fit_err)

    print("")
    msd_diff.fit_msd_ensemble_alpha()
    print(msd_diff.anomolous_fit_rsq)
    print(msd_diff.anomolous_fit_rmse)
    print(msd_diff.anomolous_fit_K)
    print(msd_diff.anomolous_fit_alpha)
    print(msd_diff.anomolous_fit_errs)

    msd_diff.plot_msd_ensemble(fit_line=True)

    print("")
    msd_diff.save_msd_data()
    df = msd_diff.save_fit_data()
    print(np.median(df['D']))











