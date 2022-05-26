from matplotlib import cm
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

class rainbow_tracks:
    def __init__(self, ):

        # minimum and maximum diffusion coefficient for display
        self.min_D = 0
        self.max_D = np.inf

        # min/max diffusion coefficient corresponding to blue (min) and red (max)
        self.blue_D=0
        self.red_D=2

        # min/max step size corresponding to blue (min) and red (max)
        # in microns
        self.blue_ss=0
        self.red_ss=1

        # line width for rainbow track - in points
        self.line_width=0.1

        # beginning of time labeling is start of track (if True)
        # beginning of time labeling is start of movie (if False)
        self.time_label_by_track_start=True

        self.tracks_id_col=0
        self.tracks_x_col=1
        self.tracks_y_col=2
        self.tracks_color_val_col=3

        self.DPI = 300
        self.figsize_div=100

        # TODO - add rainbow tracks for... relative angle - 0 to 180 deg
        # TODO - time colored rainbow tracks

    def plot_diffusion(self, img_file, track_data, diff_data, output_file, min_length=None):

        self.plot_tracks(img_file,
                         track_data,
                         diff_data,
                         output_file,
                         min_length=min_length,
                         filter_min=self.min_D,
                         filter_max=self.max_D,
                         blue_val=self.blue_D,
                         red_val=self.red_D)

    def plot_step_sizes(self, img_file, track_data, output_file, min_length=3):

        self.plot_tracks_multi(img_file,
                         track_data,
                         output_file,
                         min_length=min_length,
                         blue_val=self.blue_ss,
                         red_val=self.red_ss)

    # Plots a color for each step in the track
    # value taken from column index "self.tracks_color_val_col" of track_data
    # x/y positions are in track_data (self.tracks_x_col, self.tracks_y_col)
    # Used to plot tracks colored by step size
    def plot_tracks_multi(self,
                          img_file,
                          track_data,
                          output_file,
                          min_length,
                          blue_val,
                          red_val):

        bk_img = io.imread(img_file)
        fig = plt.figure(figsize=(bk_img.shape[1] / self.figsize_div,
                                  bk_img.shape[0] / self.figsize_div),
                         dpi=self.DPI)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.imshow(bk_img, cmap="gray")

        # min_ss/max_ss are in microns (not pixels)
        ids = np.unique(track_data[:, self.tracks_id_col])
        for id in ids:
            cur_track = track_data[track_data[:, self.tracks_id_col] == id]
            if (min_length == None or len(cur_track) >= min_length):

                for step_i in range(1, len(cur_track), 1):
                    cur_ss = cur_track[step_i, self.tracks_color_val_col]
                    if (cur_ss < blue_val):
                        cur_ss = blue_val
                    if (cur_ss > red_val):
                        cur_ss = red_val

                    show_color = cur_ss / red_val

                    ax.plot([cur_track[step_i - 1, self.tracks_x_col], cur_track[step_i, self.tracks_x_col]],
                            [cur_track[step_i - 1, self.tracks_y_col], cur_track[step_i, self.tracks_y_col]],
                            '-', color=cm.jet(show_color), linewidth=self.line_width)

        fig.tight_layout()
        fig.savefig(output_file, dpi=self.DPI)
        plt.close(fig)

    # Plots a single color, value taken from "color_data", for each track
    # x/y positions are in track_data (self.tracks_x_col, self.tracks_y_col)
    # Used to plot tracks colored by diffusion coefficient
    def plot_tracks(self,
                    img_file,
                    track_data,
                    color_data,
                    output_file,
                    min_length,
                    filter_min,
                    filter_max,
                    blue_val,
                    red_val):

        bk_img = io.imread(img_file)
        fig = plt.figure(figsize=(bk_img.shape[1] / self.figsize_div,
                                  bk_img.shape[0] / self.figsize_div),
                         dpi=self.DPI)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.imshow(bk_img, cmap="gray")
        for i in range(len(color_data)):
            id=int(color_data[i][0])
            cur_track = track_data[track_data[:, self.tracks_id_col] == id]
            if (min_length == None or len(cur_track) >= min_length):
                x_vals = cur_track[:, self.tracks_x_col]
                y_vals = cur_track[:, self.tracks_y_col]
                color_val=color_data[i][1]
                if (color_val >= filter_min or color_val <= filter_max):
                    if (color_val < blue_val):
                        color_val = blue_val
                    elif (color_val > red_val):
                        color_val = red_val
                    show_color = color_val / red_val
                    ax.plot(x_vals, y_vals, '-', color=cm.jet(show_color), linewidth=self.line_width)
        fig.tight_layout()
        fig.savefig(output_file, dpi=self.DPI)
        plt.close(fig)

    def plot_time(self,
                  img_file,
                  track_data,
                  output_file,
                  reverse_coords=False,
                  min_length=3):

        bk_img = io.imread(img_file)
        fig = plt.figure(figsize=(bk_img.shape[1] / self.figsize_div,
                                  bk_img.shape[0] / self.figsize_div),
                         dpi=self.DPI)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.imshow(bk_img, cmap="gray")

        ids = np.unique(track_data[:, self.tracks_id_col])
        for id in ids:
            cur_track = track_data[track_data[:, self.tracks_id_col] == id]
            if (min_length == None or len(cur_track) >= min_length):
                max_step = len(cur_track)
                for step_i in range(1, max_step, 1):
                    show_color = step_i / max_step
                    if (reverse_coords):
                        ax.plot(
                            [cur_track[step_i - 1, self.tracks_y_col], cur_track[step_i, self.tracks_y_col]],
                            [cur_track[step_i - 1, self.tracks_x_col], cur_track[step_i, self.tracks_x_col]],
                            '-', color=cm.jet(show_color), linewidth=self.line_width)

                    else:
                        ax.plot(
                            [cur_track[step_i - 1, self.tracks_x_col], cur_track[step_i, self.tracks_x_col]],
                            [cur_track[step_i - 1, self.tracks_y_col], cur_track[step_i, self.tracks_y_col]],
                            '-', color=cm.jet(show_color), linewidth=self.line_width)



        if(self.time_label_by_track_start):
            for id in ids:
                cur_track = self.tracks[self.tracks[:, self.tracks_id_col] == id]
                max_step = len(cur_track)
                for step_i in range(1, max_step, 1):

                    show_color = step_i / max_step

                    if (reverse_coords):
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
        else:
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



