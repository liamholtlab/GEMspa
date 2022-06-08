
import rainbow_tracks as rt
import pandas as pd
import numpy as np
import glob
import wx
import wx.grid as gridlib
import os
import datetime
import wx.lib.mixins.inspection

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
                        micron_per_px,
                        max_D,
                        max_ss,
                        lw,
                        min_track_len,
                        rt_D,
                        rt_ss,
                        rt_time):

    # input_dir is the directory where the GEMSpa output files are located
    # output_dir is the directory to SAVE the tif files (rainbow tracks drawn on images)
    # image locations are read from the GEMSpa output files
    # if output_dir == '', then the rainbow tracks files will be saved to input_dir

    if(output_dir == ''):
        output_dir = input_dir

    rainbow_tr = rt.rainbow_tracks()
    rainbow_tr.tracks_id_col = 0
    rainbow_tr.tracks_frame_col=1
    rainbow_tr.tracks_x_col = 2
    rainbow_tr.tracks_y_col = 3
    rainbow_tr.tracks_color_val_col = 4
    rainbow_tr.time_label_by_track_start = True

    rainbow_tr.red_D=max_D
    rainbow_tr.red_ss=max_ss
    rainbow_tr.line_width=lw

    ret_str = ""
    output_file_count=0

    # read the all_data.txt file
    all_data_df = pd.read_csv(os.path.join(input_dir, "all_data.txt"),sep='\t')

    #print("")
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
            print(img_file)
            if(len(tracks_df)>0):
                if(rt_D):
                    out_file = os.path.split(img_file)[1][:-4] + '_tracks_Deff.tif'
                    rainbow_tr.plot_diffusion(img_file,
                                          np.asarray(tracks_df),
                                          np.asarray(cur_all_data_df[['Trajectory','D']]),
                                          output_dir+'/'+out_file)
                if(rt_ss):
                    out_file = os.path.split(img_file)[1][:-4] + '_tracks_ss.tif'
                    tracks_arr=fill_track_sizes(np.asarray(tracks_df), micron_per_px)
                    rainbow_tr.plot_step_sizes(img_file,
                                           tracks_arr,
                                           output_dir + '/' + out_file,
                                           min_track_len)
                if(rt_time):
                    out_file = os.path.split(img_file)[1][:-4] + '_tracks_time.tif'
                    rainbow_tr.plot_time(img_file,
                                     np.asarray(tracks_df),
                                     output_dir+'/'+out_file,
                                     False,
                                     min_track_len)
                output_file_count += 1
            else:
                ret_str += f"Note: Track file has 0 tracks: {traj_file}.\n"

        else:
            ret_str += f"Error! Image file not found: {img_file} for rainbow tracks/ROIs.\n"

    ret_str += f"Rainbow tracks were produced for {output_file_count} track files.\n"
    return ret_str

class RerunRTMainFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, -1, 'Rerun GEMspa Rainbow Tracks', size=(900,725))

        self.default_dir=os.getcwd() #"/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/GEMspa_Trial/results9" #os.getcwd()

        self.top_panel = wx.Panel(self, size=(900,725))

        self.left_panel = wx.Panel(self.top_panel, -1, size=(450,725))

        self.right_panel = wx.Panel(self.top_panel, -1, size=(450,725))
        self.right_panel.SetBackgroundColour('#6f8089')

        self.output_text = wx.TextCtrl(self.right_panel, -1, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(450,620))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.output_text, 1, wx.EXPAND)
        self.right_panel.SetSizer(sizer)

        self.left_panel_upper = wx.Panel(self.left_panel, -1, size=(450,150))
        self.left_panel_mid = wx.Panel(self.left_panel, -1, size=(450,275))
        self.left_panel_lower = wx.Panel(self.left_panel, -1, size=(450,300))

        # left upper elements
        wx.StaticText(self.left_panel_upper, label="Select GEMspa results dir:", pos=(10, 20))
        self.results_dir = wx.DirPickerCtrl(self.left_panel_upper,
                                          path=self.default_dir,
                                          message="GEMspa results directory",
                                          pos=(10, 40), size=(425, 20))

        wx.StaticText(self.left_panel_upper, label="Select NEW rainbow tracks output dir:", pos=(10, 70))
        self.output_dir = wx.DirPickerCtrl(self.left_panel_upper,
                                            path=self.default_dir,
                                            message="Rainbow tracks output directory",
                                            pos=(10, 90), size=(425, 20))

        self.add_dir_button = wx.Button(self.left_panel_upper,
                                        label="Add to list",
                                                 pos=(10, 120))

        ## GRID
        self.mainGrid = gridlib.Grid(self.left_panel_mid, -1, pos=(10,20), size=(440,275))
        self.mainGrid.CreateGrid(0, 2)

        # left lower elements
        start_y = 15
        spacing = 27
        params_list = ['Scale (microns per px):',
                       'Min. track length, time/step-size:',
                       'Max. D for rainbow tracks:',
                       'Max. step size for rainbow tracks (microns):',
                       'Line width for rainbow tracks (pts):',
                       'Prefix for image file name:']
        default_values = [0.11, 3, 2, 1, 0.1, 'DNA_']

        self.text_ctrl_run_params = []
        for i, param in enumerate(params_list):
            wx.StaticText(self.left_panel_lower, label=param, pos=(10, start_y + i * spacing))
            self.text_ctrl_run_params.append(
                wx.TextCtrl(self.left_panel_lower, id=wx.ID_ANY, value=str(default_values[i]), pos=(300, start_y + i * spacing)))

        next_start = start_y + (i + 1) * spacing + 10
        self.D_rt_chk = wx.CheckBox(self.left_panel_lower,
                                    label="D",
                                    pos=(10, next_start))
        self.ss_rt_chk = wx.CheckBox(self.left_panel_lower,
                                    label="Step size",
                                    pos=(50, next_start))
        self.time_rt_chk = wx.CheckBox(self.left_panel_lower,
                                    label="Time",
                                    pos=(135, next_start))
        self.D_rt_chk.SetValue(True)
        self.ss_rt_chk.SetValue(True)
        self.time_rt_chk.SetValue(True)

        next_start = start_y + (i + 2) * spacing + 10
        self.execute_button = wx.Button(self.left_panel_lower,
                                        label="Run!",
                                        pos=(10, next_start))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.left_panel_upper, 0, ) #wx.SHAPED | wx.ALL, border=1)
        sizer.Add(self.left_panel_mid, 0, wx.FIXED_MINSIZE) #wx.EXPAND | wx.ALL, border=1)
        sizer.Add(self.left_panel_lower, 0, ) #wx.SHAPED | wx.ALL, border=1)
        self.left_panel.SetSizer(sizer)

        # add left and right panels to main panel
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.left_panel, 0, wx.EXPAND | wx.ALL, border=1)
        sizer.Add(self.right_panel, 1, wx.EXPAND | wx.ALL, border=1)
        self.top_panel.SetSizer(sizer)

        self.Bind(wx.EVT_BUTTON, self.on_click_add_dir_button, self.add_dir_button)
        self.Bind(wx.EVT_BUTTON, self.on_click_execute_button, self.execute_button)

        # set up grid
        self.mainGrid.SetColLabelValue(0,"GEMspa results directory")
        self.mainGrid.SetColSize(0,200)
        self.mainGrid.SetColLabelValue(1, "Output directory")
        self.mainGrid.SetColSize(1, 200)
        self.next_grid_row_pos = 0

        #self.mainGrid.SetMinSize(size=(-1, 250))

    def get_micron_per_px(self):
        return float(self.text_ctrl_run_params[0].GetValue())

    def get_min_track_len_step_size(self):
        return int(self.text_ctrl_run_params[1].GetValue())

    def get_max_D_rainbow_tracks(self):
        return float(self.text_ctrl_run_params[2].GetValue())

    def get_max_step_size_rainbow_tracks(self):
        return int(self.text_ctrl_run_params[3].GetValue())

    def get_line_width_for_rainbow_tracks(self):
        return float(self.text_ctrl_run_params[4].GetValue())

    def get_prefix_for_image_name(self):
        return self.text_ctrl_run_params[5].GetValue().strip()

    def on_click_add_dir_button(self, e):
        self.mainGrid.AppendRows(1)

        self.mainGrid.SetCellValue(self.next_grid_row_pos, 0, self.results_dir.GetPath())
        self.mainGrid.SetCellValue(self.next_grid_row_pos, 1, self.output_dir.GetPath())

        self.next_grid_row_pos += 1

    def on_click_execute_button(self, e):

        # open log file
        #save_path = os.getcwd()
        #dt_suffix = datetime.datetime.now().strftime("%c").replace("/", "-").replace("\\", "-").replace(
        #    ":","-").replace(" ", "_")
        #self.log_file = os.path.join(save_path, f"rainbow_tracks_results_{dt_suffix}.txt")

        # get parameters for rainbow tracks from user input
        micron_per_px=self.get_micron_per_px()
        min_tl=self.get_min_track_len_step_size()
        max_D=self.get_max_D_rainbow_tracks()
        max_ss=self.get_max_step_size_rainbow_tracks()
        lw=self.get_line_width_for_rainbow_tracks()
        prefix=self.get_prefix_for_image_name()

        params_str =  f"Settings:\n"
        params_str += f"micron/px={micron_per_px}\n"
        params_str += f"min track len (time/ss)={min_tl}\n"
        params_str += f"max D={max_D}\n"
        params_str += f"max ss={max_ss}\n"
        params_str += f"line width={lw}\n"
        params_str += f"prefix={prefix}\n"
        self.save_results(params_str)

        # get list of dirs to check
        nrows = self.mainGrid.GetNumberRows()
        dir_list=[]
        for i in range(nrows):
            dir_list.append([self.mainGrid.GetCellValue(i, 0),
                             self.mainGrid.GetCellValue(i, 1)])

        for cur_dirs in dir_list:
            results_dir = cur_dirs[0]
            output_dir = cur_dirs[1]
            self.save_results(f"Reading files in: {results_dir}:\n")

            cur_results_str = make_rainbow_tracks(results_dir,
                                                  prefix,
                                                  output_dir,
                                                  micron_per_px,
                                                  max_D,
                                                  max_ss,
                                                  lw,
                                                  min_tl,
                                                  self.D_rt_chk.IsChecked(),
                                                  self.ss_rt_chk.IsChecked(),
                                                  self.time_rt_chk.IsChecked())

            self.save_results(f"{cur_results_str}\n")
        self.save_results("Finished!\n\n")

    def save_results(self, output_str):
        self.output_text.write(output_str)
        self.output_text.flush()
        #self.output_text.SaveFile(self.log_file)

#---------------------------------------------------------------------------

class MyApp(wx.App, wx.lib.mixins.inspection.InspectionMixin):
    def OnInit(self):
        # Initialize the inspection tool.
        self.Init()

        frame = RerunRTMainFrame()
        frame.Show()
        self.SetTopWindow(frame)

        return True

#---------------------------------------------------------------------------

def main():
    app = MyApp(redirect=False)

    #wx.lib.inspection.InspectionTool().Show()
    app.MainLoop()

#---------------------------------------------------------------------------

if __name__ == "__main__" :
    main()



