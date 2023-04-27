import wx
import wx.grid as gridlib
import pandas as pd
import numpy as np
import os
import trajectory_analysis as tja
import re

import rerun_rainbow_tracks as rerun_rt
import test_time_interval as tti


class ConditionsDialog(wx.Dialog):
    def __init__(self):
        super().__init__(parent=None, title='Experiment Conditions', size=(600, 600))
        panel = wx.Panel(self)

        wx.StaticText(panel, label="List Conditions (condition-name: cond-1,cond-2,...), one per line:", pos=(10, 25))
        self.txtCtrl=wx.TextCtrl(panel, id=wx.ID_ANY, value="", pos=(10,60), size=(350,200), style=wx.TE_MULTILINE)

        self.save_button = wx.Button(panel, wx.ID_OK, label="Create File", size=(100, 20), pos=(10, 275))
        self.cancel_button = wx.Button(panel, wx.ID_CANCEL, label="Cancel", size=(100, 20), pos=(150, 275))

    # make get/set for the "variables"
    def get_conditions(self):
        conditions_list=[]
        error=False
        data=self.txtCtrl.GetValue()
        conditions=data.split("\n")
        for condition in conditions:
            clist=condition.split(":")
            if(len(clist)>1):
                cur_list=[clist[0].strip(),]
                clist=clist[1].split(',')
                if(len(clist)>=1):
                    for val in clist:
                        cur_list.append(val.strip())
                    conditions_list.append(cur_list)
                else:
                    error=True
            else:
                error=True
        return (error, conditions_list)

class RunDialog(wx.Dialog):
    def __init__(self, default_dir, default_filepath):
        super().__init__(parent=None, title='Run GEM Analysis', size=(775, 825))
        panel = wx.Panel(self)

        wx.StaticText(panel, label="Enter directory to save the results:", pos=(10, 15))
        self.chosen_dir=wx.DirPickerCtrl(panel, path=default_dir, message="Directory to save results",
                                         pos=(10, 35), size=(500,20))

        wx.StaticText(panel, label="Enter filename for input file:", pos=(10, 75))
        self.chosen_file = wx.FilePickerCtrl(panel, path=default_filepath, message="Input file for analysis (txt)",
                                             wildcard="txt files (*.txt)|*.txt", size=(500, 20), pos=(10, 95))

        start_y=135
        spacing=27
        params_list = ['Time between frames (s):', 'Scale (microns per px):',
                       'Min. track length:', 'Number of tau for fitting: (< Min. track length)',
                       'Min. track length, ensemble ave:', 'Number of tau for fitting, ensemble ave: (< Min. track length)',
                       'Min. track length, step-size/angles:', 'Max. tau, step-size/angles:',
                       'Time step tolerance (uneven time steps) (s):',
                       'Min. D for filtered plots:','Max. D for filtered plots:', 'Max. D for rainbow tracks:',
                       'Max. step size for rainbow tracks (microns):','Line width for rainbow tracks (pts):',
                       'Radius for gem intensity measurement (px):',
                       'Prefix for image file name:']
        default_values = [0.010,0.11,11,10,11,10,3,10,0.005,0,2,2,1,0.1,3,'DNA_']
        self.text_ctrl_run_params=[]
        for i,param in enumerate(params_list):
            wx.StaticText(panel, label=param, pos=(10,start_y+i*spacing))
            self.text_ctrl_run_params.append(wx.TextCtrl(panel, id=wx.ID_ANY, value=str(default_values[i]), pos=(400, start_y+i*spacing)))

        next_start=start_y+(i+1)*spacing+10
        wx.StaticText(panel, label="Color time-coded rainbow tracks relative to:", pos=(10, next_start))
        self.rb_track_start = wx.RadioButton(panel, label='track start', pos=(300, next_start), style=wx.RB_GROUP)
        self.rb_frame_start = wx.RadioButton(panel, label='frame start', pos=(415, next_start), )
        self.rb_track_start.SetValue(True)
        self.rb_frame_start.SetValue(False)

        next_start = next_start+spacing
        wx.StaticText(panel, label="Linear fit with", pos=(10, next_start))
        self.rb_no_error_term = wx.RadioButton(panel, label='no error term', pos=(300, next_start), style=wx.RB_GROUP)
        self.rb_error_term = wx.RadioButton(panel, label='error term', pos=(415, next_start), )
        self.rb_both_error_term = wx.RadioButton(panel, label='both', pos=(515, next_start), )
        self.rb_no_error_term.SetValue(True)
        self.rb_error_term.SetValue(False)
        self.rb_both_error_term.SetValue(False)

        self.read_movie_metadata_chk = wx.CheckBox(panel, label="Use movie files to read scale/time-step", pos=(10, next_start+spacing))
        self.uneven_time_steps_chk = wx.CheckBox(panel, label="Check for uneven time steps", pos=(10, next_start+spacing*2))
        self.draw_rainbow_tracks_chk = wx.CheckBox(panel, label="Draw rainbow tracks on image files", pos=(10, next_start+spacing*3))
        self.limit_with_rois_chk = wx.CheckBox(panel, label="Use ImageJ ROI or mask files to filter tracks", pos=(10, next_start+spacing*4))
        self.measure_gem_intensities_chk = wx.CheckBox(panel, label="Measure gem intensities", pos=(10, next_start + spacing * 5))
        self.save_filtered_csvs_chk = wx.CheckBox(panel, label="Save filtered trajectory .csv files", pos=(10, next_start+spacing*6))

        self.run_button = wx.Button(panel, wx.ID_OK, label="Run Analysis", size=(100, 20), pos=(300, next_start+spacing*6))
        self.cancel_button = wx.Button(panel, wx.ID_CANCEL, label="Cancel", size=(75, 20), pos=(450, next_start+spacing*6))

    def get_rainbow_tracks_by_frame(self):
        # returns True if plot rainbow tracks by frame
        # returns False if plot rainbow tracks by track
        return self.rb_frame_start.GetValue()

    def get_fit_with_error_term(self):
        return self.rb_error_term.GetValue() or self.rb_both_error_term.GetValue()

    def get_fit_with_no_error_term(self):
        return self.rb_no_error_term.GetValue() or self.rb_both_error_term.GetValue()

    def get_save_dir(self):
        return self.chosen_dir.GetPath()
    def get_filepath(self):
        return self.chosen_file.GetPath()

    def read_movie_metadata(self):
        return self.read_movie_metadata_chk.IsChecked()
    def uneven_time_steps(self):
        return self.uneven_time_steps_chk.IsChecked()
    def draw_rainbow_tracks(self):
        return self.draw_rainbow_tracks_chk.IsChecked()
    def limit_with_rois(self):
        return self.limit_with_rois_chk.IsChecked()
    def save_filtered_csvs(self):
        return self.save_filtered_csvs_chk.IsChecked()
    def measure_gem_intensities(self):
        return self.measure_gem_intensities_chk.IsChecked()

    def get_time_step(self):
        return float(self.text_ctrl_run_params[0].GetValue())
    def get_micron_per_px(self):
        return float(self.text_ctrl_run_params[1].GetValue())

    def get_min_track_len(self):
        return int(self.text_ctrl_run_params[2].GetValue())
    def get_tlag_cutoff(self):
        return int(self.text_ctrl_run_params[3].GetValue())

    def get_min_track_len_ensemble(self):
        return int(self.text_ctrl_run_params[4].GetValue())
    def get_tlag_cutoff_ensemble(self):
        return int(self.text_ctrl_run_params[5].GetValue())

    def get_min_track_len_step_size(self):
        return int(self.text_ctrl_run_params[6].GetValue())
    def get_max_tlag_step_size(self):
        return int(self.text_ctrl_run_params[7].GetValue())

    def get_time_step_resolution(self):
        return float(self.text_ctrl_run_params[8].GetValue())
    def get_min_D_cutoff(self):
        return float(self.text_ctrl_run_params[9].GetValue())
    def get_max_D_cutoff(self):
        return float(self.text_ctrl_run_params[10].GetValue())
    def get_max_D_rainbow_tracks(self):
        return float(self.text_ctrl_run_params[11].GetValue())
    def get_max_step_size_rainbow_tracks(self):
        return int(self.text_ctrl_run_params[12].GetValue())
    def get_line_width_for_rainbow_tracks(self):
        return float(self.text_ctrl_run_params[13].GetValue())
    def get_r_for_intensity(self):
        return float(self.text_ctrl_run_params[14].GetValue())
    def get_prefix_for_image_name(self):
        return self.text_ctrl_run_params[15].GetValue().strip()

class GEMSAnalyzerMainFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, -1, 'GEMspa', size=(1200,775))
        self.statusbar =self.CreateStatusBar()
        self.statusbar.SetStatusText('Welcome!')
        self.create_menu()
        self.top_panel = wx.Panel(self)

        self.left_panel = wx.Panel(self.top_panel, -1, size=(300, 800))
        self.left_panel.SetBackgroundColour('#6f8089')

        self.right_panel = wx.Panel(self.top_panel, -1)

        self.left_panel_lower = wx.Panel(self.left_panel, -1)
        self.left_panel_upper = wx.Panel(self.left_panel, -1)

        self.leftGridSizer = wx.GridSizer(cols=2, vgap=1, hgap=1)
        self.left_panel_upper.SetSizer(self.leftGridSizer)

        self.add_cell_col_chk = wx.CheckBox(self.left_panel_lower, label="Add column for cell label using file name",pos=(10,10))
        self.choose_files_button = wx.Button(self.left_panel_lower, label="1. Choose files",pos=(10,40))
        self.choose_movie_dir_button = wx.Button(self.left_panel_lower, label="2. Choose movie files directory",pos=(10, 70))
        self.choose_image_dir_button = wx.Button(self.left_panel_lower, label="3. Choose image files directory",pos=(10, 100))
        self.delete_selected_button = wx.Button(self.left_panel_lower, label="Delete Selected Rows", pos=(10,145))

        self.Bind(wx.EVT_BUTTON, self.on_click_choose_files_button, self.choose_files_button)
        self.Bind(wx.EVT_BUTTON, self.on_click_choose_movie_dir_button, self.choose_movie_dir_button)
        self.Bind(wx.EVT_BUTTON, self.on_click_choose_image_dir_button, self.choose_image_dir_button)
        self.Bind(wx.EVT_BUTTON, self.on_click_delete_selected_button, self.delete_selected_button)

        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.left_panel_upper, 1, wx.SHAPED | wx.ALL, border=2)
        sizer.Add(self.left_panel_lower, 0, wx.EXPAND | wx.ALL, border=2)
        self.left_panel.SetSizer(sizer)

        self.mainGrid = gridlib.Grid(self.right_panel)
        self.mainGrid.CreateGrid(1, 1)

        # set relative position and add grid to right panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.mainGrid, 1, wx.EXPAND)
        self.right_panel.SetSizer(sizer)

        # add left and right panels to main panel
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.left_panel, 0, wx.EXPAND | wx.ALL, border=2)
        sizer.Add(self.right_panel, 1, wx.EXPAND | wx.ALL, border=2)
        self.top_panel.SetSizer(sizer)

        self.known_headers=['id','directory','file name', 'movie file dir','image file dir']

        self.conditions_list=[]
        self.choice_boxes = []
        self.static_texts = []

        self.default_save_dir=os.getcwd()
        self.default_input_file = os.getcwd() + "/input_file.txt"

        #self.left_panel_upper.Layout()
        #self.left_panel_lower.Layout()
        #self.left_panel.Layout()
        #self.right_panel.Layout()

        #self.Layout()
        self.Show()

    def on_click_delete_selected_button(self, e):
        # delete rows
        rows_selected = self.mainGrid.GetSelectedRows()
        while(len(rows_selected)>0):
            self.mainGrid.DeleteRows(rows_selected[0], 1)
            self.next_grid_row_pos -= 1
            rows_selected = self.mainGrid.GetSelectedRows()

    def choose_dir(self, col_name):
        with wx.DirDialog(self, "Select Directory") as dirDialog:
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return
            pathname = dirDialog.GetPath()

        # add the dir name to each row that was selected by user
        rows_selected = self.mainGrid.GetSelectedRows()
        ind=self.known_headers.index(col_name)
        for row_i in rows_selected:
            self.mainGrid.SetCellValue(row_i, ind, pathname)

    def on_click_choose_movie_dir_button(self, e):
        self.choose_dir("movie file dir")

    def on_click_choose_image_dir_button(self, e):
        self.choose_dir("image file dir")

    def on_click_choose_files_button(self, e):
        with wx.FileDialog(self, "Select files", wildcard="csv files (*.csv)|*.csv",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            pathnames = fileDialog.GetPaths()

        # add file list to grid with condition choices
        add_to_row=[]
        for choice_box in self.choice_boxes:
            # conditions/choice boxes are in the order that they are listed in the grid
            selection=choice_box.GetString(choice_box.GetSelection())
            add_to_row.append(selection)

        if (self.add_cell_col_chk.IsChecked()):
            val=self.mainGrid.GetCellValue(0,self.mainGrid.GetNumberCols()-1)
            if(val != 'cell'):
                self.mainGrid.AppendCols(1)
                self.mainGrid.SetCellValue(0,self.mainGrid.GetNumberCols()-1,'cell')

        rows_to_select=[]
        for pathname in pathnames:
            # fill in row with: id, dir, filename, movie-filename, tiff-file-name, and conditions in list
            self.mainGrid.AppendRows(1)
            (dir,file)=os.path.split(pathname)

            self.mainGrid.SetCellValue(self.next_grid_row_pos, self.known_headers.index('id'), str(self.next_id))
            self.mainGrid.SetCellValue(self.next_grid_row_pos, self.known_headers.index('directory'), dir)
            self.mainGrid.SetCellValue(self.next_grid_row_pos, self.known_headers.index('file name'), file)
            self.mainGrid.SetCellValue(self.next_grid_row_pos, self.known_headers.index('movie file dir'), "")
            self.mainGrid.SetCellValue(self.next_grid_row_pos, self.known_headers.index('image file dir'), "")

            for i,choice_box in enumerate(self.choice_boxes):
                # conditions/choice boxes are in the order that they are listed in the grid
                selection = choice_box.GetString(choice_box.GetSelection())
                self.mainGrid.SetCellValue(self.next_grid_row_pos, i+len(self.known_headers), selection)

            if (self.add_cell_col_chk.IsChecked()):
                # ADD column for cell id and fill it
                # cell id is the last part of the file: ..._ID.csv
                mo = re.match(r'.+[\_\s\-]([^\s\-\_]+)\.csv$', file)
                if (mo):
                    cell_id = mo.group(1)
                    self.mainGrid.SetCellValue(self.next_grid_row_pos, i+len(self.known_headers)+1, cell_id)

            rows_to_select.append(self.next_grid_row_pos)
            self.next_grid_row_pos+=1
            self.next_id+=1

        #select all the rows just created
        self.mainGrid.SelectRow(rows_to_select[0], addToSelected=False)
        for row in rows_to_select:
            self.mainGrid.SelectRow(row, addToSelected=True)

    def load_conditions_to_panel(self):
        self.leftGridSizer.SetRows(len(self.conditions_list)+1)

        self.choice_boxes=[]
        self.static_texts=[]

        for i,condition in enumerate(self.conditions_list):

            new_text = wx.StaticText(self.left_panel_upper, label=condition[0], )
            self.static_texts.append(new_text)
            self.leftGridSizer.Add(new_text, 0, wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_RIGHT)

            choices_arr = condition[1:]
            choices_arr.insert(0,"")

            new_choice = wx.Choice(self.left_panel_upper, choices=choices_arr, )
            self.choice_boxes.append(new_choice)
            self.leftGridSizer.Add(new_choice, 0, wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_RIGHT)

        # for spacing
        self.leftGridSizer.Add(wx.StaticText(self.left_panel_upper, label="    ", ), 0,
                               wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_RIGHT)

        self.left_panel_upper.Layout()

    def clear_conditions_panel(self):
        for cur_text in self.static_texts:
            cur_text.Hide()
            cur_text.Destroy()
        for cur_choice in self.choice_boxes:
            cur_choice.Hide()
            cur_choice.Destroy()

        self.static_texts=[]
        self.choice_boxes=[]
        self.left_panel_upper.Layout()

    def clear_grid(self):
        self.mainGrid.ClearGrid()
        ncols = self.mainGrid.GetNumberCols()
        if(ncols > 0):
            self.mainGrid.DeleteCols(0, ncols)
        nrows = self.mainGrid.GetNumberRows()
        if (nrows > 0):
            self.mainGrid.DeleteRows(0, nrows)

    def load_data_to_grid(self, df):
        self.clear_grid()

        if(df is None):
            # no data, create new - load columns names from conditions
            ncols = len(self.conditions_list) + len(self.known_headers)
            self.mainGrid.AppendCols(ncols)
            self.mainGrid.AppendRows(1)
            for col_i,col in enumerate(self.known_headers):
                self.mainGrid.SetCellValue(0, col_i, col)

            for col_i,condition in enumerate(self.conditions_list):
                self.mainGrid.SetCellValue(0,col_i+len(self.known_headers),condition[0])
            self.mainGrid.AutoSizeColumns()
            self.next_grid_row_pos=1
            self.next_id=1
        else:
            nrows = len(df) + 1
            ncols = len(df.columns)

            self.mainGrid.AppendCols(ncols)
            self.mainGrid.AppendRows(nrows)

            for col_i, col in enumerate(df.columns):
                self.mainGrid.SetCellValue(0, col_i, col)
            self.mainGrid.AutoSizeColumns()
            row_i = 1
            id=0
            for (idx, row) in df.iterrows():
                for col_i, col in enumerate(df.columns):
                    if(col == 'id'):
                        id=row[col]
                        self.mainGrid.SetCellValue(row_i, col_i, str(row[col]))
                    else:
                        self.mainGrid.SetCellValue(row_i, col_i, str(row[col]))
                row_i += 1
            self.next_grid_row_pos = row_i
            self.next_id = id+1

    def get_user_select_file(self, label="", ):
        with wx.FileDialog(self, label, wildcard="txt files (*.txt)|*.txt",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return None
            pathname = fileDialog.GetPath()
            try:
                input_df = pd.read_csv(pathname, sep='\t')
                return input_df
            except IOError:
                wx.LogError("Cannot open file '%s'." % pathname)
                return None

    def load_conditions_from_grid_data(self, df):
        self.conditions_list = []
        for col in df.columns:
            if(not col in self.known_headers):
                vals = np.unique(df[col].dropna())
                vals=list(vals.astype('str'))
                vals.insert(0,col)
                self.conditions_list.append(vals)

    def load_conditions(self, df):
        self.conditions_list=[]
        df.columns = df.columns.str.lower()
        for (idx, row) in df.iterrows():
            name=row["condition"]
            cur_arr = [name,]
            for col_i, col in enumerate(df.columns):
                if(col != "condition"):
                    if(not pd.isna(row[col])):
                        cur_arr.append(str(row[col]))
            self.conditions_list.append(cur_arr)

    def on_new(self, event):
        # load conditions file to get column names
        df_conditions = self.get_user_select_file("Open conditions file")
        if(not (df_conditions is None)):
            # delete any conditions / grid data that are already loaded
            self.clear_conditions_panel() #hides and destroys conditions text/boxes on left panel, sets the class vars to empty
            self.load_conditions(df_conditions) # sets conditions list to empty and loads in new conditions
            self.load_data_to_grid(None) # clears current grid and loads new data
            self.load_conditions_to_panel() # set num rows and load conditions to gridSizer

    def on_new_conditions(self, event):
        #open dialog to get info
        with ConditionsDialog() as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                (error, conditions)=dlg.get_conditions()
                # TODO add error handling
                if(len(conditions)>0):
                    self.clear_conditions_panel()
                    self.conditions_list=conditions.copy()
                    self.load_data_to_grid(None)
                    self.load_conditions_to_panel()

    def on_clear(self, event):
        self.conditions_list=[]
        self.clear_conditions_panel()
        self.clear_grid()

    def check_grid_columns(self, df):
        #when loading data to grid from pre-created file, check that headers are correct and add if needed
        for col in self.known_headers:
            if(not col in df.columns):
                if(col != 'movie file dir' and col != 'image file dir'):
                    return [] # error missing mandatory columns

        # all okay so far, check if need to add the missing columns
        if(not 'movie file dir' in df.columns):
            df['movie file dir']=''
        if (not 'image file dir' in df.columns):
            df['image file dir'] = ''

        #finally, fix ordering of columns
        cols = list(df.columns)
        for col in self.known_headers:
            cols.remove(col)
            cols.insert(self.known_headers.index(col),col)
        df=df[cols]

        return df

    def on_open(self, event):
        # load previously created input file
        # put data int he grid
        # get the conditions from the file
        df_grid_data = self.get_user_select_file("Open analysis input file")

        # load file (it should be txt tab delimited) into grid
        # check that it has required columns.  if missing "movie file dir" or "image file dir", add them
        if(not (df_grid_data is None)):
            valid_grid_data=self.check_grid_columns(df_grid_data)
            if(len(valid_grid_data) > 0):
                self.load_data_to_grid(valid_grid_data)

                # load conditions using column headers and unique column values
                self.clear_conditions_panel()
                self.load_conditions_from_grid_data(valid_grid_data)
                self.load_conditions_to_panel()
            else:
                with wx.MessageDialog(self, None,"Invalid input file.", "Invalid") as msgDialog:
                    msgDialog.ShowModal()

    def save_grid_to_file(self, pathname):
        # save data in grid to a txt file
        ncols = self.mainGrid.GetNumberCols()
        nrows = self.mainGrid.GetNumberRows()

        # first row is the headers
        cols = []
        for j in range(ncols):
            cols.append(self.mainGrid.GetCellValue(0, j))

        all_data = []

        for i in range(1, nrows):
            row_data = []
            for j in range(ncols):
                row_data.append(self.mainGrid.GetCellValue(i, j))
            all_data.append(row_data)

        df = pd.DataFrame(all_data, columns=cols)
        df.to_csv(pathname, sep='\t', index=False)

    def on_save(self, event):
        with wx.FileDialog(self, "Save As", wildcard="txt files (*.txt)|*.txt",
                           style=wx.FD_SAVE) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return None
            pathname = fileDialog.GetPath()
            self.save_grid_to_file(pathname)
            self.default_save_dir=os.path.split(pathname)[0]
            self.default_input_file=pathname

    def on_exit(self, event):
        self.Destroy()

    def on_go(self, event):
        #open dialog to get info
        with RunDialog(self.default_save_dir, self.default_input_file) as dlg:

            if dlg.ShowModal() == wx.ID_OK:
                save_results_dir = dlg.get_save_dir()
                input_file = dlg.get_filepath()
                read_movie_metadata = dlg.read_movie_metadata()
                uneven_time_steps = dlg.uneven_time_steps()
                rainbow_tracks=dlg.draw_rainbow_tracks()
                limit_with_rois=dlg.limit_with_rois()
                save_filtered=dlg.save_filtered_csvs()
                measure_gem_intensities=dlg.measure_gem_intensities()

                if(not save_results_dir or not input_file):
                    wx.MessageDialog(self, "Please do not leave the results directory or the file name blank.  Cannot run analysis.").ShowModal()
                else:
                    self.statusbar.SetStatusText('Please wait, running analysis...')

                    fit_msd_with_error_term = dlg.get_fit_with_error_term()

                    fit_msd_with_no_error_term = dlg.get_fit_with_no_error_term()

                    # run the analysis
                    traj_an = tja.trajectory_analysis(input_file, results_dir=save_results_dir,
                                                      use_movie_metadata=read_movie_metadata,
                                                      uneven_time_steps=uneven_time_steps,
                                                      make_rainbow_tracks=rainbow_tracks,
                                                      limit_to_ROIs=limit_with_rois,
                                                      measure_track_intensities=measure_gem_intensities,
                                                      img_file_prefix=dlg.get_prefix_for_image_name())

                    traj_an.time_step = dlg.get_time_step()
                    traj_an.micron_per_px = dlg.get_micron_per_px()
                    traj_an.ts_resolution=dlg.get_time_step_resolution()
                    traj_an.intensity_radius=dlg.get_r_for_intensity()

                    traj_an.fit_msd_with_error_term = dlg.get_fit_with_error_term()

                    traj_an.fit_msd_with_no_error_term = dlg.get_fit_with_no_error_term()

                    traj_an.output_filtered_tracks=save_filtered

                    traj_an.min_track_len_linfit = dlg.get_min_track_len()
                    traj_an.min_track_len_ensemble = dlg.get_min_track_len_ensemble() #user can set

                    traj_an.tlag_cutoff_linfit = dlg.get_tlag_cutoff()
                    traj_an.tlag_cutoff_ensemble = dlg.get_tlag_cutoff_ensemble()

                    traj_an.min_track_len_step_size = dlg.get_min_track_len_step_size()
                    traj_an.max_tlag_step_size = dlg.get_max_tlag_step_size()

                    traj_an.min_D_cutoff = dlg.get_min_D_cutoff()
                    traj_an.max_D_cutoff = dlg.get_max_D_cutoff()
                    traj_an.max_ss_rainbow_tracks = dlg.get_max_step_size_rainbow_tracks()
                    traj_an.max_D_rainbow_tracks = dlg.get_max_D_rainbow_tracks()

                    traj_an.line_width_rainbow_tracks=dlg.get_line_width_for_rainbow_tracks()
                    traj_an.time_coded_rainbow_tracks_by_frame=dlg.get_rainbow_tracks_by_frame()

                    traj_an.write_params_to_log_file()

                    traj_an.calculate_msd_and_diffusion()
                    traj_an.make_plot()
                    traj_an.make_plot_combined_data()
                    traj_an.calculate_step_sizes_and_angles()
                    traj_an.plot_alpha_D_heatmap()
                    traj_an.plot_msd_ensemble_by_group()
                    traj_an.plot_cos_theta_by_group()

                    #traj_an.plot_distribution_step_sizes(tlags=[1,])
                    #traj_an.plot_distribution_angles(tlags=[1,])
                    traj_an.plot_distribution_Deff(bin_size=0.01)
                    traj_an.plot_distribution_alpha(bin_size=0.01)
                    traj_an.plot_distribution_Deff(plot_type='', bin_size=0.1)
                    traj_an.plot_distribution_alpha(plot_type='', bin_size=0.1)

                    if(limit_with_rois):
                        traj_an.make_plot_roi_area()
                    if(measure_gem_intensities):
                        traj_an.make_plot_intensity()

                    self.statusbar.SetStatusText('Finished!')
                    print("Finished!")

    def on_test_time_interval(self, e):
        frame = tti.CheckMoviesMainFrame()
        frame.Show()


    def on_rerun_rainbow_tracks(self, e):
        frame = rerun_rt.RerunRTMainFrame()
        frame.Show()


    def create_menu(self):
        menu_bar = wx.MenuBar()
        file_menu = wx.Menu()
        edit_menu = wx.Menu()
        run_menu = wx.Menu()

        file_new_dlg = file_menu.Append(wx.ID_ANY, "&New", "Create new input file from dialog")
        file_menu.AppendSeparator()
        #file_new = file_menu.Append(wx.ID_ANY, "&New from file", "Create new input file from conditions file (txt)")
        #file_menu.AppendSeparator()
        file_open=file_menu.Append(wx.ID_ANY, "&Open", "Open input file")
        file_menu.AppendSeparator()
        file_save=file_menu.Append(wx.ID_ANY, "&Save As", "Save input file")
        file_menu.AppendSeparator()

        file_clear = file_menu.Append(wx.ID_ANY, "&Clear", "Clear conditions")
        file_menu.AppendSeparator()

        file_exit=file_menu.Append(wx.ID_ANY, "E&xit", "Close GEMspa")

        self.Bind(event=wx.EVT_MENU, handler=self.on_open, source=file_open)
        self.Bind(event=wx.EVT_MENU, handler=self.on_save, source=file_save)
        self.Bind(event=wx.EVT_MENU, handler=self.on_new_conditions, source=file_new_dlg)
        #self.Bind(event=wx.EVT_MENU, handler=self.on_new, source=file_new)
        self.Bind(event=wx.EVT_MENU, handler=self.on_clear, source=file_clear)
        self.Bind(event=wx.EVT_MENU, handler=self.on_exit, source=file_exit)

        run_go=run_menu.Append(wx.ID_ANY, "&GO", "Run the analysis")
        run_tti = run_menu.Append(wx.ID_ANY, "&Check time intervals", "Check movie time intervals")
        run_rerun_rt = run_menu.Append(wx.ID_ANY, "&Rerun Rainbow tracks", "Rerun rainbow tracks")

        self.Bind(event=wx.EVT_MENU, handler=self.on_go, source=run_go)
        self.Bind(event=wx.EVT_MENU, handler=self.on_test_time_interval, source=run_tti)
        self.Bind(event=wx.EVT_MENU, handler=self.on_rerun_rainbow_tracks, source=run_rerun_rt)

        menu_bar.Append(file_menu, '&File')
        menu_bar.Append(edit_menu, '&Edit')
        menu_bar.Append(run_menu, '&Run')
        self.SetMenuBar(menu_bar)

if __name__ == '__main__':
    app = wx.App()

    frame = GEMSAnalyzerMainFrame()
    app.MainLoop()

