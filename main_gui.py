import wx
import wx.grid as gridlib
import pandas as pd
import numpy as np
import os
import trajectory_analysis as tja
import re

class ConditionsDialog(wx.Dialog):
    def __init__(self):
        super().__init__(parent=None, title='Experiment Conditions', size=(800, 800))
        panel = wx.Panel(self)

        wx.StaticText(panel, label="Conditions:", pos=(10, 25))

        wx.StaticLine(panel, size=(250, 1), pos=(5, 55), style=wx.LI_HORIZONTAL)
        wx.StaticText(panel, label="Add/Edit names:", pos=(10, 85))
        self.conditions_choice = wx.ComboBox(panel, wx.ID_ANY, choices=[], pos=(10, 115))
        self.edit_button = wx.Button(panel, wx.ID_ANY, label="Add/Edit labels", size=(100, 20), pos=(200, 115))

        wx.StaticLine(panel, size=(250, 1), pos=(5, 145), style=wx.LI_HORIZONTAL)
        wx.StaticText(panel, label="Labels:", pos=(10, 175))
        self.conditions_list = wx.TextCtrl(panel, wx.ID_ANY, pos=(10, 205))
        self.save_button = wx.Button(panel, wx.ID_ANY, label="Save labels", size=(100, 20), pos=(200, 205))

        wx.StaticLine(panel, size=(250, 1), pos=(5, 235), style=wx.LI_HORIZONTAL)
        self.create_button = wx.Button(panel, wx.ID_OK, label="Create", size=(100, 20), pos=(10, 265))
        self.cancel_button = wx.Button(panel, wx.ID_CANCEL, label="Cancel", size=(100, 20), pos=(125, 265))

        self.Bind(wx.EVT_BUTTON, self.on_click_edit_button, self.edit_button)
        self.Bind(wx.EVT_BUTTON, self.on_click_save_button, self.save_button)

        self.conditions = {}

    def on_click_edit_button(self):
        # get condition labels that are stored for currently selected title
        # Add them to the list below
        text = self.conditions_choice.GetStringSelection().strip()
        if(text):
            if(text in self.conditions):
                # place labels in the text control
                self.conditions_list.SetValue(self.conditions[text]) #string with each line separated by '\n'

    def on_click_save_button(self):
        # update the conditions in the text control
        text = self.conditions_choice.GetStringSelection().strip()
        if (text):
            pass

        # TODO: set up variables for condition titles/labels
        # fill in the functions above
        # make get/set for the variables

        # add function to the main program to create the headers on the GRID once "Create" button is pressed

class RunDialog(wx.Dialog):
    def __init__(self, default_dir, default_filepath):
        super().__init__(parent=None, title='Run GEM Analysis', size=(800, 800))
        panel = wx.Panel(self)

        wx.StaticText(panel, label="Enter directory to save the results:", pos=(10, 25))
        self.chosen_dir=wx.DirPickerCtrl(panel, path=default_dir, message="Directory to save results",
                                         pos=(10, 50), size=(500,20))

        wx.StaticText(panel, label="Enter filename for input file:", pos=(10, 100))
        self.chosen_file = wx.FilePickerCtrl(panel, path=default_filepath, message="Input file for analysis (txt)",
                                             wildcard="txt files (*.txt)|*.txt", size=(500, 20), pos=(10, 125))

        start_y=200
        spacing=35
        params_list = ['Time between frames (s):', 'Scale (microns per px):', 'Min. track length (fit):',
                       'Track length cutoff (fit):', 'Min track length (step size/angles):',
                       'Max t-lag (step size/angles):','Min D for plots:','Max D for plots:']
        default_values = [0.010,0.11,11,11,3,3,0,2]
        self.text_ctrl_run_params=[]
        for i,param in enumerate(params_list):
            wx.StaticText(panel, label=param, pos=(10,start_y+i*spacing))
            self.text_ctrl_run_params.append(wx.TextCtrl(panel, id=wx.ID_ANY, value=str(default_values[i]), pos=(250, start_y+i*spacing)))

        next_start=start_y+(i+1)*spacing+25
        self.read_movie_metadata_chk = wx.CheckBox(panel, label="Use movie files to read scale/time-step", pos=(10, next_start))
        self.run_button = wx.Button(panel, wx.ID_OK, label="Run Analysis", size=(100, 20), pos=(10, next_start+35))
        self.cancel_button = wx.Button(panel, wx.ID_CANCEL, label="Cancel", size=(75, 20), pos=(150, next_start+35))

    def get_save_dir(self):
        return self.chosen_dir.GetPath()
    def get_filepath(self):
        return self.chosen_file.GetPath()
    def read_movie_metadata(self):
        return self.read_movie_metadata_chk.IsChecked()

    def get_time_step(self):
        return float(self.text_ctrl_run_params[0].GetValue())
    def get_micron_per_px(self):
        return float(self.text_ctrl_run_params[1].GetValue())
    def get_min_track_len_linfit(self):
        return int(self.text_ctrl_run_params[2].GetValue())
    def get_track_len_cutoff_linfit(self):
        return int(self.text_ctrl_run_params[3].GetValue())
    def get_min_track_len_step_size(self):
        return int(self.text_ctrl_run_params[4].GetValue())
    def get_max_tlag_step_size(self):
        return int(self.text_ctrl_run_params[5].GetValue())
    def get_min_D_cutoff(self):
        return float(self.text_ctrl_run_params[6].GetValue())
    def get_max_D_cutoff(self):
        return float(self.text_ctrl_run_params[7].GetValue())

class GEMSAnalyzerMainFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, -1, 'GEMSAnalyzer', size=(1200,800))
        self.statusbar =self.CreateStatusBar()
        self.statusbar.SetStatusText('Welcome!')
        self.create_menu()
        self.top_panel = wx.Panel(self)


        self.left_panel = wx.Panel(self.top_panel, -1, size=(300, 800))
        self.left_panel.SetBackgroundColour('#6f8089')

        self.right_panel = wx.Panel(self.top_panel, -1)

        self.left_panel_lower = wx.Panel(self.left_panel, -1,)
        self.left_panel_upper = wx.Panel(self.left_panel, -1)

        self.leftGridSizer = wx.GridSizer(cols=2, vgap=1, hgap=1)
        self.left_panel_upper.SetSizer(self.leftGridSizer)

        self.add_cell_col_chk = wx.CheckBox(self.left_panel_lower, label="Add column for cell label using file name", pos=(10,10))
        self.choose_files_button = wx.Button(self.left_panel_lower, label="1. Choose files", pos=(10,40))
        self.choose_movie_file_button = wx.Button(self.left_panel_lower, label="2. Choose movie file", pos=(10, 70))

        self.Bind(wx.EVT_BUTTON, self.on_click_choose_files_button, self.choose_files_button)
        self.Bind(wx.EVT_BUTTON, self.on_click_choose_movie_file_button, self.choose_movie_file_button)

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

    def on_click_choose_movie_file_button(self, e):
        with wx.FileDialog(self, "Select files", wildcard="nd2 files (*.nd2)|*.nd2",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fileDialog.GetPath()

        # add the movie file to each row that was selected by user

        # return value is a 0-indexed list of selected rows
        rows_selected = self.mainGrid.GetSelectedRows()
        for row_i in rows_selected:
            print(row_i)
            self.mainGrid.SetCellValue(row_i, 3, pathname)


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

        for pathname in pathnames:
            # fill in row with: id, dir, filename, movie-filename and conditions in list
            self.mainGrid.AppendRows(1)
            (dir,file)=os.path.split(pathname)

            self.mainGrid.SetCellValue(self.next_grid_row_pos, 0, str(self.next_id))
            self.mainGrid.SetCellValue(self.next_grid_row_pos, 1, dir)
            self.mainGrid.SetCellValue(self.next_grid_row_pos, 2, file)
            self.mainGrid.SetCellValue(self.next_grid_row_pos, 3, "")

            for i,choice_box in enumerate(self.choice_boxes):
                # conditions/choice boxes are in the order that they are listed in the grid
                selection = choice_box.GetString(choice_box.GetSelection())
                self.mainGrid.SetCellValue(self.next_grid_row_pos, i+4, selection)

            if (self.add_cell_col_chk.IsChecked()):
                # ADD column for cell id and fill it
                # cell id is the last part of the file: ..._ID.csv
                mo = re.match(r'.+[\_\s\-]([^\s\-\_]+)\.csv$', file)
                if (mo):
                    cell_id = mo.group(1)
                    self.mainGrid.SetCellValue(self.next_grid_row_pos, i+4+1, cell_id)

            self.next_grid_row_pos+=1
            self.next_id+=1

    def load_conditions_to_panel(self):
        self.leftGridSizer.SetRows(len(self.conditions_list)+1)

        self.choice_boxes=[]
        self.static_texts=[]

        start_pos=10
        incr=25
        for i,condition in enumerate(self.conditions_list):

            new_text = wx.StaticText(self.left_panel_upper, label=condition[0], ) #pos=(10,start_pos+(i*incr)))
            self.static_texts.append(new_text)
            self.leftGridSizer.Add(new_text, 0, wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_RIGHT)


            choices_arr = condition[1:]
            choices_arr.insert(0,"")

            new_choice = wx.Choice(self.left_panel_upper, choices=choices_arr, ) #pos=(200,start_pos+(i*incr)))
            self.choice_boxes.append(new_choice)
            self.leftGridSizer.Add(new_choice, 0, wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_RIGHT)

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
        self.mainGrid.DeleteCols(0, ncols)
        nrows = self.mainGrid.GetNumberRows()
        self.mainGrid.DeleteRows(0, nrows)

    def load_data_to_grid(self, df):
        self.clear_grid()

        if(df is None):
            # no data, create new - load columns names from conditions
            n=4
            ncols = len(self.conditions_list) + n  # first 4 columns are id, directory, file name, movie file name
            self.mainGrid.AppendCols(ncols)
            self.mainGrid.AppendRows(1)
            self.mainGrid.SetCellValue(0, 0, "id")
            self.mainGrid.SetCellValue(0, 1, "directory")
            self.mainGrid.SetCellValue(0, 2, "file name")
            self.mainGrid.SetCellValue(0, 3, "movie file name")

            for col_i,condition in enumerate(self.conditions_list):
                self.mainGrid.SetCellValue(0,col_i+n,condition[0])
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
            if(col != 'id' and col != 'file name' and col != 'directory' and col != 'movie file name'):
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
                pass

    def on_clear(self, event):
        self.conditions_list=[]
        self.clear_conditions_panel()

    def on_open(self, event):
        # load previously created input file
        # put data int he grid
        # get the conditions from the file
        df_grid_data = self.get_user_select_file("Open analysis input file")

        # load file (it should be txt tab delimited) into grid
        if(not (df_grid_data is None)):
            self.load_data_to_grid(df_grid_data)

            # load conditions using column headers and unique column values
            self.clear_conditions_panel()
            self.load_conditions_from_grid_data(df_grid_data)
            self.load_conditions_to_panel()

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

                if(not save_results_dir or not input_file):
                    wx.MessageDialog(self, "Please do not leave the results directory or the file name blank.  Cannot run analysis.").ShowModal()
                else:
                    self.statusbar.SetStatusText('Please wait, running analysis...')

                    # run the analysis
                    # def __init__(self, data_file, results_dir='.', movie_file_col=False, log_file=''):
                    if (read_movie_metadata):
                        # time step and micron per px is read from the movie files' metatdata
                        traj_an = tja.trajectory_analysis(input_file, save_results_dir, True)
                    else:
                        traj_an = tja.trajectory_analysis(input_file, save_results_dir, False)

                    traj_an.time_step = dlg.get_time_step()
                    traj_an.micron_per_px = dlg.get_micron_per_px()

                    traj_an.min_track_len_linfit = dlg.get_min_track_len_linfit()
                    traj_an.min_track_len_step_size = dlg.get_min_track_len_step_size()

                    traj_an.track_len_cutoff_linfit = dlg.get_track_len_cutoff_linfit()
                    traj_an.max_tlag_step_size = dlg.get_max_tlag_step_size()
                    traj_an.min_D_cutoff = dlg.get_min_D_cutoff()
                    traj_an.max_D_cutoff = dlg.get_max_D_cutoff()

                    traj_an.write_params_to_log_file()

                    traj_an.calculate_step_sizes_and_angles()
                    traj_an.calculate_msd_and_diffusion()
                    traj_an.make_plot()
                    traj_an.plot_distribution_step_sizes(tlags=[1,])
                    #traj_an.plot_distribution_angles(tlags=[1,])

                    self.statusbar.SetStatusText('Finished!')

    def create_menu(self):
        menu_bar = wx.MenuBar()
        file_menu = wx.Menu()
        edit_menu = wx.Menu()
        run_menu = wx.Menu()

        file_new_dlg = file_menu.Append(wx.ID_ANY, "&New (don't use)", "Create new input file from dialog")
        file_menu.AppendSeparator()
        file_new = file_menu.Append(wx.ID_ANY, "&New from file", "Create new input file from conditions file (txt)")
        file_menu.AppendSeparator()
        file_open=file_menu.Append(wx.ID_ANY, "&Open", "Open input file")
        file_menu.AppendSeparator()
        file_save=file_menu.Append(wx.ID_ANY, "&Save As", "Save input file")
        file_menu.AppendSeparator()

        file_clear = file_menu.Append(wx.ID_ANY, "&Clear", "Clear conditions")
        file_menu.AppendSeparator()

        file_exit=file_menu.Append(wx.ID_ANY, "E&xit", "Close GEMSAnalyzer")

        self.Bind(event=wx.EVT_MENU, handler=self.on_open, source=file_open)
        self.Bind(event=wx.EVT_MENU, handler=self.on_save, source=file_save)
        self.Bind(event=wx.EVT_MENU, handler=self.on_new_conditions, source=file_new_dlg)
        self.Bind(event=wx.EVT_MENU, handler=self.on_new, source=file_new)
        self.Bind(event=wx.EVT_MENU, handler=self.on_clear, source=file_clear)
        self.Bind(event=wx.EVT_MENU, handler=self.on_exit, source=file_exit)

        run_go=run_menu.Append(wx.ID_ANY, "&GO", "Run the analysis")

        self.Bind(event=wx.EVT_MENU, handler=self.on_go, source=run_go)

        menu_bar.Append(file_menu, '&File')
        menu_bar.Append(edit_menu, '&Edit')
        menu_bar.Append(run_menu, '&Run')
        self.SetMenuBar(menu_bar)

if __name__ == '__main__':
    app = wx.App()

    frame = GEMSAnalyzerMainFrame()
    app.MainLoop()

