from tifffile import TiffFile
from nd2reader import ND2Reader
import re
import numpy as np
import glob
import wx
import wx.grid as gridlib
import os
import datetime
import wx.lib.mixins.inspection

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

def read_movie_nd2(file_name):
    with ND2Reader(file_name) as images:
        images.iter_axes = 'z'
        for i,image in enumerate(images):
            image=np.asarray(image).astype('float32')
            if(i > 0):
                subtr=np.abs(np.asarray(image)-prev_image)
                #print(np.percentile(subtr.flatten(), [25, 50, 75]), end=' (')
                print(f"({np.mean(subtr.flatten()):.3f}, {np.std(subtr.flatten()):.3f})", end=' ')
                if(i%10==0):
                    print("")
            prev_image=image.copy()
        print("")

def read_movie_metadata_nd2(file_name):
    with ND2Reader(file_name) as images:
        err_msg = ""

        if (len(images.metadata['experiment']['loops']) > 0):
            step=images.metadata['experiment']['loops'][0]['sampling_interval']

            #print(f"Time step: {step/1000}")
            step = np.round(step, 0) #3)
            step = step / 1000
        else:
            err_msg="Could not read sampling_interval from movie metadata."
            #print(f"Error reading exposure from nd2 movie! {file_name}")
            #print(images.metadata['experiment']['loops'])
            step=0

        steps = images.timesteps[1:] - images.timesteps[:-1]

        # round steps to the nearest ms
        steps = np.round(steps, 0) #3)

        #convert steps from ms to s
        steps=steps/1000

        microns_per_pixel=images.metadata['pixel_microns']
        return [err_msg, microns_per_pixel, step, steps]

class CheckMoviesMainFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, -1, 'Verify Time Intervals for nd2 files', size=(900,625))

        self.default_dir=os.getcwd()

        self.top_panel = wx.Panel(self, size=(900,625))

        self.left_panel = wx.Panel(self.top_panel, -1, size=(450,625))

        self.right_panel = wx.Panel(self.top_panel, -1, size=(450,625))
        self.right_panel.SetBackgroundColour('#6f8089')

        self.output_text = wx.TextCtrl(self.right_panel, -1, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(450,570))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.output_text, 1, wx.EXPAND)
        self.right_panel.SetSizer(sizer)

        self.left_panel_upper = wx.Panel(self.left_panel, -1, size=(450,125))
        self.left_panel_mid = wx.Panel(self.left_panel, -1, size=(450,350))
        self.left_panel_lower = wx.Panel(self.left_panel, -1, size=(450,125))

        # left upper elements
        wx.StaticText(self.left_panel_upper, label="Select nd2 files directory, then click 'Add to list':", pos=(10, 20))
        self.movie_dir = wx.DirPickerCtrl(self.left_panel_upper,
                                          path=self.default_dir,
                                          message="Movie files directory",
                                          pos=(10, 40), size=(425, 20))

        self.recursive_chk = wx.CheckBox(self.left_panel_upper,
                                         label="Recurse into directory",
                                         pos=(10, 60))

        self.add_dir_button = wx.Button(self.left_panel_upper,
                                        label="Add to list",
                                                 pos=(10, 100))

        ## GRID
        self.mainGrid = gridlib.Grid(self.left_panel_mid, -1, pos=(10,20), size=(440,325))
        self.mainGrid.CreateGrid(0, 2)

        # left lower elements
        wx.StaticText(self.left_panel_lower, label="Directory to save results:", pos=(10, 20))
        self.save_dir = wx.DirPickerCtrl(self.left_panel_lower,
                                          path=self.default_dir,
                                          message="Output directory",
                                          pos=(10, 35), size=(425, 20))

        wx.StaticText(self.left_panel_lower, label="Expected time step:", pos=(10, 65))
        self.expected_time_step_box = wx.TextCtrl(self.left_panel_lower,
                                                  size=wx.Size(35,20),
                                                  value="10",
                                                  pos=(135, 65))
        wx.StaticText(self.left_panel_lower, label="+/-", pos=(170, 65))
        self.time_resolution_box = wx.TextCtrl(self.left_panel_lower,
                                                  size=wx.Size(25, 20),
                                                  value="5",
                                                  pos=(195, 65))
        wx.StaticText(self.left_panel_lower, label="ms", pos=(225, 65))


        self.execute_button = wx.Button(self.left_panel_lower,
                                        label="Run!",
                                        pos=(350, 65))

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
        self.mainGrid.SetColLabelValue(0,"Directory")
        self.mainGrid.SetColSize(0,270)
        self.mainGrid.SetColLabelValue(1, "Recursive")
        self.next_grid_row_pos = 0

        #self.mainGrid.SetMinSize(size=(-1, 250))

    def on_click_add_dir_button(self, e):
        self.mainGrid.AppendRows(1)

        self.mainGrid.SetCellValue(self.next_grid_row_pos, 0, self.movie_dir.GetPath())
        self.mainGrid.SetCellValue(self.next_grid_row_pos, 1, str(int(self.recursive_chk.IsChecked())))

        self.next_grid_row_pos += 1

    def on_click_execute_button(self, e):

        # get expected time step from user input
        self.expected_time_step=float(self.expected_time_step_box.GetValue())
        self.expected_time_step/=1000

        self.time_resolution = float(self.time_resolution_box.GetValue())
        self.time_resolution /= 1000

        # get list of dirs to check
        nrows = self.mainGrid.GetNumberRows()
        dir_list=[]
        for i in range(nrows):
            dir_list.append([self.mainGrid.GetCellValue(i, 0),
                             int(self.mainGrid.GetCellValue(i, 1))])

        # open log file
        save_path=self.save_dir.GetPath()
        dt_suffix=datetime.datetime.now().strftime("%c").replace("/", "-").replace("\\", "-").replace(":","-").replace(" ", "_")
        self.log_file = os.path.join(save_path, f"time_check_results_{dt_suffix}.txt")

        for cur_dir in dir_list:
            to_check = cur_dir[0]
            self.save_results(f"Checking files in: {to_check}:\n")
            cur_results_str = self.check_dir(to_check)
            self.save_results(f"{cur_results_str}\n")

            if(cur_dir[1]):
                for root, dirs, files in os.walk(cur_dir[0], ):
                    for name in dirs:
                        to_check=os.path.join(root, name)
                        self.save_results(f"Checking files in: {to_check}:\n")
                        cur_results_str = self.check_dir(to_check)
                        self.save_results(f"{cur_results_str}\n")

    def save_results(self, output_str):
        self.output_text.write(output_str)
        self.output_text.flush()
        self.output_text.SaveFile(self.log_file)

    def check_dir(self, dir_to_check, verbose=True):
        full_ret_str=""

        extens = 'nd2'  # 'tif'
        movie_files = glob.glob(f"{dir_to_check}/*.{extens}")
        for movie_file in movie_files:
            ret_str=""
            if(verbose):
                print(movie_file)
            if (extens == 'nd2'):
                ret_vals = read_movie_metadata_nd2(movie_file)

                # error reading sampling interval from meta data
                if(ret_vals[0]):
                    ret_str += f"{ret_vals[0]}\n"

                # difference in sampling interval?  (I think it is the average of the full time step list)
                elif(abs(ret_vals[2] - self.expected_time_step) > self.time_resolution):
                    ret_str += f"'sampling_interval' [{ret_vals[2]} ms] does not match expected time step.\n"

                # full time step list: are any differences greater than the resolution?
                time_step_diffs=np.abs(ret_vals[3]-self.expected_time_step)
                if((time_step_diffs > self.time_resolution).any()):
                    ret_str += f"Some image time steps do not match expected time step.  Values found (sec): {np.unique(ret_vals[3])}\n"

                if(verbose):
                    print(f"Cal: {np.round(ret_vals[1], 3)}, Time-step: {ret_vals[2]} ms")
                    print(f"All steps: {ret_vals[3]}")
                    print(f"Mean: {np.round(np.mean(ret_vals[3]), 3)}")

                # read_movie_nd2(movie_file)
            else:
                #ret_vals = read_movie_metadata_tif(movie_file)
                #if(verbose):
                #    print(f"Cal: {ret_vals[0]}, Time-step: {ret_vals[1]} s")
                #    print(f"All steps: {ret_vals[2]}")
                pass

            if(ret_str):

                full_ret_str += f"\n *Error*: {os.path.split(movie_file)[1]}\n{ret_str}"

        return full_ret_str

#---------------------------------------------------------------------------

class MyApp(wx.App, wx.lib.mixins.inspection.InspectionMixin):
    def OnInit(self):
        # Initialize the inspection tool.
        self.Init()

        frame = CheckMoviesMainFrame()
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