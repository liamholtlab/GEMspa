# GemSpa

This program will calculate the Diffusion coefficient for trajectories from single particle tracking.  The input files are expected to be in .csv format, as output from MOSAIC Fiji plugin.  They must have headers titled: Trajectory, Frame, x, y.  (Any additional columns will be ignored)

(1) Create txt (tab-separated) indicating experimental conditions.  (see example in this repository: "conditions.txt")

(2) Start GemSpa by running the program from the command line, e.g. "python main_gui.py"

(3) Load in the txt file (from step 1) to GemSpa: Go to File -> Load from file.  This will open a grid view with the column headers already filled in for you.

(4) You may now select the relevant experimental conditions on the left side and click "1. Choose files", and choose the csv trajectory files for selected condition.  This will add these csv files to the grid view.  Continue with each combination of conditions in your experiment.

(5) When finished, Save this file File->Save as a txt file.

(6) RUN: go to Run->GO and this will bring up the Run Dialog box.  Choose the results directory and select the file that you saved in step (5).  It will already be selected if you just created the file.  You can always run the analysis again using the saved file.  Then, enter your parameter values - pay attention to "time between frames" and "Scale",  Then click on "Run Analysis"

(7) The program will run and the analysis results will appear in the chosen directory.  summary.txt contains one row per csv file, with Median Diffusion Coefficient and other relevant information for each file.  all_data.txt contains all Diffusion Coefficients for all tracks with min. length for all files.  "summary_D_median.pdf" is a box plot of the median Deff for each file in each experimental group, as set up in the input file.  

Note: Further plots are also available and will be described here in future versions, as they are incorporated into the GUI.  These plots can be made directly in python by calling functions in the trajectory_analysis class (defined in trajectory_analysis.py).  The basic box plot can also be further customized in this way.
