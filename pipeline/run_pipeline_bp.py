import glob
import argparse
import os
import datetime


# Input is a directory where Trajectory files (Mosaic output) are stored and the number of files to run per group
# This script will execute a bash script that submits a job to bigpurple for each group of Traj files in the directory
# When all jobs are completed, individual outputs can be combined into one summary file
# The group size should correspond to the number of cpus-per-task that is set in the file run_gemspa.sh


# max of 40 cores per NODE on big purple
# 384GB is total memory of a standard compute node on big purple
CORES_PER_NODE=40
MEM_PER_NODE=320 #384 # in practice, this seems to be 320 (otherwise job fails)
MAX_MEM_TO_REQUEST=20


def process_path(input_path, output_path, stdout_path, group_size, job_time, time_step, micron_per_pixel, verbose):
    files_list = glob.glob(f"{input_path}/*.csv")

    group_size=int(group_size)

    if(group_size > CORES_PER_NODE):
        group_size=CORES_PER_NODE

    mem_per_cpu=int(MEM_PER_NODE/group_size)
    if(mem_per_cpu > MAX_MEM_TO_REQUEST):
        mem_per_cpu=MAX_MEM_TO_REQUEST

    # create bash script that will be executed
    x = str(datetime.datetime.now())
    x=x.replace(" ","-")
    x=x.replace(":", "-")
    x=x.replace(".", "-")
    fname=f"run_file_group-{x}.sh"
    f = open(fname, "w")
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH --time={job_time}\n")
    f.write(f"#SBATCH --mem-per-cpu={mem_per_cpu}G\n")
    f.write(f"#SBATCH --nodes=1\n")
    f.write(f"#SBATCH --ntasks=1\n")
    f.write(f"#SBATCH --cpus-per-task={group_size}\n")
    f.write(f"#SBATCH --output={stdout_path}/%x-%j.out\n")
    f.write(f"#SBATCH --error={stdout_path}/%x-%j.out\n")

    f.write("\n")
    f.write("input_path=$1\n")
    f.write("output_path=$2\n")
    f.write("group_start=$3\n")
    f.write("group_end=$4\n")
    f.write("time_step=$5\n")
    f.write("micron_per_px=$6\n")
    f.write("\n")
    f.write("/gpfs/data/holtlab/GEMS/anaconda/bin/python -m pipeline.run_msd_diffusion_group")
    f.write(" $input_path $output_path $group_start $group_end $time_step $micron_per_px\n")
    f.close()

    print("Running pipeline:")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Stdout/err file path: {stdout_path}")
    print(f"Time step: {time_step}")
    print(f"Micron per px: {micron_per_pixel}")
    print(f"Number of files found in input path: {len(files_list)}")
    print(f"Group size: {group_size}")
    print(f"Mem per cpu: {mem_per_cpu}")
    print(f"Job time: {job_time}")
    print("")

    for i in range(0, len(files_list), group_size):
        run_str=f"sbatch {fname} \"{input_path}\" \"{output_path}\" \"{i}\" \"{i+group_size}\" \"{time_step}\" \"{micron_per_pixel}\""
        if(verbose):
            print(run_str)
        os.system(run_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", help="full path to input csv trajectory files",
                        type=str)
    parser.add_argument("-o", "--output_path", help="full path to save output (input path)",
                        type=str)
    parser.add_argument("-so", "--stdout_path", help="full path to save script stdout/stderr messages (output path)",
                        type=str)
    parser.add_argument("-s", "--group_size", help="number of files for each job submission (8)",
                        type=int, default=8)
    parser.add_argument("-jt", "--job_time", help="job time requested (0-12:00:00)",
                        type=str, default="0-12:00:00")
    parser.add_argument("-m", "--micron_per_px", help="pixel size in microns (0.0917)",
                        type=float, default=0.0917)
    parser.add_argument("-ts", "--time_step", help="time step in seconds (0.010)",
                        type=float, default=0.010)
    parser.add_argument("-v", "--verbose", help="print job submission commands to stdout",
                        action="store_true")

    args = parser.parse_args()
    if(args.output_path is None):
        args.output_path=args.input_path
    if (args.stdout_path is None):
        args.stdout_path = args.output_path

    process_path(args.input_path,
                 args.output_path,
                 args.stdout_path,
                 args.group_size,
                 args.job_time,
                 args.time_step,
                 args.micron_per_px,
                 args.verbose)




