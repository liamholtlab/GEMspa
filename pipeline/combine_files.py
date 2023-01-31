import glob
import argparse
import pandas as pd

# Input is a path to the output files (txt files)
# This script will combine all files into one

def combine_files(input_path, file_pre, file_suffix=None):
    # Combine files
    all_files = glob.glob(f"{input_path}/{file_pre}-*.txt")
    full_df = pd.DataFrame()
    for file in all_files:
        cur_df = pd.read_csv(file, sep='\t', index_col=0)
        full_df = pd.concat([full_df, cur_df])

    full_df.index = range(len(full_df))
    if (file_suffix is None):
        full_df.to_csv(f"{input_path}/{file_pre}-merged.txt", sep="\t")
    else:
        full_df.to_csv(f"{input_path}/{file_pre}-{file_suffix}", sep="\t")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="full path to pipeline output (txt files to be combined)",
                        type=str)
    parser.add_argument("-n", "--file_name", help="file name of merged file (merged.txt)",
                        type=str)

    args = parser.parse_args()

    combine_files(args.input_path, "summary", args.file_name)
    combine_files(args.input_path, "all_data", args.file_name)
    combine_files(args.input_path, "step_sizes", args.file_name)
    combine_files(args.input_path, "angles", args.file_name)





