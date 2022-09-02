import glob
import argparse
import pandas as pd


# Input is a path to the output files (txt files)
# This script will combine all files into one


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="full path to pipeline output (txt files to be combined)",
                        type=str)
    parser.add_argument("-n", "--file_name", help="file name of merged file (merged.txt)",
                        type=str)

    args = parser.parse_args()
    all_files = glob.glob(f"{args.input_path}/summary-*.txt")
    full_df=pd.DataFrame()
    for file in all_files:
        cur_df = pd.read_csv(file, sep='\t', index_col=0)
        full_df = pd.concat([full_df, cur_df])

    full_df.index=range(len(full_df))
    if(args.file_name is None):
        full_df.to_csv(f"{args.input_path}/merged.txt", sep="\t")
    else:
        full_df.to_csv(f"{args.input_path}/{args.file_name}", sep="\t")




