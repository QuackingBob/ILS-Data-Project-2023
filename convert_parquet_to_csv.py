import pandas as pd
import sys

def main():
    input_file = sys.argv[1]
    print(f"Converting f{input_file} to csv ... ")
    df = pd.read_parquet(input_file)
    file_name = input_file.split(".")[0]
    print(f"Saving as {file_name}.csv ...")
    df.to_csv(file_name + ".csv")
    print("Done")

if __name__ == "__main__":
    main()