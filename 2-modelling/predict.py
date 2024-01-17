import pandas as pd
import argparse

parser = argparse.ArgumentParser(prog='Training GB model',
                                 description='Just for reference',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('filename', type=str)
args = parser.parse_args()

# read the test dataset
test_df = pd.read_csv(args.filename)

# apply the derived function from explore notebook
test_df['target'] = test_df['6'].pow(2) + test_df['7']

# write to csv
test_df.to_csv('test_results.csv', index=False)