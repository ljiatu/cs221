"""
Preprocess the training data file to remove invalid lines.
"""
import pandas as pd


def run(text_file_path: str, label_file_path: str, output_path: str):
    text_df = pd.read_csv(text_file_path)
    label_df = pd.read_csv(label_file_path)
    joined = text_df.set_index('id').join(label_df.set_index('id'), how='inner')
    print(len(joined))
    filtered = joined[joined.toxic != -1]
    print(len(filtered))
    filtered.to_csv(output_path)


if __name__ == '__main__':
    run('data/test.csv', 'data/test_labels.csv', 'data/processed.csv')
