import sys

import pandas as pd


def pick(output_file_path: str, truth_file_path: str, result_file_path: str):
    output_df = pd.read_csv(output_file_path)
    truth_df = pd.read_csv(truth_file_path)
    renamed_truth_df = truth_df.rename(str.upper, axis='columns')
    joined = output_df.set_index('id').join(renamed_truth_df.set_index('ID'))
    filtered = joined[
        incorrect(joined.toxic, joined.TOXIC)
        | incorrect(joined.severe_toxic, joined.SEVERE_TOXIC)
        | incorrect(joined.obscene, joined.OBSCENE)
        | incorrect(joined.threat, joined.THREAT)
        | incorrect(joined.insult, joined.INSULT)
        | incorrect(joined.identity_hate, joined.IDENTITY_HATE)
    ]
    filtered.to_csv(result_file_path)


def incorrect(a: float, b: float) -> bool:
    return abs(a - b) > 0.5


if __name__ == '__main__':
    output_file_path = f'outputs/{sys.argv[1]}'
    pick(output_file_path, 'data/processed.csv', 'error_analysis.csv')
