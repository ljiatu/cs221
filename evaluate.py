import os
import sys

from utils.evaluator import evaluate

def main():
    output_file_name = sys.argv[1]
    evaluate(
        f'{os.getcwd()}/outputs/{output_file_name}',
        f'{os.getcwd()}/data/test_labels.csv'
    )


if __name__ == '__main__':
    main()
