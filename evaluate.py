import os

from utils.evaluator import evaluate


def main():
    evaluate(
        f'{os.getcwd()}/outputs/2018-12-09T00:45:43.323536.txt',
        f'{os.getcwd()}/data/test_labels.csv'
    )


if __name__ == '__main__':
    main()
