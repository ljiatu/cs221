import pandas as pd

TRAIN_DATA_FILE_PATH = 'data/train.csv'


def main():
    # counter = Counter()
    training_data = pd.read_csv(TRAIN_DATA_FILE_PATH)
    lengths = []
    for _, row in training_data.iterrows():
        lengths.append(len(row[1].split()))
        # counter[len(row[1].split())] += 1

    lengths.sort()
    print(lengths.index(150))
    print(len(lengths))
    print(lengths.index(150) / len(lengths))
    # lists = sorted(counter.items())[:100]
    # x, y = zip(*lists)
    # plt.bar(x, y)
    # plt.ylabel('Count')
    # plt.title('# comments by word count')
    # plt.show()


if __name__ == '__main__':
    main()
