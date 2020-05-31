import csv
import os
import argparse


train_perc = 0.8
val_perc = 0.1
test_perc = 0.1

def _read_csv(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return list(csv.reader(f))

def _write_csv(out_file, data):
    with open(out_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'question', 'contexts', 'ending0', 'ending1', 'ending2', 'ending3', 'label'])
        writer.writerows(data)

def list_splitter(train_test_data, val_data):
    elements = len(train_test_data)
    train_end = int(elements*train_perc)
    val_end = int(elements*(train_perc + val_perc))
    return train_test_data[:train_end], val_data[train_end:val_end], train_test_data[val_end:]

def  main(train_data, val_data):
    train_test_data = _read_csv(train_data)
    val_data = _read_csv(val_data)
    train, val, test = list_splitter(train_test_data, val_data)

    base_data_dir = "splitdata/"
    os.mkdir(base_data_dir)
    _write_csv(base_data_dir + "train.csv", train)
    _write_csv(base_data_dir + "val.csv", val)
    _write_csv(base_data_dir + "test.csv", test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_data", "--train_data", help="Train Data")
    parser.add_argument("-val_data", "--val_data", help="Validation Data")
    args = parser.parse_args()
    main(args.train_data, args.val_data)
