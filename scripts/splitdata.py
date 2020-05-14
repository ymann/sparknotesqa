import pandas as pd
from sklearn.model_selection import train_test_split

def get_train_val_test_split(input_file):
    x = pd.read_csv(input_file, names=['id', 'question', 'contexts', 'ending0', 'ending1', 'ending2', 'ending3', 'label'])
    train, test_val, _, _ = train_test_split(x,x.label,test_size=0.2)

    val, test, _, _ = train_test_split(test_val,test_val.label,test_size=0.5)
    train = train.astype({"label":int})
    test = test.astype({"label":int})
    val = val.astype({"label":int})
    return (train, test, val)

def create_split(input_file):
    train, test, val = get_train_val_test_split(input_file)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    return

if __name__ == '__main__':
    create_split('data/paragraph_extracted_data/sentence_bert_processed_data.csv')
