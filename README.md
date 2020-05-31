# SparknotesQA

# Running the Sparknotes models

## 1. Paragraph extraction
From the main directory, run paragraph_extraction.py to generate the paragraph extracted data. Note that for Infersent you must first download fastText vectors:

```
mkdir scripts/fastText
curl -Lo scripts/fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip scripts/fastText/crawl-300d-2M.vec.zip -d scripts/fastText/
```
Then, download the Infersent model:
```
mkdir scripts/encoder
curl -Lo scripts/encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
```

For all extraction methods, finish by running:

```
python scripts/paragraph_extraction.py
```

## 2. Split the data
Generate the train, validation, and test sets:

```
python scripts/splitdata.py
``` 

## 3. Clone the fine-tuning model
Clone the Hugging Face transformers library:

```
git clone https://github.com/huggingface/transformers.git
```

## 4. Move the split data over to Hugging Face
From the transformers directors, run:

```
mkdir data
cp sparknotesqa/data/train.csv ./data
cp sparknotesqa/data/test.csv ./data
cp sparknotesqa/data/val.csv ./data
```

## 5. Run the fine-tuning model
Finish by running python 

```
./examples/run_multiple_choice.py \
--model_type roberta \
--task_name swag \
--model_name_or_path roberta-base \
--do_train \
--do_eval \
--do_lower_case \
--data_dir ./data \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--output_dir models_bert/swag_base \
--per_gpu_eval_batch_size=16 \
--per_gpu_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output
``` 

# Running RTE 

## 1. Jiant setup

First, clone the repo:

```
git clone --branch v1.3.2  --recursive https://github.com/nyu-mll/jiant.git jiant
```

Follow the steps here: https://github.com/nyu-mll/jiant/edit/master/tutorials/setup_tutorial.md to finish setup. Once setup, copy superGLUE_RTE.jsonl from sparknotesqa/data to the data directory you created inside the RTE directory and copy superglue_exp.conf to /jiant/jiant/config.

## 2. Run BERT model
To run the model -

```sh
python main.py --config_file jiant/config/superglue_exp.conf \
    --overrides "exp_name = my_exp, run_name = foobar"
```
