import argparse
import sys
from metrics import macro_f1_score
from utils import find_best_checkpoint, clean
from transformers_models import MODEL_CLASSES
from simpletransformers.classification import ClassificationModel
from utils_1 import get_preprocessed_data, labels
from models_list import get_models
from config import get_fine_tuning_args, global_config
#sys.path.append(1,'/content/drive/MyDrive/Colab Notebooks/Project_NLP/depression-detection-lt-edi-2022/dataset')
#from dataset.utils import get_preprocessed_data, labels

import pandas as pd
from sklearn.utils import shuffle


labels = ['severe', 'moderate', 'not depression']


text_column_names = {
    "dev": "Text data",
    "train": "Text_data",
    "test": "text data"
}

pid_column_names = {
    "dev": "PID",
    "train": "PID",
    "test": "Pid"
}


def get_data(data_split, use_shuffle=False, without_label=False):
    df = pd.read_csv(f'../data/original_dataset/{data_split}.tsv', sep='\t', header=0)

    pid_column = pid_column_names.get(data_split)
    text_column = text_column_names.get(data_split)
    label_column = "Label"
    if without_label:
        df = df[[pid_column, text_column]]
        df.columns = ["pid", "text"]
    else:
        df[label_column] = df[label_column].transform(lambda label: labels.index(label))
        df.columns = ["pid", "text", "labels"]
    if use_shuffle:
        return shuffle(df)
    return df


def get_preprocessed_data(data_split, use_shuffle=False):
    df = pd.read_csv(f'../data/preprocessed_dataset/{data_split}.csv', header=0, lineterminator='\n')
    if use_shuffle:
        return shuffle(df)
    return df


def fine_tune():
    print(f'Fine-tuning\t{model_info.description()}')

    train_data = get_preprocessed_data("train_multilingual_robust", use_shuffle=True)
    eval_data = get_preprocessed_data("dev_1")

    model = ClassificationModel(
        model_info.model_type,
        model_info.get_model_path(),
        num_labels=len(labels),
        args=model_args,
    )

    set_dropout(model)
    model.train_model(train_data, eval_df=eval_data, f1=macro_f1_score)

    best_checkpoint = find_best_checkpoint(model_args.output_dir)
    clean(model_args.output_dir, best_checkpoint)


def set_dropout(model: ClassificationModel):
    config = model.config
    config.attention_probs_dropout_prob = dropout.att_dropout
    config.hidden_dropout_prob = dropout.h_dropout
    config.classifier_dropout = dropout.c_dropout

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_info.model_type]
    model.model = model_class.from_pretrained(model_info.get_model_path(), config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="basic")
    args = parser.parse_args()

    for model_info in get_models(args.models):
        for version in range(global_config.runs):
            model_info.model_version = f'v{version + 1}'
            model_args, dropout = get_fine_tuning_args(model_info)

            fine_tune()
