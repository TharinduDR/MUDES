import os
import statistics

from sklearn.model_selection import train_test_split

from examples.english.transformer_configs import transformer_config, MODEL_TYPE, MODEL_NAME, LANGUAGE_FINETUNE, \
    language_modeling_args, TEMP_DIRECTORY
from mudes.algo.evaluation import f1
from mudes.algo.mudes_model import MUDESModel
from mudes.algo.language_modeling import LanguageModelingModel
from mudes.algo.predict import predict_spans
from mudes.algo.preprocess import read_datafile, format_data, format_lm
import torch

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

train = read_datafile('examples/english/data/tsd_train.csv')
dev = read_datafile('examples//english/data/tsd_trial.csv')


if LANGUAGE_FINETUNE:
    train_list = format_lm(train)
    dev_list = format_lm(dev)

    complete_list = train_list + dev_list
    lm_train = complete_list[0: int(len(complete_list)*0.8)]
    lm_test = complete_list[-int(len(complete_list)*0.2):]

    with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
        for item in lm_train:
            f.write("%s\n" % item)

    with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
        for item in lm_test:
            f.write("%s\n" % item)

    model = LanguageModelingModel("auto", MODEL_NAME, args=language_modeling_args, use_cuda=torch.cuda.is_available())
    model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
    MODEL_NAME = language_modeling_args["best_model_dir"]


train_df = format_data(train)
tags = train_df['labels'].unique().tolist()

model = MUDESModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=transformer_config)

if transformer_config["evaluate_during_training"]:
    train_df, eval_df = train_test_split(train_df, test_size=0.1,  shuffle=False)
    model.train_model(train_df, eval_df=eval_df)

else:
    model.train_model(train_df)

model = MUDESModel(MODEL_TYPE, transformer_config["best_model_dir"], labels=tags, args=transformer_config)


scores = []
for n, (spans, text) in enumerate(dev):
    predictions = predict_spans(model, text)
    score = f1(predictions, spans)
    scores.append(score)


print('avg F1 %g' % statistics.mean(scores))







