import pandas as pd

from mudes.algo.evaluation import binary_macro_f1, binary_weighted_f1
from mudes.app.mudes_app import MUDESApp


test = pd.read_csv('data/test_a_tweets.tsv', sep="\t")
test = test.rename(columns={'tweet': 'text'})
test = test[['text']]

test_sentences = test['text'].tolist()

app = MUDESApp("small", use_cuda=False)

toxic_spans_list = []
predictions = []
for test_sentence in test_sentences:
    toxic_spans = app.predict_toxic_spans(test_sentence, spans=True)
    toxic_spans_list.append(toxic_spans)

    if len(toxic_spans) > 0:
        predictions.append("OFF")

    else:
        predictions.append("NOT")

labels = pd.read_csv('data/test_a_labels.csv', sep=",", names=['id', 'label'], header=None)
print("Macro F1 ", binary_macro_f1(labels['label'].tolist(), predictions) )
print("Weighted F1 ", binary_weighted_f1(labels['label'].tolist(), predictions) )


