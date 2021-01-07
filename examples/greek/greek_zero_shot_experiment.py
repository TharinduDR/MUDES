import pandas as pd

from mudes.algo.evaluation import binary_macro_f1, binary_weighted_f1
from mudes.app.mudes_app import MUDESApp

test = pd.read_csv('data/offenseval-gr-test-v1.tsv', sep="\t")
test = test.rename(columns={'tweet': 'text'})
test = test[['text']]

test_sentences = test['text'].tolist()

app = MUDESApp("multilingual-large", use_cuda=False)

toxic_spans_list = []
toxic_word_list = []
predictions = []
for test_sentence in test_sentences:
    toxic_spans = app.predict_toxic_spans(test_sentence, spans=True, language="el")
    toxic_words = ""
    for toxic_span in toxic_spans:
        toxic_words = toxic_words + " " + test_sentence[toxic_span[0]:toxic_span[1]]
    toxic_spans_list.append(toxic_spans)
    toxic_word_list.append(toxic_words)

    if len(toxic_spans) > 0:
        predictions.append("OFF")

    else:
        predictions.append("NOT")


test["toxic_words"] = toxic_word_list
print(pd.Series(' '.join(test.toxic_words).split()).value_counts()[:20])

labels = pd.read_csv('data/offenseval-gr-labela-v1.csv', sep=",", names=['id', 'label'], header=None)

print("Macro F1 ", binary_macro_f1(labels['label'].tolist(), predictions) )
print("Weighted F1 ", binary_weighted_f1(labels['label'].tolist(), predictions) )


