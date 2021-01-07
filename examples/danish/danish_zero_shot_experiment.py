import pandas as pd

from mudes.algo.evaluation import binary_macro_f1, binary_weighted_f1
from mudes.app.mudes_app import MUDESApp

test = pd.read_csv('data/offenseval-da-test-v1.tsv', sep="\t")
test = test.rename(columns={'tweet': 'text'})


test_sentences = test['text'].tolist()

app = MUDESApp("multilingual-large", use_cuda=False)

hate_spans_list = []
hate_word_list = []
predictions = []
for test_sentence in test_sentences:
    hate_words = ""
    hate_spans = app.predict_toxic_spans(test_sentence, spans=True, language="da")
    for hate_span in hate_spans:
        hate_words = hate_words + " " + test_sentence[hate_span[0]:hate_span[1]]
    hate_spans_list.append(hate_spans)
    hate_word_list.append(hate_words)
    if len(hate_spans) > 0:
        predictions.append("OFF")

    else:
        predictions.append("NOT")

test["hate_words"] = hate_word_list

print(pd.Series(' '.join(test.hate_words).split()).value_counts()[:20])
print("Macro F1 ", binary_macro_f1(test['subtask_a'].tolist(), predictions) )
print("Weighted F1 ", binary_weighted_f1(test['subtask_a'].tolist(), predictions) )


