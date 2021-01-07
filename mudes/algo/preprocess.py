import itertools
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
import csv
import string

SPECIAL_CHARACTERS = string.whitespace


def contiguous_ranges(span_list):
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output


def fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS):
    """Applies minor edits to trim spans and remove singletons."""
    cleaned = []
    for begin, end in contiguous_ranges(spans):
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned


def read_datafile(filename):
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            fixed = fix_spans(ast.literal_eval(row['spans']), row['text'])
            data.append((fixed, row['text']))
    return data


def format_data(data: []):
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    output = []
    for n, (spans, text) in enumerate(data):
        contiguous_spans = contiguous_ranges(spans)
        # toxic_words = []
        # for contiguous_span in contiguous_spans:
        #     toxic_text = text[contiguous_span[0]:contiguous_span[1]+1]
        #     toxic_tokens = tokenizer(toxic_text)
        #     for toxic_token in toxic_tokens:
        #         toxic_words.append(toxic_token.text)
        # print(toxic_words)
        tokens = tokenizer(text)
        for token in tokens:
            is_toxic = False
            for toxic_span in contiguous_spans:
                if toxic_span[0] <= token.idx <= toxic_span[1]:
                    is_toxic = True
                    break
            if is_toxic:
                output.append([n, token.text, "TOXIC"])
            else:
                output.append([n, token.text, "NOT_TOXIC"])

    return pd.DataFrame(output, columns=['sentence_id', 'words', 'labels'])


def split_data(data: [], seed: int):
    data_frame = pd.DataFrame(data, columns=['spans', 'text'])
    print(seed)
    train_df, val_df = train_test_split(data_frame, test_size=0.2, random_state=seed)

    return train_df.to_numpy().tolist(), val_df.to_numpy().tolist()


def format_lm(data: []):
    text_list = []
    for n, (spans, text) in enumerate(data):
        text_list.append(text)
    return text_list
