from spacy.lang.ar import Arabic
from spacy.lang.da import Danish
from spacy.lang.el import Greek
from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage
from spacy.tokens import Token
import logging


class IndexedToken:
    def __init__(self, token: Token, clean_index):
        self.token = token
        self.clean_index = clean_index


def predict_spans(model, text: str, language: str = "en"):
    if language == "en":
        nlp = English()

    elif language == "el":
        nlp = Greek()

    elif language == "da":
        nlp = Danish()

    elif language == "ar":
        nlp = Arabic()

    else:
        nlp = MultiLanguage()

    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    tokens = tokenizer(text)
    sentences = []
    tokenised_text = []
    cleaned_tokens = []
    cleaned_index = 0
    for token in tokens:
        if not token.text.isspace():
            tokenised_text.append(token.text)
            indexed_token = IndexedToken(token, cleaned_index)
            cleaned_tokens.append(indexed_token)
            cleaned_index += 1
        else:
            indexed_token = IndexedToken(token, token.i)
            cleaned_tokens.append(indexed_token)

    sentences.append(tokenised_text)

    predictions, raw_outputs = model.predict(sentences, split_on_space=False)
    span_predictions = []
    sentence_prediction = predictions[0]

    for cleaned_token in cleaned_tokens:

        if cleaned_token.clean_index >= len(sentence_prediction):
            break

        if cleaned_token.token.text.isspace():
            continue

        word_prediction = sentence_prediction[cleaned_token.clean_index]
        toxicness = word_prediction[cleaned_token.token.text]
        if toxicness == "TOXIC":
            location = cleaned_token.token.idx
            if len(span_predictions) > 0:
                last_index = span_predictions[-1]
                if location == last_index + 2:
                    span_predictions.append(location - 1)
            length = len(cleaned_token.token.text)
            for i in range(length):
                span_predictions.append(location + i)
    return span_predictions
