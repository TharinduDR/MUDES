import os

from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage

from mudes.algo.mudes_model import MUDESModel
import logging
from google_drive_downloader import GoogleDriveDownloader as gdd

from mudes.algo.predict import predict_spans
from mudes.algo.preprocess import contiguous_ranges


class PredictedToken:
    def __init__(self, text, is_toxic):
        self.text = text
        self.is_toxic = is_toxic


class MUDESApp:
    def __init__(self, model_name_or_path, model_type=None,  use_cuda=True,  cuda_device=-1):

        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device

        MODEL_CONFIG = {
            "en-base": ("bert", "mudes/en-base"),
            "en-large": ("roberta", "mudes/en-large"),
            "multilingual-base": ("xlmroberta", "mudes/multilingual-base"),
            "multilingual-large": ("xlmroberta", "mudes/multilingual-large")
        }

        if model_name_or_path in MODEL_CONFIG:
            self.trained_model_type, self.model_id = MODEL_CONFIG[model_name_or_path]

            self.model = MUDESModel(self.trained_model_type, self.model_id, use_cuda=self.use_cuda,
                                        cuda_device=self.cuda_device)

        else:
            self.model = MUDESModel(model_type, self.model_name_or_path, use_cuda=self.use_cuda,
                                        cuda_device=self.cuda_device)

    @staticmethod
    def _download(drive_id, model_name):
        gdd.download_file_from_google_drive(file_id=drive_id,
                                            dest_path= os.path.join(".mudes", model_name, "model.zip"),
                                            unzip=True)

    def predict_toxic_spans(self, text: str, spans: bool = False, language: str = "en"):
        toxic_spans = predict_spans(self.model, text, language)
        if spans:
            return contiguous_ranges(toxic_spans)
        else:
            return toxic_spans

    def predict_tokens(self, text: str, language: str = "en"):
        toxic_spans = contiguous_ranges(predict_spans(self.model, text))

        if language == "en":
            nlp = English()

        else:
            nlp = MultiLanguage()

        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        tokens = tokenizer(text)
        output_tokens = []
        for token in tokens:
            is_toxic = False
            for toxic_span in toxic_spans:
                if toxic_span[0] <= token.idx <= toxic_span[1]:
                    is_toxic = True
                    break

            predicted_token = PredictedToken(token.text, is_toxic)
            output_tokens.append(predicted_token)

        return output_tokens


