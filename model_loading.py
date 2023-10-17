# model_loading.py

import joblib
import text_preprocessing as tp
import fasttext


def load_models():
    models = {
        'BoW + SVM': {
            'model': joblib.load('./bow/svc_bow_model.joblib'),
            'vectorizer': joblib.load('./bow/vectorizer.joblib'),
            'preprocessor': None  # or specify a preprocessor function if needed
        },
        'BoW + Lemmatization + SVM': {
            'model': joblib.load('./bow_lemm/svc_bow_lemm_model.joblib'),
            'vectorizer': joblib.load('./bow_lemm/vectorizer_lemm.joblib'),
            'preprocessor': tp.lemmatize_sentence  # assuming lemmatize_sentence is defined
        },
        'FastText': {
            'model': fasttext.load_model('./fasttext/fasttext_model.bin'),
            'vectorizer': None,  # No vectorizer needed for FastText
            'preprocessor': None  # or specify a preprocessor function if needed
        },
        
        # ... add more models...
    }
    return models

