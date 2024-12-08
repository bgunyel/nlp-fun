import os
from pydantic import BaseModel

from source.config import settings


class ModelConfig(BaseModel):
    name: str
    pre_tokenizer: str


class TokenizationModel:
    def __init__(self):
        self.language_data = {
            'tur': [
                'afr-tur', 'amh-tur', 'ara-tur', 'asm-tur', 'aze-tur',
                'bak-tur', 'bar-tur', 'bel-tur', 'bul-tur',
                'cat-tur', 'ces-tur', 'chv-tur', 'crh-tur',
                'dan-tur', 'deu-tur',
                'ell-tur', 'eng-tur', 'epo-tur', 'est-tur',
                'fao-tur', 'fas-tur', 'fin-tur', 'fra-tur', 'fry-tur',
                'gla-tur', 'gle-tur', 'glg-tur',
                'hat-tur', 'hbs-tur', 'heb-tur', 'hin-tur', 'hun-tur', 'hye-tur',
                'ido-tur', 'ina-tur', 'isl-tur', 'ita-tur',
                'jpn-tur',
                'kat-tur', 'kaz-tur', 'kor-tur', 'kur-tur',
                'lat-tur', 'lav-tur', 'lit-tur',
                'nld-tur',
                'pol-tur', 'por-tur',
                'ron-tur', 'rus-tur',
                'spa-tur', 'swe-tur',
                'tat-tur',
                'tur-tur', 'tur-uig', 'tur-ukr', 'tur-uzb', 'tur-yid', 'tur-zho',
                ]
        }

    def get_train_data(self, language_code: str):
        if language_code not in self.language_data.keys():
            raise RuntimeError(f'Language code {language_code} is not supported.')

        out = []
        for lang_pair in self.language_data[language_code]:
            idx = lang_pair.find('-')
            if language_code == lang_pair[: idx]:
                out.append(os.path.join(settings.DATA_FOLDER, 'tatoeba', lang_pair, 'train.src'))
            if language_code == lang_pair[idx+1 :]:
                out.append(os.path.join(settings.DATA_FOLDER, 'tatoeba', lang_pair, 'train.trg'))
        return out
