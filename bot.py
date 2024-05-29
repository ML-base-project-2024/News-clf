import logging
import pickle
import re

import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from nltk import PorterStemmer
from sklearn.base import BaseEstimator

stemmer = PorterStemmer()
targets = {
    0: 'Arts',
    1: 'Sports',
    2: 'Health',
    3: 'Sunday Review',
    4: 'Your Money',
    5: 'Times Insider',
    6: 'Public Editor',
    7: 'Travel',
    8: 'Movies',
    9: 'The Upshot',
    10: 'Science',
    11: 'Technology',
    12: 'The Learning Network',
    13: 'Admin',
    14: 'Giving',
    15: 'Open',
    16: 'Food',
    17: 'Crosswords & Games',
    18: 'Corrections',
    19: 'Theater',
    20: 'Automobiles',
    21: 'Job Market',
    22: 'Homepage',
    23: 'NYT Now',
    24: 'U.S.',
    25: 'Magazine',
    26: 'T Magazine',
    27: 'Well',
    28: 'Blogs',
    29: 'Education',
    30: 'Today’s Paper',
    31: 'Multimedia/Photos',
    32: 'Briefing',
    33: 'Obituaries',
    34: 'Books',
    35: 'Opinion',
    36: 'Real Estate',
    37: 'Universal',
    38: 'International Home',
    39: 'Podcasts',
    40: 'Watching',
    41: 'Style',
    42: 'N.Y. / Region',
    43: 'Afternoon Update',
    44: 'Business Day',
    45: 'Multimedia',
    46: 'Fashion & Style',
    47: 'World'
}

def preprocess_text(text):
    return re.sub(r"[^\w\s]+", '', text).lower().split()


def preprocess_sentence_eng(text):
    return ' '.join(map(stemmer.stem, preprocess_text(text)))


class StemmerEng(BaseEstimator):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self, x, y=None):
        return self

    def _stem(self, word):
        return self.stemmer.stem(word)

    def _transform_text(self, text):
        return ' '.join(map(self._stem, text.split()))

    def transform(self, x):
        return list(map(self._transform_text, x))


class BasePreprocessor(BaseEstimator):
    """
    Класс для базовой обработки текста - нижний регистр и пунктуация
    """

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.array(list(map(lambda x: ' '.join(preprocess_text(x)), x)))

model_path = 'mlp48.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Настройка логирования
logging.basicConfig(level=logging.INFO)

TOKEN = '7330334149:AAGqOrSo-D0z79mHxv6SuGIwN6uyvippdKw'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.reply("Привет! Отправьте мне текст новости, и я определю её категорию.")


@dp.message_handler()
async def classify_news(message: types.Message):
    text = message.text
    prediction = model.predict([text])[0]
    await message.reply(f"Категория новости: {targets[prediction]}")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
