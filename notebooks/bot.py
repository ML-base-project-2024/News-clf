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
    0: 'NYT Now',
    1: 'N.Y. / Region',
    2: 'Science',
    3: 'International Home',
    4: 'Opinion',
    5: 'Technology',
    6: 'Style',
    7: 'Your Money',
    8: 'Books',
    9: 'Watching',
    10: 'Food',
    11: 'Education',
    12: 'Times Insider',
    13: 'Health',
    14: 'Sports',
    15: 'Open',
    16: 'Job Market',
    17: 'Multimedia/Photos',
    18: 'Multimedia',
    19: 'U.S.',
    20: 'Briefing',
    21: 'Corrections',
    22: 'Well',
    23: 'Real Estate',
    24: 'The Upshot',
    25: 'Movies',
    26: 'Public Editor',
    27: 'Automobiles',
    28: 'Business Day',
    29: 'World',
    30: 'T Magazine',
    31: 'Travel',
    32: 'Today’s Paper',
    33: 'Theater',
    34: 'Magazine',
    35: 'Obituaries',
    36: 'Podcasts',
    37: 'Afternoon Update',
    38: 'Giving',
    39: 'Fashion & Style',
    40: 'The Learning Network',
    41: 'Universal',
    42: 'Homepage',
    43: 'Arts',
    44: 'Admin',
    45: 'Sunday Review',
    46: 'Blogs',
    47: 'Crosswords & Games'
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
