{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8ea0a0bd-e53a-43ca-905c-3c4b3bc9acdc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading Packages\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore',category=DeprecationWarning)\n",
    "\n",
    "# import libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import nltk\n",
    "import matplotlib.pyplot as plt\n",
    "import re,random,os\n",
    "import seaborn as sns\n",
    "from nltk.corpus import stopwords\n",
    "import string\n",
    "from pprint import pprint as pprint\n",
    "import random\n",
    "from random import choices\n",
    "\n",
    "# spacy for basic processing, optional, can use nltk as well(lemmatisation etc.)\n",
    "import spacy\n",
    "\n",
    "#gensim for LDA\n",
    "import gensim\n",
    "import gensim.corpora as corpora\n",
    "from gensim.utils import simple_preprocess\n",
    "from gensim.models import CoherenceModel\n",
    "\n",
    "#plotting tools\n",
    "import pyLDAvis\n",
    "import pyLDAvis.gensim #dont skip this\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "21060db8-1d8f-4089-b9ac-f3051f484348",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/2m/rrfmdhf53vdb6pytyx9q_z280000gn/T/ipykernel_79416/3525326992.py:52: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.\n",
      "  df1 = pd.concat([df1, df2])\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Valence Difference</th>\n",
       "      <th>Arousal Difference</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.043045</td>\n",
       "      <td>-0.017598</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Valence Difference  Arousal Difference\n",
       "0            0.043045           -0.017598"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Loading in the movie and book reviews for The Hunger Games\n",
    "\n",
    "reviews_movie = pd.read_csv(\"/Users/easonwarren/Downloads/MSA/Text Analytics/IMDBHungerGamesReviews (1).csv\")\n",
    "\n",
    "# Randomly selecting the same number as the book revies\n",
    "random.seed(1234)\n",
    "random_500 = random.sample(range(len(reviews_movie)), 595)\n",
    "\n",
    "# Grabs the random sample of rows from the Hunger Games Movie Review\n",
    "reviews_movie = reviews_movie.iloc[random_500]\n",
    "\n",
    "reviews_book = pd.read_csv(\"/Users/easonwarren/Downloads/MSA/Text Analytics/Goodreads-HungerGames.csv\")\n",
    "\n",
    "# Randomly sampling with replacement for both sets of reviews\n",
    "bootstrap_random_movie = choices(range(595), k = 595)\n",
    "movie_bootstrap = reviews_movie.iloc[bootstrap_random_movie]\n",
    "\n",
    "bootstrap_random_book = choices(range(595), k = 595)\n",
    "book_bootstrap = reviews_book.iloc[bootstrap_random_book]\n",
    "\n",
    "# tokenize using gensims simple_preprocess\n",
    "def sent_to_words(sentences, deacc=True):  # deacc=True removes punctuations\n",
    "    for sentence in sentences:\n",
    "        yield(simple_preprocess(str(sentence)))\n",
    "        \n",
    "data_book = book_bootstrap['Title'].values.tolist()\n",
    "data_words_book = list(sent_to_words(data_book))\n",
    "\n",
    "data_movie = movie_bootstrap['Review'].values.tolist()\n",
    "data_words_movie = list(sent_to_words(data_movie))\n",
    "\n",
    "from itertools import chain\n",
    "\n",
    "book_reviews_list = list(chain(*data_words_book))\n",
    "\n",
    "movie_reviews_list = list(chain(*data_words_movie))\n",
    "\n",
    "from sentiment_module import sentiment\n",
    "\n",
    "book_sentiment = sentiment.sentiment(book_reviews_list)\n",
    "\n",
    "movie_sentiment = sentiment.sentiment(movie_reviews_list)\n",
    "\n",
    "valence_diff = book_sentiment['valence'] - movie_sentiment['valence']\n",
    "\n",
    "arousal_diff = book_sentiment['arousal'] - movie_sentiment['arousal']\n",
    "\n",
    "df1 = pd.DataFrame(columns = ['Valence Difference', 'Arousal Difference'])\n",
    "\n",
    "df2 = pd.DataFrame({'Valence Difference': [valence_diff], 'Arousal Difference': arousal_diff})\n",
    "\n",
    "df1 = pd.concat([df1, df2])\n",
    "\n",
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7ba6b6a0-61c7-45b7-93f8-bf5954a05486",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/2m/rrfmdhf53vdb6pytyx9q_z280000gn/T/ipykernel_79416/2134962256.py:30: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.\n",
      "  df1 = pd.concat([df1, df2])\n"
     ]
    }
   ],
   "source": [
    "df1 = pd.DataFrame(columns = ['Number', 'Valence Difference', 'Arousal Difference'])\n",
    "\n",
    "for i in range(1000):\n",
    "    bootstrap_random_movie = choices(range(595), k = 595)\n",
    "    movie_bootstrap = reviews_movie.iloc[bootstrap_random_movie]\n",
    "\n",
    "    bootstrap_random_book = choices(range(595), k = 595)\n",
    "    book_bootstrap = reviews_book.iloc[bootstrap_random_book]\n",
    "    \n",
    "    data_book = book_bootstrap['Title'].values.tolist()\n",
    "    data_words_book = list(sent_to_words(data_book))\n",
    "\n",
    "    data_movie = movie_bootstrap['Review'].values.tolist()\n",
    "    data_words_movie = list(sent_to_words(data_movie))\n",
    "    \n",
    "    book_reviews_list = list(chain(*data_words_book))\n",
    "\n",
    "    movie_reviews_list = list(chain(*data_words_movie))\n",
    "    \n",
    "    book_sentiment = sentiment.sentiment(book_reviews_list)\n",
    "\n",
    "    movie_sentiment = sentiment.sentiment(movie_reviews_list)\n",
    "    \n",
    "    valence_diff = book_sentiment['valence'] - movie_sentiment['valence']\n",
    "\n",
    "    arousal_diff = book_sentiment['arousal'] - movie_sentiment['arousal']\n",
    "    \n",
    "    df2 = pd.DataFrame({'Number': [i], 'Valence Difference': [valence_diff], 'Arousal Difference': arousal_diff})\n",
    "\n",
    "    df1 = pd.concat([df1, df2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "1d318a5f-2791-417d-8fcd-f93af4386cb0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Number</th>\n",
       "      <th>Valence Difference</th>\n",
       "      <th>Arousal Difference</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.017228</td>\n",
       "      <td>-0.021460</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.031812</td>\n",
       "      <td>-0.026597</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>0.050672</td>\n",
       "      <td>-0.030978</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>0.037493</td>\n",
       "      <td>-0.035049</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4</td>\n",
       "      <td>0.026574</td>\n",
       "      <td>-0.015966</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>995</td>\n",
       "      <td>0.013497</td>\n",
       "      <td>-0.008986</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>996</td>\n",
       "      <td>0.008117</td>\n",
       "      <td>0.014334</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>997</td>\n",
       "      <td>0.029880</td>\n",
       "      <td>-0.021989</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>998</td>\n",
       "      <td>0.041813</td>\n",
       "      <td>-0.016931</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>999</td>\n",
       "      <td>0.034472</td>\n",
       "      <td>-0.007405</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1000 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Number  Valence Difference  Arousal Difference\n",
       "0       0            0.017228           -0.021460\n",
       "0       1            0.031812           -0.026597\n",
       "0       2            0.050672           -0.030978\n",
       "0       3            0.037493           -0.035049\n",
       "0       4            0.026574           -0.015966\n",
       "..    ...                 ...                 ...\n",
       "0     995            0.013497           -0.008986\n",
       "0     996            0.008117            0.014334\n",
       "0     997            0.029880           -0.021989\n",
       "0     998            0.041813           -0.016931\n",
       "0     999            0.034472           -0.007405\n",
       "\n",
       "[1000 rows x 3 columns]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "996a34d2-55e1-44a5-9b36-a9ab600b2fae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df1['Valence Difference'].hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "3103e75e-cc3e-4b34-8f84-ea27c9e59a98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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M8e88cYbxu12m+XLl/dQmvOLRN6ioiLz3HPPmT179phPP/3U3H///SY5OdkcPXo0bOumoETR/v37jaSgv6w7d+40ksxnn33W4X1aW1uN2+02S5YsCYx9//33Jjk52bzwwguBsfz8fPPoo49GbO2himRWY4z5+OOPzbBhw0xtbW3UC0qks/7Yv/71LyPJHDx4MHwBQtSTef/4xz+a/v37G7/fH74AIeiJrH/729+iUlCuu+4689BDDwWNXXHFFWbevHkdzp87d6654oorgsZmzJhhxowZE7g+bdo0M3HixKA5EyZMMHfeeWeYVn1hIpH1x7Kzs60qKJHOa4wxp0+fNomJieaVV17p/oL/h7d4omjnzp1KTk7W6NGjA2NjxoxRcnKyKisrO7xPTU2N6urq5PF4AmMul0v5+fnt7vOHP/xBqampuvrqqzVnzpzA11xHQySzNjc366677tKqVavkdrsjF6KLIv17bXPy5EmtXbtWOTk5ysrKCm+IEPRUXklqbGxUUlKS4uOj8xmTPZm1J7W0tKiqqipojZLk8XjOucadO3e2mz9hwgTt3r1bfr+/0znRzB2prLbqqbzNzc3y+/1KSUkJz8LFMShRVVdXp7S0tHbjaWlp7b5I8cf3kdTuSxXT09OD7nP33XfrjTfe0LZt27RgwQK99dZbmjJlShhXH5pIZn3sscc0duxYTZ48OYwrvnCRzCr98F7y4MGDNXjwYL333nsqLy9X//79w7T60EU6b5tjx47pqaee0owZM7q54gvXU1l72jfffKMzZ86EtMa6uroO558+fVrffPNNp3OimTtSWW3VU3nnzZunn/zkJxo3blx4Fi4KSkR4vd52B6iefdm9e7ckyeFwtLu/MabD8R87+/az7/PAAw9o3LhxysvL05133qk333xTW7du1T//+c8wJPz/op31nXfe0fvvv98jB6RFO2ubu+++W3v27FFFRYVyc3M1bdo0ff/9991M154teaUfvv79lltu0VVXXaWFCxd2I1XHbMoaTaGusaP5Z4/bmjsSWW0WybxLly7VG2+8oY0bN2rAgAFhWO0PovZdPLFs5syZ5z1jZvjw4frkk0/09ddft7vtv//9b7v22qbtLYy6ujplZGQExuvr6895H0m65ppr5HQ6deDAAV1zzTVdidEl0c76/vvv69///rcuuuiioPtOnTpVN9xwg7Zt2xZCms5FO2ubtrNHcnNzNWbMGA0ZMkRlZWW66667Qo3UKVvyHj9+XBMnTtTgwYNVVlYmp9MZapTzsiVrtKSmpiouLq7d/6g7W6Pb7e5wfnx8vIYOHdrpnGjmjlRWW0U67//93/+ppKREW7du1c9+9rPwLj5sR7MgZG0H3P3jH/8IjO3atatLB9w9++yzgTGfz3fegwurq6uNJFNRURG+ACGIVNba2lpTXV0ddJFkfv/735v//Oc/kQ11Dj35e/X5fGbgwIFm7dq1YVt/qCKZt7Gx0YwZM8bk5+ebkydPRi5EF/XE7zaaB8n+5je/CRq78sorOz2Q8sorrwwae+ihh9odJFtYWBg0Z+LEiVYcJBvurD9m40Gykci7dOlSk5SUZHbu3BneBf8PBSXKJk6caH72s5+ZnTt3mp07d5oRI0a0O2Xx8ssvNxs3bgxcX7JkiUlOTjYbN2401dXV5q677go6ZfHgwYNm0aJF5qOPPjI1NTVm06ZN5oorrjC/+MUvono6aiSydkSWnGYc7qz//ve/TUlJidm9e7c5fPiwqaysNJMnTzYpKSnm66+/7tF8Z4tE3qamJjN69GgzYsQIc/DgwaBT5mPxeVxbW2v27NljXn75ZSPJbN++3ezZs8ccO3asR3K1nYq6Zs0as3//flNcXGwGDRpkDh06ZIwxZt68eeaee+4JzG87FfWxxx4z+/fvN2vWrGl3Kurf//53ExcXZ5YsWWI+/fRTs2TJEqtOMw5nVp/PZ/bs2WP27NljMjIyzJw5c8yePXvMgQMHejzf2SKR99lnnzX9+/c3b775ZtDfzePHj4dt3RSUKDt27Ji5++67TWJioklMTDR33313u/85SQr6H3Jra6tZuHChcbvdxuVymRtvvNFUV1cHbv/iiy/MjTfeaFJSUkz//v3NZZddZn7729/22AvduUQia0dsKCiRyPrll1+awsJCk5aWZpxOpxk2bJgpKio65//ce1Ik8rbtSejoUlNT0zPBOhCp5/HChQs7zNqTe8eee+45k52dbfr372+uueaaoD2u9957r8nPzw+av23bNvOLX/zC9O/f3wwfPtysXr263Tb/9Kc/mcsvv9w4nU5zxRVXmLfeeivSMbok3Flramo6/P2dvZ1oCXfe7OzsDvMuXLgwbGt2GPO/I18AAAAswVk8AADAOhQUAABgHQoKAACwDgUFAABYh4ICAACsQ0EBAADWoaAAAADrUFAAAIB1KCgAAMA6FBQAAGAdCgoAALAOBQUAAFjn/wErCP7KW4lsxwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df1['Arousal Difference'].hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "e7d95ec0-54ff-4934-8a5e-0c2da74a018f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-0.051625014799043294\n",
      "0.013881427885444149\n"
     ]
    }
   ],
   "source": [
    "print(np.quantile(df1['Arousal Difference'], 0.001))\n",
    "print(np.quantile(df1['Arousal Difference'], 0.999))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "b1cdbbcf-41fd-4ca3-b87d-5ae243e420c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.to_csv(\"/Users/easonwarren/Downloads/MSA/Text Analytics/sentiment_difference.csv\", encoding = 'utf-7')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
