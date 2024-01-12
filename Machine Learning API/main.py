import csv
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from nltk.tokenize import WordPunctTokenizer
import itertools
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask import Flask, request, jsonify

NUM_WORDS = 30000
EMBEDDING_DIM = 16
MAXLEN = 50
PADDING = 'post'
OOV_TOKEN = "<OOV>"
#TRAIN_FILE = "./dataset/processed_train.csv"  # cloud storage url
STOPWORDS_FILE = "./src/stopwordbahasa.csv"  # cloud storage url
STOPWORDS = []

EXTERNAL_DATA_PATH = './src/external'

# Translate emoticon
emoticon_data_path = '{}/emoticon.txt'.format(EXTERNAL_DATA_PATH)
emoticon_df = pd.read_csv(emoticon_data_path, sep='\t', header=None)
emoticon_dict = dict(zip(emoticon_df[0], emoticon_df[1]))

def translate_emoticon(t):
    for w, v in emoticon_dict.items():
        pattern = re.compile(re.escape(w))
        match = re.search(pattern,t)
        if match:
            t = re.sub(pattern,v,t)
    return t

def remove_newline(text):
    text = re.sub(r'\\n', ' ',text)
    return re.sub(r'\n', ' ',text)

def remove_kaskus_formatting(text):
    text = re.sub('\[', ' [', text)
    text = re.sub('\]', '] ', text)
    text = re.sub('\[quote[^ ]*\].*?\[\/quote\]', ' ', text)
    text = re.sub('\[[^ ]*\]', ' ', text)
    text = re.sub('&quot;', ' ', text)
    text = text.strip()
    return text

def remove_url(text):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)

def remove_excessive_whitespace(text):
    return re.sub('  +', ' ', text)

def tokenize_text(text, punct=False):
    text = WordPunctTokenizer().tokenize(text)
    text = [word for word in text if punct or word.isalnum()]
    text = ' '.join(text)
    text = text.strip()
    return text

slang_words = pd.read_csv('{}/slangword.csv'.format(EXTERNAL_DATA_PATH))
slang_dict = dict(zip(slang_words['original'],slang_words['translated']))

def transform_slang_words(text):
    word_list = text.split()
    word_list_len = len(word_list)
    transformed_word_list = []
    i = 0
    while i < word_list_len:
        if (i + 1) < word_list_len:
            two_words = ' '.join(word_list[i:i+2])
            if two_words in slang_dict:
                transformed_word_list.append(slang_dict[two_words])
                i += 2
                continue
        transformed_word_list.append(slang_dict.get(word_list[i], word_list[i]))
        i += 1
    return ' '.join(transformed_word_list)

def remove_non_alphabet(text):
    output = re.sub('[^a-zA-Z ]+', '', text)
    return output

def remove_twitter_ig_formatting(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = text.strip()
    return text

def remove_repeating_characters(text):
    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))

def load_stopwords(stopwords_file=STOPWORDS_FILE):
    with open(stopwords_file, 'r') as f:
        stopwords = []
        reader = csv.reader(f)
        for row in reader:
            stopwords.append(row[0])

        return stopwords


STOPWORDS = load_stopwords(STOPWORDS_FILE)


def remove_stopwords(sentence, stopwords=STOPWORDS):
    sentence = sentence.lower()

    words = sentence.split()
    no_words = [w for w in words if w not in stopwords]
    sentence = " ".join(no_words)

    return sentence

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(sentence, stemmer=stemmer):
    return stemmer.stem(sentence)

def preprocess_text(text):
    transformed_text = text.lower()
    transformed_text = remove_newline(transformed_text)
    transformed_text = remove_url(transformed_text)
    transformed_text = remove_twitter_ig_formatting(transformed_text)
    transformed_text = remove_kaskus_formatting(transformed_text)
    transformed_text = translate_emoticon(transformed_text)
    transformed_text = transformed_text.lower()
    transformed_text = tokenize_text(transformed_text)
    transformed_text = transform_slang_words(transformed_text)
    transformed_text = remove_repeating_characters(transformed_text)
    transformed_text = transform_slang_words(transformed_text)
    transformed_text = remove_non_alphabet(transformed_text)
    transformed_text = remove_excessive_whitespace(transformed_text)
    transformed_text = transformed_text.lower().strip()
    transformed_text = remove_stopwords(transformed_text)
    transformed_text = stemming(transformed_text)
    return transformed_text


def parse_data_from_file(filename):
    sentences = []
    labels = []
    with open(filename, 'r', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            sentence = row[6]
            sentence = remove_stopwords(sentence)
            sentences.append(sentence)

    return sentences

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def seq_and_pad(sentences, tokenizer, padding, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding=padding, maxlen=maxlen)

    return padded_sequences


model = tf.keras.models.load_model('toxic_comment_model_var_1_v3.h5', compile=False)

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        text = request.form['sentence']
        if text is None:
            return jsonify({"error": "No text"})
        
        try:
            sentence = text
            sentence = preprocess_text(sentence)
            sentences = []
            sentences.append(sentence)

            sentences_padded_seq = seq_and_pad(sentences, tokenizer, PADDING, MAXLEN)

            prediction = model.predict(sentences_padded_seq)
            data = {"prediction": float(prediction),
                    "text" : str(text)}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)
