import pandas as pd
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import re


# # Import Dataset

data = pd.read_csv("Dataset/SemEval2017-task4-test.subtask-A.english.txt", header=0, sep="	", usecols=[1,2], encoding='latin-1')
data = data[data['Content'].notnull()]
data = data[data['Polarity'].notnull()]


# data.shape


# # Remove HTML

def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    html_free = soup.get_text()
    return html_free

data['Content'] = data['Content'].apply(lambda x:remove_html(x))


# # Remove http


def remove_http(text):
    no_http = re.sub(r"http\S+", '', text)
    return no_http

data['Content'] = data['Content'].apply(lambda x:remove_http(x))


# # Remove Name Entity

# Function to reverse tokenization
def untokenize(tokens):
    return("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip())

# Remove named entities
def remove_nameEntity(text):
    tokens = nltk.word_tokenize(text)
    chunked = nltk.ne_chunk(nltk.pos_tag(tokens))
    tokens = [leaf[0] for leaf in chunked if type(leaf) != nltk.Tree]
    return(untokenize(tokens))

data['Content'] = data['Content'].apply(lambda x:remove_nameEntity(x))

# data.head(25)


# # Remove punctuation:

def remove_punc(text):
    no_p = "". join([c for c in text if c not in string.punctuation])
    return no_p

data['Content'] = data['Content'].apply(lambda x:remove_punc(x))

# data.columns


# # Remove Digits

def remove_digit(text):
    re_digit = re.sub(r"\w*\d\w*", ' ', text)
    return re_digit

data['Content'] = data['Content'].apply(lambda x:remove_digit(x))

# data.head(25)


# # Tokenize:

# Instantiate Tokenizer
tokenizer = RegexpTokenizer(r'\w+')


# ### converting all letters to lower case

data['Content'] = data['Content'].apply(lambda x:tokenizer.tokenize(x.lower()))


# # Remove stop words:

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

data['Content'] = data['Content'].apply(lambda x: remove_stopwords(x))


# # Remove words that length is less than 3

def remove_less3word(text):
    words = [w for w in text if len(w)>=3]
    return words

data['Content'] = data['Content'].apply(lambda x: remove_less3word(x))


# # Lemmatizing:

# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

data['Content'] = data['Content'].apply(lambda x: word_lemmatizer(x))


# # Stemming

# Instantiate Stemmer
stemmer = PorterStemmer()

def word_stemmer(text):
    stem_text = " ". join([stemmer.stem(i) for i in text])
    return stem_text


data['Content'] = data['Content'].apply(lambda x: word_stemmer(x))

# data.head(20)


# # Correcting

from textblob import TextBlob

data['Content'] = data['Content'].apply(lambda x: str(TextBlob(x).correct()))

# data.to_csv("Dataset/Preprocess_Train_A.csv")
data.to_csv("Dataset/Preprocess_Test_A.csv")

