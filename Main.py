import pandas as pd
import numpy as np
from numpy import array, asarray, zeros

import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model
from keras import layers
# from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D, LSTM, Dropout, Dense, Flatten, Activation, Bidirectional, GRU
from keras.layers import BatchNormalization, merge, concatenate, Input, Conv1D, SpatialDropout1D, AveragePooling1D, Embedding

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import pydot
import pydotplus
from pydotplus import graphviz
from keras.utils.vis_utils import plot_model, model_to_dot

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import os

seed_value = 42
os.environ['PYTHONHASHSEED']= str(seed_value)

import random as rn

np.random.seed(seed_value)
rn.seed(seed_value)

from sklearn import metrics
from sklearn.metrics import recall_score, confusion_matrix, classification_report, multilabel_confusion_matrix


#import dataset

dataset = pd.read_csv("Dataset/Preprocess_Train_A.csv", header=None, usecols=[1,2])
dataset = dataset[dataset[1].notnull()]
dataset = dataset[dataset[2].notnull()]
# dataset.head()
# len(dataset)


# dataset distribution
import seaborn as sns

sns.countplot(x=1, data = dataset)
dataset[1].value_counts()

#split dataset
X = dataset.values[:, 1]

#target
Y = dataset.values[:, 0]

Y = np.array(list(map(lambda x: 2 if x=="positive" else 0 if x=="negative" else 1, Y)))


from tensorflow.keras.utils import to_categorical
# Y = to_categorical(Y)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X
Y_train = Y

# word embedding

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# Glove

embeddings_dictionary = dict()
glove_file = open('data_embedding/glove/glove.6B.200d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 200))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# Cross validation

data = pd.read_csv("Dataset/Preprocess_Test_A.csv", header=None, usecols=[1,2])
data = data[data[2].notnull()]
data = data[data[1].notnull()]
data.head()

Xtest = data.values[:, 1]
Ytest = data.values[:, 0]

Ytest = np.array(list(map(lambda x: 2 if x=="positive" else 0 if x=="negative" else 1, Ytest)))

Xtest = tokenizer.texts_to_sequences(Xtest)

Ytest = to_categorical(Ytest)

Xtest = pad_sequences(Xtest, padding='post', maxlen=maxlen)
len(Xtest)

#create hybrid model with CNN, LSTM, GRU

# Define the K-fold Cross Validator
kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=0)
fold_no = 1
acc_per_fold_Test=[]
loss_per_fold_Test=[]
acc_per_fold_Train = []
loss_per_fold_Train = []
results=[]

# K-fold Cross Validation model evaluation
for train, test in kfold.split(X_train, Y_train):
    Y_train_New = to_categorical(Y_train)
    
    embedding_dim = 200
    pooled_outputs1 = []
    pooled_outputs2 = []
    pooled_outputs3 = []


    embed_input = Input(shape=(maxlen,))
    dropout=0.4
    filter_sizes = [1,3,5]
    num_filters = [16,32,64]

    ###################block1##########################
#     x = Embedding(vocab_size, embedding_dim, input_length=maxlen)(embed_input)
    x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen)(embed_input)


    for i in range(len(filter_sizes)):
        conv = (Convolution1D(num_filters[i], filter_sizes[i], padding='same', activation='relu'))(x)
        conv = (MaxPooling1D(pool_size=3))(conv)
        conv = Dropout(dropout)(conv)
        pooled_outputs1.append(conv)

    merge1 = concatenate(pooled_outputs1)

    for i in range(2):
        t = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.1))(merge1)
        t = (MaxPooling1D(pool_size=3))(t)
        pooled_outputs2.append(t)

    merge2 = concatenate(pooled_outputs2)

    for i in range(2):
        x = Dense(50, activation='relu')(merge2)
        x = Bidirectional(GRU(128, return_sequences=False))(x)
        pooled_outputs3.append(x)
    merge3 = concatenate(pooled_outputs3)

    x = Flatten()(merge3)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs=[embed_input] , outputs=[x])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    history = model.fit(X_train[train], Y_train_New[train], validation_data=(X_train[test], Y_train_New[test]), epochs=4, batch_size=512)
        
    # plot the loss and accuracy in each fold Validation

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title(f'model accuracy in fold {fold_no} ...')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title(f'model loss in fold {fold_no} ...')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.show()
    
  # Generate generalization metrics
    scores = model.evaluate(Xtest, Ytest, verbose=1)
    
    preds = np.round(model.predict(Xtest))
    plrty = ['Negative', 'Neutral', 'Positive']
    results.append(recall_score(Ytest, preds, average='macro'))
    print(results)
    
    print(f'Score for fold {fold_no}: {model.metrics_names[1]} of {scores[1]}%')
    acc_per_fold_Test.append(scores[1])
    loss_per_fold_Test.append(scores[0])
    
 # save acc & loss of Train of each fold
    acc_per_fold_Train.append(history.history['accuracy'][3])
    loss_per_fold_Train.append(history.history['loss'][3])
  
  # Increase fold number
    fold_no = fold_no + 1

print('.............Avg_Recall_score.............')
print(np.mean(results))

