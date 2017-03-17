#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# In[29]:

import os
import sys

import numpy as np
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential

np.random.seed(42)


# In[30]:

"""
one-directional LSTM model with character embedding

 * define a model and run for 400 iterations without early stopping,
 * store model params after each 100 iterations.
"""

# In[31]:

data_model = sys.argv[1]
print('Using data model "%s".' % data_model)

lstm_size = 90
batch_size = 32
nb_epochs = (100,) * 4

# In[32]:

data_dir = 'preprocessed_data.d/paper_runs/'
models_dir = 'lstm_models.d/paper_runs'


def load_data(data_model, fn):
    return np.loadtxt(
        os.path.join(data_dir, data_model, fn + '.txt'),
        dtype=np.dtype('int32'))


X_train = load_data(data_model, 'X_train')
X_dev = load_data(data_model, 'X_dev')
X_test = load_data(data_model, 'X_test')

y_train = load_data(data_model, 'y_train')
y_dev = load_data(data_model, 'y_dev')
y_test = load_data(data_model, 'y_test')
print('Loaded data.')

# In[33]:

vocab_size = X_train.max()  # NB: 0 is not in the vocabulary, but is padding
embedding_size = int(round(2 / 3 * vocab_size, 0))
print('Vocabulary size, embedding layer size: %d, %d' % (vocab_size,
                                                         embedding_size))

# In[44]:

model = Sequential()

model.add(Embedding(input_dim=vocab_size + 1,
                    output_dim=embedding_size,
                    mask_zero=True,
                    dropout=0.1,
                    name='Character Embedding'))

model.add(LSTM(output_dim=lstm_size,
               dropout_U=0.2,
               dropout_W=0.2,
               name='LSTM'))

model.add(Dense(output_dim=4,
                activation='softmax',
                name='Softmax'))

# In[45]:

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('Compiled model.')
print(model.summary())


# In[46]:

def save_model(i, model,
               data_model,
               with_emb=True):
    path = os.path.join(models_dir,
                        ('emb' if with_emb else 'no_emb'),
                        data_model)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # serealize model
    model_json = model.to_json()
    with open(os.path.join(path,
                           'model_%d.json' % i), 'w') as json_file:
        json_file.write(model_json)

    # serialize weights
    model.save_weights(os.path.join(path,
                                    'model_%d.h5' % i))


# In[47]:

# fitting
for i, nb_epoch in enumerate(nb_epochs):
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_dev, y_dev),
              verbose=1)

    loss, acc = model.evaluate(X_test, y_test,
                               batch_size=batch_size)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    # serialize model
    save_model(i, model, data_model, with_emb=True)

print('Learning for data model "%s" finished.' % data_model)
