
# coding: utf-8

# In[1]:

import json
import sys
import os
import numpy as np
np.random.seed(42)

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding, LSTM, Masking
from keras.callbacks import EarlyStopping


# In[2]:

"""
one-directional LSTM model without character embedding

 * define a model and run for 400 iterations without early stopping,
 * store model params at every 100 iterations.
""";


# In[3]:

data_model = sys.argv[1]
print('Using data model "%s".' % data_model)

lstm_size = 90
batch_size = 32
nb_epochs = (100,)*4


# In[49]:

data_dir = '/mnt/storage/clfiles/users/makarov/preprocessed_data/paper_runs/'
models_dir = '/mnt/storage/clfiles/users/makarov/models/paper_runs'

# one-hot encode data as dense vectors
def encode_data(X, vocab_size):
    num_samples, num_timesteps = X.shape
    X_enc = np.zeros((num_samples, num_timesteps, vocab_size),
                     dtype=np.int8)
    # how to do it with numpy indexing?
    for i in ((s, t, f - 1) for s in range(num_samples)
              for t, f in enumerate(X[s]) if f != 0):
        X_enc[i] = 1
    return X_enc

def load_data(data_model, fn):
    return np.loadtxt(
        os.path.join(data_dir, data_model, fn + '.txt'),
        dtype=np.dtype('int32'))

X_train = load_data(data_model, 'X_train')
vocab_size = X_train.max()  # NB: 0 is not in the vocabulary, but is padding
seq_length = X_train.shape[1]
print('Vocabulary size, sequence length: %d, %d' % (vocab_size, seq_length))

X_train = encode_data(X_train, vocab_size)
X_dev = encode_data(load_data(data_model, 'X_dev'), vocab_size)
X_test = encode_data(load_data(data_model, 'X_test'), vocab_size)

y_train = load_data(data_model, 'y_train')
y_dev = load_data(data_model, 'y_dev')
y_test = load_data(data_model, 'y_test')
print('Loaded data.')


# In[63]:

model = Sequential()

model.add(Masking(mask_value=0.,
                  input_shape=(seq_length,
                               vocab_size),
                  name='Masking'))

model.add(LSTM(output_dim=lstm_size,
               dropout_U=0.2,
               dropout_W=0.2,
               name='LSTM'))

model.add(Dense(output_dim=4,
                activation='softmax',
                name='Softmax'))


# In[64]:

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
        os.mkdir(path)
    
    # serealize model
    model_json = model.to_json()
    with open(os.path.join(path,
        'model_%d.json' % i), 'w') as json_file:
        json_file.write(model_json)

    # serialize weights
    model.save_weights(os.path.join(path,
        'model_%d.h5' % i))


# In[65]:

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
    save_model(i, model, data_model, with_emb=False)

