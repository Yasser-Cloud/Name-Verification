#!/usr/bin/env python
# coding: utf-8

# ## That is google colab training Notebook

# In[1]:


import os
import pandas as pd
import numpy as np
import random
import json



import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split


# In[2]:


pathf = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
pathd=os.path.join(pathf ,'Datasets')


# In[6]:


new_name = pd.read_csv(os.path.join(pathd,'Nname.csv'))
new_name.sample(5)


# In[4]:


len(new_name['Name'].values)


# In[4]:


new_name = new_name.sample(frac=1).reset_index(drop=True)
new_name


# In[5]:


# Initialize the Tokenizer class
tokenizer = Tokenizer()

# Generate the word index dictionary
tokenizer.fit_on_texts(new_name['Name'].values)

# Define the total words. You add 1 for the index `0` which is just the padding token.
total_words = len(tokenizer.word_index) + 1

print(f'word index dictionary: {tokenizer.word_index}')
print(f'total words: {total_words}')


# In[6]:


with open('tokenizer.json', 'w') as fp:
    json.dump(tokenizer.word_index, fp)


# ## Preprocessing the Dataset
# 
# 

# In[7]:


# Initialize the sequences list
input_sequences = []

# Loop over every line
for line in new_name['Name'].values :

    # Tokenize the current line
    token_list = tokenizer.texts_to_sequences([line])[0]
    input_sequences.append(token_list)


# In[8]:


input_sequences[:10]


# In[27]:


print(tokenizer.word_index['غفار'])
print(list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(141)]) 


# In[10]:


max_sequence_len = max([len(x) for x in input_sequences])
max_sequence_len


# In[11]:


# Pad all sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))


# In[12]:



xs, ys = input_sequences,new_name['Real'].values


# In[13]:


60000 *.05


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.05, 
                                                    random_state=42,stratify=ys)


# In[15]:


X_train.shape , X_test.shape


# In[16]:


X_train[0]


# ## Now let's build the model

# In[17]:


model_bi = tf.keras.models.Sequential([    
    tf.keras.layers.Embedding(total_words, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[18]:


model_bi.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[19]:


mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)


# In[20]:


history = model_bi.fit(X_train, y_train, epochs=10, batch_size=128,callbacks=[es,mc],
                    validation_data=(X_test, y_test), 
                    validation_steps=10)


# In[21]:


from matplotlib import pyplot

# evaluate the model
_, train_acc = model_bi.evaluate(X_train, y_train, verbose=0)
_, test_acc = model_bi.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[22]:


# load a saved model
from tensorflow.keras.models import load_model
saved_model = load_model('best_model.h5')


# In[23]:


# evaluate the model
_, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
_, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# ## Let's predict some examples

# In[24]:


test_name = "محمود ياسر عبدالقادر"

token_list = tokenizer.texts_to_sequences([test_name])[0]

# Pad the sequence
token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='post')

# Feed to the model and get the probabilities
probabilities = saved_model.predict(token_list)

if probabilities[0][0] > 0.5:
  print(f'Real name with high confidence equal {probabilities[0][0]:.3f}')
else:
  print(f'Real name with low confidence {probabilities[0][0]:.3f}')
 


# In[25]:


test_name = "سماح محمد حامد"

token_list = tokenizer.texts_to_sequences([test_name])[0]

# Pad the sequence
token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='post')

# Feed to the model and get the probabilities 
probabilities = saved_model.predict(token_list)

if probabilities[0][0] > 0.5:
  print(f'Real name with high confidence equal {probabilities[0][0]:.3f}')
else:
  print(f'Real name with low confidence {probabilities[0][0]:.3f}')


# In[26]:


test_name = "دحلاب ابو المحاريب"

token_list = tokenizer.texts_to_sequences([test_name])[0]

# Pad the sequence
token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='post')

# Feed to the model and get the probabilities 
probabilities = saved_model.predict(token_list)

if probabilities[0][0] > 0.5:
  print(f'Real name with high confidence equal {probabilities[0][0]:.3f}')
else:
  print(f'Real name with low confidence {probabilities[0][0]:.3f}')

