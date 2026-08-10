

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

df = pd.read_csv('/content/bbc_text_cls.csv')

df.head()

#map classes to integers
df['labels'].astype("category").cat.codes

df['targets'] = df['labels'].astype("category").cat.codes

df_train, df_test = train_test_split(df,test_size=0.3)

tfidf = TfidfVectorizer(stop_words='english')
Xtrain = tfidf.fit_transform(df_train['text'])

Xtest = tfidf.transform(df_test['text'])

Ytrain = df_train['targets']
Ytest = df_test['targets']

# num of classes
K = df['targets'].max()+1

#input dimensions
D = Xtrain.shape[1]


#build model
i = Input(shape=(D,))
x = Dense(300, activation='relu')(i)
x = Dense(int(K))(x) #softmax included in loss

model = Model(i,x)

model.summary()

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = 'adam',
    metrics=['accuracy']
)

#tensorflow not accept sparse matrices
Xtrain = Xtrain.toarray()
Xtest = Xtest.toarray()

r = model.fit(
    Xtrain, Ytrain,
    validation_data=(Xtest, Ytest),
    epochs = 7,
    batch_size=128
)

#plot loss per iteration
plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.legend()

#plot accuracy per iteration
plt.plot(r.history['accuracy'], label = 'train acc')
plt.plot(r.history['val_accuracy'], label = 'val acc')
plt.legend()

df['labels'].hist()