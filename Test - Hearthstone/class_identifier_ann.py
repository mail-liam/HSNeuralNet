# -*- coding: utf-8 -*-

# Algorithm trained to identify Hearthstone card classes

# Importing the libraries
import numpy as np
import pandas as pd
import re

# Importing the dataset
dataset = pd.read_json('cards_KFT.json')

# Preprocesssing the data
dataset.drop(["artist", "classes", "collectible", "collectionText", "dbfId", "elite", "entourage", "faction", "flavor", "howToEarn", "howToEarnGolden", "id", "mechanics", "multiClassGroup", "overload", "playRequirements", "playerClass", "rarity", "referencedTags", "spellDamage", "targetingArrowText"], 
             axis = 1, 
             inplace = True)
# Remove Neutral cards & Heroes
dataset = dataset.query('cardClass != "NEUTRAL"')
dataset = dataset.query('type != "HERO"')
dataset.race = dataset.race.fillna('A')
dataset.text = dataset.text.fillna('')
dataset = dataset.fillna(0.5)

# Cleaning the texts
text = np.array(dataset.text.values.tolist())
name = np.array(dataset.name.values.tolist())
setname = np.array(dataset.set.values.tolist())
corpus = []
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
for i in range(0, len(text)): # len(text)
    card_text = re.sub('(Â\\xa0)|(<\/*[bi]>)|[,\-;\.]|(\\n)|(\[x\])', ' ', text[i])
    card_text = card_text.lower()
    card_text = card_text.split()
    card_text = [ps.stem(word) for word in card_text if not word in set(stoplist)]
    card_text = ' '.join(card_text)
    corpus.append(card_text)    

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
text_encoding = cv.fit_transform(corpus).toarray()

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
cardtype = dataset.type.values
labelencoder_type = LabelEncoder()
cardtype = labelencoder_type.fit_transform(cardtype)
OHencoder_type = OneHotEncoder()
cardtype = OHencoder_type.fit_transform(cardtype.reshape(-1, 1)).toarray()
cardtype = cardtype[:, 1:]

race = dataset.race.values
labelencoder_race = LabelEncoder()
race = labelencoder_race.fit_transform(race)
OHencoder_race = OneHotEncoder()
race = OHencoder_race.fit_transform(race.reshape(-1, 1)).toarray()
race = race[:, 1:]

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(dataset.cardClass.values)

# Merge all the independent variables
X = np.column_stack([setname, name, text_encoding, cardtype, race, dataset.cost.values, dataset.attack.values, dataset.health.values, dataset.durability.values])
# Features: [name] text, type, race, cost, attack, health, durability

# Splitting the dataset into the Training set and Test set
X_train = np.zeros([712, 322], dtype='<U32')
y_train = np.zeros([712, 1], dtype='int64')
X_test = np.zeros([81, 322], dtype='<U32')
y_test = np.zeros([81, 1], dtype='int64')
i_train = 0
i_test = 0

for i in range(0, X.shape[0]):
    if X[i][0] == "ICECROWN":
        X_test[i_test] = X[i]
        y_test[i_test] = y[i]
        i_test += 1
    else:
        X_train[i_train] = X[i]
        y_train[i_train] = y[i]
        i_train += 1

# Remove set as it is no longer needed
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
        
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 1:] = sc_X.fit_transform(X_train[:, 1:])
X_test[:, 1:] = sc_X.transform(X_test[:, 1:])
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

# Create the ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", units = 160, input_dim = 320, kernel_initializer = "uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units = 80, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="softmax", units = 9, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train[:, 1:], y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test[:, 1:])
"""max_pred = -1
for i in range(0, y_pred.shape[0]):
    max_pred = np.argmax(y_pred[i])
    for j in range(0, y_pred.shape[1]):
        y_pred[i][j] = 1 if j == max_pred else 0"""
y_pred2 = np.zeros(y_pred.shape[0], dtype= np.int8)
for i in range (0, len(y_pred2)):
    y_pred2[i] = np.argmax(y_pred[i])
y_df = pd.DataFrame({'card_name': X_test[:, 0], 'in_Test': labelencoder_y.inverse_transform(y_test), 'out_Predicted': labelencoder_y.inverse_transform(y_pred2)})

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
correct_pred = 0
for i in range(0, 9):
    correct_pred += cm[i][i]
acc = correct_pred / sum(sum(cm))

# Filter incorrect predictions
# y_dft['p_error'] = np.where(y_dft['in_Test'] == y_dft['out_Predicted'], True, False)
# incorrect = y_dft.query('p_error == False')

# Output:
# cardClass