# -*- coding: utf-8 -*-

# Algorithm trained to identify Hearthstone card classes

# Importing the libraries
import numpy as np
import pandas as pd
import re

# Importing the dataset
dataset = pd.read_json('cards.json')

# Preprocesssing the data
dataset.drop(["artist", "classes", "collectible", "collectionText", "dbfId", "elite", "entourage", "faction", "flavor", "howToEarn", "howToEarnGolden", "id", "mechanics", "multiClassGroup", "overload", "playRequirements", "playerClass", "rarity", "referencedTags", "set", "spellDamage", "targetingArrowText"], 
             axis = 1, 
             inplace = True)
# Remove Neutral cards & Heroes
dataset = dataset.query('cardClass != "NEUTRAL"')
dataset = dataset.query('type != "HERO"')
y = dataset.cardClass.values
dataset.race = dataset.race.fillna('A')
dataset = dataset.fillna(-1)

# Cleaning the texts
dataset.text = dataset.text.fillna('')
text = np.array(dataset["text"].values.tolist())
name = np.array(dataset.name.values.tolist())
corpus = []
#for i in range(0, len(text)):
#    card_text = re.sub('(Â\\xa0)|(<\/*[bi]>)|[,\-;\.]|(\\n)|(\[x\])', ' ', text[i])
#    corpus.append(card_text.lower())
#    print(corpus[i])

# nltk.download('stopwords')
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
cv = CountVectorizer(max_features = 1500)
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
y = labelencoder_y.fit_transform(y)

# Merge all the independent variables
X = np.column_stack([name, text_encoding, cardtype, race, dataset.cost.values, dataset.attack.values, dataset.health.values, dataset.durability.values])
# Features: [name] text, type, race, cost, attack, health, durability

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train[:, 1:] = sc_X.fit_transform(X_train[:, 1:])
#X_test[:, 1:] = sc_X.transform(X_test[:, 1:])
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_train[:, 1:], y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test[:, 1:])
y_df = pd.DataFrame({'card name': X_test[:, 0], 'in_Test': labelencoder_y.inverse_transform(y_test), 'out_Predicted': labelencoder_y.inverse_transform(y_pred)})

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
correct_pred = 0
for i in range(0, 9):
    correct_pred += cm[i][i]
acc = correct_pred / sum(sum(cm))



# Output:
# cardClass