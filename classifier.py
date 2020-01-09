from stemmer import Stemmer
import pickle
import math
import os
import re


class Classifier:
    def __init__(self):
        self.D = 0  # Number of documents
        self.W = []  # Unique words
        self.C = []  # Unique classes
        self.DC = {}  # Number of documents per class
        self.WC = {}  # Number of words per class
        self.WiC = {}  # Number of certain word per class

    # Remove extra characters
    def clean(self, text):
        reg = re.compile('[^а-я ]')
        text = text.lower()
        text = reg.sub(' ', text)
        return text

    # Update statistic for classifier
    def train(self, doc, category):
        # Prepare document
        doc = self.clean(doc)

        # Update classifier:
        # Update D
        self.D += 1

        # Update C & DC
        if category not in self.C:
            self.C.append(category)
            self.DC[category] = 1
        else:
            self.DC[category] += 1

        for word in doc.split():
            if len(word) > 2:
                # 'Normalize' word
                cur_word = Stemmer.stem(u'{}'.format(word))

                # Update W
                if cur_word not in self.W:
                    self.W.append(cur_word)

                # Update WC
                if category not in self.WC.keys():
                    self.WC[category] = 1
                else:
                    self.WC[category] += 1

                # Update Wic
                if category not in self.WiC.keys():
                    self.WiC[category] = {}
                if cur_word not in self.WiC[category].keys():
                    self.WiC[category][cur_word] = 1
                else:
                    self.WiC[category][cur_word] += 1

    # Predict class of document
    def predict(self, doc):
        # Prepare document
        doc = self.clean(doc)

        # Getting class with highly score
        score = []

        for cat in self.C:
            probability = math.log10(self.DC[cat] / self.D)

            for word in doc.split():
                if len(word) > 2:
                    cur_word = Stemmer.stem(u'{}'.format(word))
                    probability += math.log10( (self.WiC[cat].get(cur_word, 0) + 1) / (len(self.W) + self.WC[cat]) )

            score.append(probability)

        return self.C[score.index(max(score))]

    # Saving current state of classifier
    def save(self, dir='dump'):
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open('{}/D.pkl'.format(dir), 'wb') as handle:
            pickle.dump(self.D, handle)
        with open('{}/W.pkl'.format(dir), 'wb') as handle:
            pickle.dump(self.W, handle)
        with open('{}/C.pkl'.format(dir), 'wb') as handle:
            pickle.dump(self.C, handle)
        with open('{}/DC.pkl'.format(dir), 'wb') as handle:
            pickle.dump(self.DC, handle)
        with open('{}/WC.pkl'.format(dir), 'wb') as handle:
            pickle.dump(self.WC, handle)
        with open('{}/WiC.pkl'.format(dir), 'wb') as handle:
            pickle.dump(self.WiC, handle)

    # Loading saved state of classifier
    def load(self, dir='dump'):
        with open('{}/D.pkl'.format(dir), 'rb') as handle:
            self.D = pickle.load(handle)
        with open('{}/W.pkl'.format(dir), 'rb') as handle:
            self.W = pickle.load(handle)
        with open('{}/C.pkl'.format(dir), 'rb') as handle:
            self.C = pickle.load(handle)
        with open('{}/DC.pkl'.format(dir), 'rb') as handle:
            self.DC = pickle.load(handle)
        with open('{}/WC.pkl'.format(dir), 'rb') as handle:
            self.WC = pickle.load(handle)
        with open('{}/WiC.pkl'.format(dir), 'rb') as handle:
            self.WiC = pickle.load(handle)