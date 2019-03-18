# def prepareData():
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
import csv

types = []
xData = []
yData = []
with open('./datasets/mbti_1.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['posts']
        tokenized = text_to_word_sequence(text)
        xData.append(tokenized)
        yData.append(row['type'])

        #np.array([])
        #np unique
        #[w for text in xData for w in text]
        #if you can'tt do this on the laptop either batch the counting and filtering

# prepareData()

# Counts
# [['ENFJ' '190']
#  ['ENFP' '675']
#  ['ENTJ' '231']
#  ['ENTP' '685']
#  ['ESFJ' '42']
#  ['ESFP' '48']
#  ['ESTJ' '39']
#  ['ESTP' '89']
#  ['INFJ' '1470']
#  ['INFP' '1832']
#  ['INTJ' '1091']
#  ['INTP' '1304']
#  ['ISFJ' '166']
#  ['ISFP' '271']
#  ['ISTJ' '205']
#  ['ISTP' '337']]
