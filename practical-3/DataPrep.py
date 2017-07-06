import xml.etree.ElementTree as etree
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
from nltk.text import FreqDist


def initData():
    allText = []
    setOfWords = {}

    # Open file and get root
    tree = etree.parse('ted_en-20160408.xml')
    root = tree.getroot()

    # Find all file children
    for file in root.findall('file'):
        # Get the child's keywords
        keywords = file.find('head').find('keywords')

        # Get the child's text
        content = file.find('content')

        allText.append(content.text)

    allTextClean = []
    for doc in allText:
        for word in word_tokenize(doc):
            if word not in stopwords.words('english') and word not in list(string.punctuation) and len(word) > 5:
                allTextClean.append(word.lower())

    fdist = FreqDist(allTextClean)
    commonWordTuples = fdist.most_common(512)

    commonWords = [wordPair[0] for wordPair in commonWordTuples]

    i = 0
    for word in set(commonWords):
        setOfWords[word] = i
        i += 1

    return setOfWords


# Returns a list of tuples containing the text and it's associated keywords
def getLabelsAndText(setOfWords):
    labelsText = []

    # Open file and get root
    tree = etree.parse('ted_en-20160408.xml')
    root = tree.getroot()

    # Find all file children
    for file in root.findall('file'):
        # Get the child's keywords
        keywords = file.find('head').find('keywords')

        # Get the child's text
        content = file.find('content')

        labelsText.append((formatKeywords(keywords.text), formatText(content.text, setOfWords)))

    return labelsText


# Formats the keyword into the proper string representation of the classes
def formatKeywords(keywordStr):
    keywordStr = keywordStr.lower()

    newStr = ''
    if 'technology' in keywordStr:
        newStr += 'T'
    else:
        newStr += 'o'

    if 'entertainment' in keywordStr:
        newStr += 'E'
    else:
        newStr += 'o'

    if 'design' in keywordStr:
        newStr += 'D'
    else:
        newStr += 'o'

    return keyVector(newStr)


def keyVector(keyStr):
    place = 0
    if keyStr == 'Too':
        place = 1
    elif keyStr == 'oEo':
        place = 2
    elif keyStr == 'ooD':
        place = 3
    elif keyStr == 'TEo':
        place = 4
    elif keyStr == 'ToD':
        place = 5
    elif keyStr == 'oED':
        place = 6
    elif keyStr == 'TED':
        place = 7

    ary = np.zeros(8)
    ary[place] = 1

    return ary


# Formats text into a list of integers
def formatText(textStr, setOfWords):
    ary = []
    wordList = [word.lower() for word in word_tokenize(textStr) if
                word not in stopwords.words('english') and word not in list(
                    string.punctuation)]

    for word in wordList:
        if word in setOfWords:
            ary.append(setOfWords[word])

    return np.array(ary)
