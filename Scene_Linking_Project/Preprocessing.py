import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re


def removePanctiuation(string):
    s = re.sub(r'[^\w\s]','',string)
    return s

def clean_str(string):
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub(r"\s'", "s", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"w\'d", " would", string)
    string = re.sub(r"c\'d", " could", string)
    string = re.sub(r"d'", "do ", string)
    string = re.sub(r"\'c", " could", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"won\'t", "will not", string)
    string = re.sub(r"n\'t", "not", string)
    #string = re.sub(r"...", "", string)
    #string = re.sub(r"\ ,", ",", string)
    #string = re.sub(r"\ !", "!", string)
    #string = re.sub(r"\ .", ".", string)
    #string = re.sub("([\(\[]).*?([\)\]])", "", string)
    #string = re.sub(r")", r"", string)
    #string = re.sub(r"\?", " \? ", string)
    #string = re.sub(r"\s{2,}", " ", string)
    return " ".join(string.strip().split())#.lower()

def clean_str2(string):
    string = re.sub(r"\’m", " am", string)
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\’s", " is", string)
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"won\’t", " will not", string)
    string = re.sub(r"n\’t", "not", string)
    string = re.sub(r"\’d", " would", string)
    string = re.sub(r"d’", "do ", string)
    string = re.sub(r"\’c", " could", string)
    string = re.sub(r"d\’you", "do you", string)

    return string

def countWords(sceneTexts):
    wordCount=0
    for i in range(len(sceneTexts)):
        if sceneTexts[i]!='None':
            wordCount=wordCount+len(sceneTexts[i].split(' '))
    print(wordCount/len(sceneTexts))
    maxPerScene=max(wordCount)
    minPerScene=min(wordCount)
    avgWords=wordCount/len(sceneTexts)
    return maxPerScene,minPerScene,avgWords

def sceneStemText(sceneTranscript):
    if sceneTranscript.lower()=="none" or sceneTranscript=="":
        return sceneTranscript
    txt=clean_str(sceneTranscript)
    txt=clean_str2(txt)
    txt=removePanctiuation(txt)
    wordList=txt.split()
    stemmer = SnowballStemmer("english")
    stemmedWords=[]
    for word in wordList:
        if word[-1]=='e' or word=='us' or word[0].isupper():
            stemmedWords.append(word)
        else:
            stemmedWords.append(stemmer.stem(word))
    return ' '.join(stemmedWords)

def lemitize_string(stringLem):
    stop_words = set(stopwords.words("english"))
    text = re.sub('[^a-zA-Z]', ' ',stringLem)

    #Convert to lowercase
    #text = text.lower()

    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)

    ##Convert to list from string
    text = text.split()

    ##Stemming
    #ps=PorterStemmer()    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in
            stop_words]
    text = " ".join(text)

    return text
