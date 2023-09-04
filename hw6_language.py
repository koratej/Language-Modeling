"""
Language Modeling Project
Name: Ravi Teja Konduri
RollNumber: 13
"""

import hw6_language_tests as test

project = "Language" # don't edit this

### Stage 1 ###

def loadBook(filename):
    f=open(filename,"r")
    text=f.read()
    f.close()
    corpus=[]
    lines=text.split("\n")
    for line in lines:
        lines=[]
        words=line.split(" ")
        for word in words:
            lines.append(word)
        corpus.append(lines)
    return corpus

def getCorpusLength(corpus):
    count=0
    for line in corpus:
        count +=len(line)
    return count

def buildVocabulary(corpus):
    unigrams=[]
    for line in corpus:
        for word in line:
            if word not in unigrams:
                unigrams.append(word)
    return unigrams

def count_target(corpus,target):
        count=0
        for line in corpus:
            for word in line:
                if word==target:
                    count +=1
        return count

def countUnigrams(corpus):
    unigramscount={}
    unigrams=buildVocabulary(corpus)
    for i in unigrams:
        unigramscount[i]=count_target(corpus,i)
    return unigramscount

def getStartWords(corpus):
    uniquestartwords=[]
    for line in corpus:
        if line[0] not in uniquestartwords:
            uniquestartwords.append(line[0])
    return uniquestartwords

def countStartWords(corpus):
    countstartwords={}
    allstartwords=[]
    for line in corpus:
        allstartwords.append(line[0])
    for startword in getStartWords(corpus):
        countstartwords[startword]=allstartwords.count(startword)
    return countstartwords

def countBigrams(corpus):
    bigrams={}
    for sentence in corpus:
        for index in range(len(sentence)-1):
            one=sentence[index]
            two=sentence[index+1]
            if one not in bigrams:
                bigrams[one]={}
            if two not in bigrams[one]:
                bigrams[one][two]=1
            else:
                bigrams[one][two] =bigrams[one][two]+1
    return bigrams


### Stage 2 ###

def buildUniformProbs(unigrams):
    uniformprobs=[]
    k=len(unigrams)
    for i in range (k):
        uniformprobs.append(1/k)
    return uniformprobs

def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    unigramprobs=[]
    for i in unigrams:
        unigramprobs.append(unigramCounts[i]/totalCount)
    return unigramprobs

def buildBigramProbs(unigramCounts, bigramCounts):
    bigramprobs={}
    for preWord in bigramCounts:
        words=[]
        probofwords=[]
        for key in bigramCounts[preWord]:
            words.append(key)
            probofwords.append(bigramCounts[preWord][key]/unigramCounts[preWord])
            dict={"words":words,"probs":probofwords}
        bigramprobs[preWord]=dict
    return bigramprobs

def getTopWords(count, words, probs, ignoreList):
    topwords={}
    while count !=len(topwords):
        index=probs.index(max(probs))
        if words[index] not in ignoreList:
            topwords[words[index]]=probs[index]
        probs.pop(index)
        words.pop(index)
    return topwords


from random import choices
def generateTextFromUnigrams(count, words, probs):
    # k=[]
    sentence=""
    # while count !=len(k) :
    for i in range(count):
        word=choices(words,weights=probs)
        # k.append(word)
        sentence += word[0] + " "
        sentence.strip()
    return sentence

def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    sentence=""
    k=[]
    while count !=len(k):
        if sentence=="" or k[-1]==".":
            word=choices(startWords,weights=startWordProbs)
            k.append(word[0])
            sentence += word[0] + " "
        # elif k[-1]==".":
        #     sentence.strip()
        #     word=choices(startWords,weights=startWordProbs)
        #     k.append(word[0])
        #     sentence += word[0] + " "
        else:
            if k[-1] in bigramProbs:
                word=choices(bigramProbs[k[-1]]["words"],weights=bigramProbs[k[-1]]["probs"])
                # if word==".":
                # sentence.strip()
                k.append(word[0])
                sentence += word[0] + " "
                # else:
                #     k.append(word[0])
                #     sentence += word[0] + " "
    # sentence.strip()
    return sentence


### Stage 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]
import numpy
import matplotlib
def graphTop50Words(corpus):
    unigrams=buildVocabulary(corpus)
    unigramcount=countUnigrams(corpus)
    totalcount=getCorpusLength(corpus)
    unigramprobs=buildUnigramProbs(unigrams, unigramcount,totalcount)
    return barPlot(getTopWords(50,unigrams,unigramprobs,ignore), "Top 50 Words")

def graphTopStartWords(corpus):
    startwords=getStartWords(corpus)
    startwordcount=countStartWords(corpus)
    totalcount=getCorpusLength(corpus)
    startwordprobs=buildUnigramProbs(startwords, startwordcount,totalcount)
    return barPlot(getTopWords(50,startwords,startwordprobs,ignore), "Top 50 Start Words")

def graphTopNextWords(corpus, word):
    bigramdict=buildBigramProbs(countUnigrams(corpus), countBigrams(corpus))
    return barPlot(getTopWords(10,bigramdict[word]["words"],bigramdict[word]["probs"],ignore), "Top Next Words")

def setupChartData(corpus1, corpus2, topWordCount):
    unigram1=buildVocabulary(corpus1)
    unigram2=buildVocabulary(corpus2)
    count1=countUnigrams(corpus1)
    count2=countUnigrams(corpus2)
    totalcount1=getCorpusLength(corpus1)
    totalcount2=getCorpusLength(corpus2)
    unigramprob1=buildUnigramProbs(unigram1, count1, totalcount1)
    unigramprob2=buildUnigramProbs(unigram2, count2, totalcount2)
    topunigram1=getTopWords(topWordCount, unigram1, unigramprob1, ignore)
    topunigram2=getTopWords(topWordCount, unigram2, unigramprob2, ignore)
    finalwords=[]
    for word in topunigram1:
        finalwords.append(word)
    for word in topunigram2:
        if word not in finalwords:
            finalwords.append(word)
    p1=[]
    for word in finalwords:
        if word in topunigram1:
            # print(word,unigram1)
            # index=unigram1.index(word)
            p1.append(topunigram1[word])
        else:
            p1.append(0)
    p2=[]
    for word in finalwords:
        if word in topunigram2:
            # index=unigram2.index(word)
            p2.append(topunigram2[word])
        else:
            p2.append(0)
    dict={}
    dict["topWords"]=finalwords
    dict["corpus1Probs"]=p1
    dict["corpus2Probs"]=p2
    return dict

def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    chartdata=setupChartData(corpus1, corpus2, numWords)
    return sideBySideBarPlots(chartdata["topWords"], chartdata["corpus1Probs"], chartdata["corpus2Probs"], name1, name2, title)

def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    chartdata=setupChartData(corpus1, corpus2, numWords)
    return scatterPlot(chartdata["corpus1Probs"], chartdata["corpus2Probs"], chartdata["topWords"], title)


### Stage 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt
    names = list(dict.keys())
    values = list(dict.values())
    plt.bar(names, values)
    plt.xticks(names, rotation='vertical')
    plt.title(title)
    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt
    x = list(range(len(xValues)))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    pos1 = []
    pos2 = []
    for i in x:
        pos1.append(i - width/2)
        pos2.append(i + width/2)
    rects1 = ax.bar(pos1, values1, width, label=category1)
    rects2 = ax.bar(pos2, values2, width, label=category2)
    ax.set_xticks(x)
    ax.set_xticklabels(xValues)
    ax.legend()
    plt.title(title)
    plt.xticks(rotation="vertical")
    fig.tight_layout()
    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    plt.title(title)
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)
    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work

if __name__ == "__main__":


    print("\n" + "#"*15 + " Stage 1 TESTS " +  "#" * 16 + "\n")
    test.stage1Tests()
    print("\n" + "#"*15 + " Stage 1 OUTPUT " + "#" * 15 + "\n")
    test.runStage1()


    ## Uncomment these for Stage 2 ##

    print("\n" + "#"*15 + " Stage 2 TESTS " +  "#" * 16 + "\n")
    test.stage2Tests()
    print("\n" + "#"*15 + " Stage 2 OUTPUT " + "#" * 15 + "\n")
    test.runStage2()


    ## Uncomment these for Stage 3 ##

    print("\n" + "#"*15 + " Stage 3 OUTPUT " + "#" * 15 + "\n")
    test.runStage3()