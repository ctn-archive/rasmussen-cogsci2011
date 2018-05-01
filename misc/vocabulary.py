"""Generates the vocabularies used in the model."""

import time
import math

from java.lang import System
from java.io import File

from ca.nengo.math import PDFTools
from ca.nengo.math.impl import GaussianPDF

from misc import RPMutils

def genVocab(d, numwords, seed):
    """Calls the appropriate function for the given number of words in vocab."""
    
    if numwords == 20:
        genVocab20(d, seed)
    elif numwords == 50:
        genVocab50(d, seed)
    elif numwords == 80:
        return genVocab80(d,seed)
    else:
        System.err.println(str(numwords) + " is not a supported vocabulary")

def saveVocab(d, numwords, seed, filename):
    """Saves the vocabulary to file."""
    
    vocab = genVocab(d, numwords, seed)
    
    output = open(filename, "w")
    for name,val in vocab:
        if name:
            output.write(name + " " + RPMutils.floatlist2str(val) + "\n")
    output.close()

def genVocab80(d, seed):
    """Vocabulary for 80 base words."""
    
    numwords = 80
    vocab = [[None,None] for i in range(numwords)]
    
    PDFTools.setSeed(seed)
    vocab = fillVectors(vocab, d, numwords, RPMutils.VECTOR_SIMILARITY)
    PDFTools.setSeed(long(time.time()))
    
    #set the null vector
    vocab[0][1] = [0.0 for i in range(d)]
    
    util = ["null"]
    
    attributes = ["shape",
                 "number",
                 "linestroke",
                 "angle",
                 "length",
                 "width",
                 "location",
                 "shading",
                 "existence",
                 "linetype",
                 "radialpos",
                 "portion",
                 "skew",
                 "misc"]
    
    values = [#generic
              "present",
             "rubbish",
             "abstract1",
             "abstract2",
             "abstract3",
            
             #number
             "one",
             "plusone",
            
             #angle
             "0deg",
             "plus45deg",
            
             #shape
             "circle",
             "square",
             "diamond",
             "dot",
             "triangle",
             "cross",
             "X",
             "rectangle",
             "hexagon",
             "line",
             
             #linestroke
             "normal",
             "dashed",
             "bold",
             
             #linetype
             "straight",
             "curved",
             "wavy",
             
             #length/width
             "short",
             "longer",
             
             #location
             "NW",
             "moveE",
             "moveS",
             
             #shading
             "none",
             "solid",
             "lefthatch",
             "righthatch",
             "crosshatch",
             "vertical",
             "horizontal",
             "dots",
             
             #radialpos
             "inner",
             "moveout",
             
             #portion
             "lefthalf",
             "righthalf",
             "tophalf",
             "bottomhalf",
             
             #skew
             "left",
             "right"]
    
    names = util + attributes + values
    
    if len(names) > numwords:
        System.err.println("Uh oh, added more words to vocabulary than can fit!")
    
    for i,name in enumerate(names):
        vocab[i][0] = name
    
    #now the higher level vocab
    norm = RPMutils.normalize
    cconv = RPMutils.cconv
    vecsum = RPMutils.vecsum
    
    #number
    vocab += [["two", norm(cconv(vocabVal("one",vocab), vocabVal("plusone",vocab)))]]
    vocab += [["three", norm(cconv(vocabVal("two",vocab), vocabVal("plusone",vocab)))]]
    vocab += [["four", norm(cconv(vocabVal("three",vocab), vocabVal("plusone",vocab)))]]
    vocab += [["five", norm(cconv(vocabVal("four",vocab), vocabVal("plusone",vocab)))]]
    vocab += [["six", norm(cconv(vocabVal("five",vocab), vocabVal("plusone",vocab)))]]
    
    #angle
    vocab += [["45deg", norm(cconv(vocabVal("0deg",vocab), vocabVal("plus45deg",vocab)))]]
    vocab += [["90deg", norm(cconv(vocabVal("45deg",vocab), vocabVal("plus45deg",vocab)))]]
    vocab += [["135deg", norm(cconv(vocabVal("90deg",vocab), vocabVal("plus45deg",vocab)))]]
    vocab += [["180deg", norm(cconv(vocabVal("135deg",vocab), vocabVal("plus45deg",vocab)))]]
    vocab += [["225deg", norm(cconv(vocabVal("180deg",vocab), vocabVal("plus45deg",vocab)))]]
    vocab += [["270deg", norm(cconv(vocabVal("225deg",vocab), vocabVal("plus45deg",vocab)))]]
    vocab += [["315deg", norm(cconv(vocabVal("270deg",vocab), vocabVal("plus45deg",vocab)))]]
    
    #location
    vocab += [["N", norm(cconv(vocabVal("NW",vocab), vocabVal("moveE",vocab)))]]
    vocab += [["NE", norm(cconv(vocabVal("N",vocab), vocabVal("moveE",vocab)))]]
    vocab += [["W", norm(cconv(vocabVal("NW",vocab), vocabVal("moveS",vocab)))]]
    vocab += [["C", norm(cconv(vocabVal("W",vocab), vocabVal("moveE",vocab)))]]
    vocab += [["E", norm(cconv(vocabVal("C",vocab), vocabVal("moveE",vocab)))]]
    vocab += [["SW", norm(cconv(vocabVal("W",vocab), vocabVal("moveS",vocab)))]]
    vocab += [["S", norm(cconv(vocabVal("SW",vocab), vocabVal("moveE",vocab)))]]
    vocab += [["SE", norm(cconv(vocabVal("S",vocab), vocabVal("moveE",vocab)))]]
    
    #width/length
    vocab += [["medium", norm(cconv(vocabVal("short",vocab), vocabVal("longer",vocab)))]]
    vocab += [["long", norm(cconv(vocabVal("medium",vocab), vocabVal("longer",vocab)))]]
    
    #radialpos
    vocab += [["middle", norm(cconv(vocabVal("inner",vocab), vocabVal("moveout",vocab)))]]
    vocab += [["outer", norm(cconv(vocabVal("middle",vocab), vocabVal("moveout",vocab)))]]
    
    #shape
    vocab += [["uptriangle", norm(cconv(vocabVal("triangle",vocab), vocabVal("90deg",vocab)))]]
    vocab += [["downtriangle", norm(cconv(vocabVal("triangle",vocab), vocabVal("270deg",vocab)))]]
    
    #portion
    vocab += [["whole", norm(vecsum(vocabVal("bottomhalf",vocab),vecsum(vocabVal("tophalf",vocab),vecsum(vocabVal("lefthalf",vocab),vocabVal("righthalf",vocab)))))]]
    
    return [[word[0],word[1]] for word in vocab if word[0] != None]

def genVocab50(d, seed):
    """Vocabulary for 50 base words."""
    
    numwords = 50
    vocab = [[None,None] for i in range(numwords*2)]
    
    PDFTools.setSeed(seed)
#    for i in range(numwords):
#        vocab[i][1] = genVector(d)
    vocab = fillVectors(vocab, d, numwords, RPMutils.VECTOR_SIMILARITY)
    
    PDFTools.setSeed(long(time.time()))
    
    #attributes
    vocab[0][0] = "shape"
    vocab[1][0] = "number" 
    vocab[2][0] = "linestroke"
    vocab[3][0] = "angle"
    vocab[4][0] = "length"
    vocab[5][0] = "rpos"
    vocab[6][0] = "hpos"
    vocab[7][0] = "vpos"
    vocab[8][0] = "shading"
    vocab[9][0] = "existence"
    vocab[10][0] = "linetype"
    vocab[11][0] = "width"


 
     
    #values
    
    #misc
    vocab[12][0] = "present"
    vocab[13][0] = "null"
    vocab[14][0] = "rubbish"
    
    #number
    vocab[15][0] = "zero"
    vocab[16][0] = "one"
    vocab[17][0] = "plusone"
    
    #angle
    vocab[18][0] = "0deg"
    vocab[19][0] = "plus45deg"
    
    #shape
    vocab[20][0] = "circle"
    vocab[21][0] = "square" 
    vocab[22][0] = "diamond"
    vocab[23][0] = "dot"
    vocab[24][0] = "triangle"
    vocab[25][0] = "cross"
    vocab[26][0] = "X"
    vocab[27][0] = "rectangle"
    
    #linestroke
    vocab[28][0] = "normal"
    vocab[29][0] = "dashed"
    vocab[30][0] = "bold"
    
    #linetype
    vocab[31][0] = "straight"
    vocab[32][0] = "curved"
    vocab[33][0] = "wavy"
#    vocab[34][0] = 

    #length
    vocab[35][0] = "short"
    vocab[36][0] = "longer"
    
    #radialpos/horizopos/verticpos
    vocab[37][0] = "rinner"
    vocab[38][0] = "hleft"
    vocab[39][0] = "vbottom"
    vocab[40][0] = "moveout"
    vocab[41][0] = "moveright"
    vocab[42][0] = "moveup"
    
    #shading
    vocab[43][0] = "none"
    vocab[44][0] = "solid"
    vocab[45][0] = "lefthatch"
    vocab[46][0] = "righthatch"
    vocab[47][0] = "crosshatch"
    vocab[48][0] = "vertical"
    vocab[49][0] = "horizontal"
    
    #now the higher level vocabulary
    vocab[50][0] = "two"
    vocab[50][1] = RPMutils.normalize(RPMutils.cconv(vocab[16][1], vocab[17][1]))
    vocab[51][0] = "three"
    vocab[51][1] = RPMutils.normalize(RPMutils.cconv(vocab[50][1], vocab[17][1]))
    vocab[52][0] = "four"
    vocab[52][1] = RPMutils.normalize(RPMutils.cconv(vocab[51][1], vocab[17][1]))
    
    vocab[55][0] = "45deg"
    vocab[55][1] = RPMutils.normalize(RPMutils.cconv(vocab[18][1], vocab[19][1]))
    vocab[56][0] = "90deg"
    vocab[56][1] = RPMutils.normalize(RPMutils.cconv(vocab[55][1], vocab[19][1]))
    vocab[57][0] = "135deg"
    vocab[57][1] = RPMutils.normalize(RPMutils.cconv(vocab[56][1], vocab[19][1]))
    vocab[58][0] = "180deg"
    vocab[58][1] = RPMutils.normalize(RPMutils.cconv(vocab[57][1], vocab[19][1]))
    vocab[59][0] = "225deg"
    vocab[59][1] = RPMutils.normalize(RPMutils.cconv(vocab[58][1], vocab[19][1]))
    vocab[60][0] = "270deg"
    vocab[60][1] = RPMutils.normalize(RPMutils.cconv(vocab[59][1], vocab[19][1]))
    vocab[61][0] = "315deg"
    vocab[61][1] = RPMutils.normalize(RPMutils.cconv(vocab[60][1], vocab[19][1]))
    
    vocab[65][0] = "rmiddle"
    vocab[65][1] = RPMutils.normalize(RPMutils.cconv(vocab[37][1], vocab[40][1]))
    vocab[66][0] = "router"
    vocab[66][1] = RPMutils.normalize(RPMutils.cconv(vocab[65][1], vocab[40][1]))
    
    vocab[67][0] = "hmiddle"
    vocab[67][1] = RPMutils.normalize(RPMutils.cconv(vocab[38][1], vocab[41][1]))
    vocab[68][0] = "hright"
    vocab[68][1] = RPMutils.normalize(RPMutils.cconv(vocab[67][1], vocab[41][1]))
    
    vocab[69][0] = "vmiddle"
    vocab[69][1] = RPMutils.normalize(RPMutils.cconv(vocab[39][1], vocab[42][1]))
    vocab[70][0] = "vtop"
    vocab[70][1] = RPMutils.normalize(RPMutils.cconv(vocab[69][1], vocab[42][1]))
    
    vocab[71][0] = "medium"
    vocab[71][1] = RPMutils.normalize(RPMutils.cconv(vocab[35][1], vocab[36][1]))
    vocab[72][0] = "long"
    vocab[72][1] = RPMutils.normalize(RPMutils.cconv(vocab[71][1], vocab[36][1]))
    
    output = open(RPMutils.vocabFile(d, numwords, seed), "w")
    for i in range(len(vocab)):
        if vocab[i][0] != None:
            output.write(vocab[i][0] + " " + RPMutils.floatlist2str(vocab[i][1]) + "\n")
    output.close()
    
def genVocab20(d, seed):
    """Vocabulary for 20 base words."""
    
    numwords = 20
    vocab = [[None,None] for i in range(numwords*2)]
    
    PDFTools.setSeed(seed)
#    for i in range(numwords):
#        vocab[i][1] = genVector(d)
    vocab = fillVectors(vocab, d, numwords, RPMutils.VECTOR_SIMILARITY)
    
    PDFTools.setSeed(long(time.time()))
    
    #attributes
    vocab[0][0] = "shape"
    vocab[1][0] = "number" 
    vocab[2][0] = "size"
    vocab[3][0] = "orientation"
    vocab[4][0] = "position"
    
    #values
    #vocab[5][0] = "small"
    #vocab[6][0] = "medium"
    #vocab[7][0] = "large"
    vocab[8][0] = "zero"
    vocab[9][0] = "one"
    vocab[10][0] = "plusone"
    vocab[11][0] = "horizontal"
    vocab[12][0] = "vertical"
    vocab[13][0] = "oblique"
#    vocab[14][0] = 
    vocab[15][0] = "circle"
    vocab[16][0] = "square" 
    vocab[17][0] = "diamond"
    vocab[18][0] = "triangle"
    vocab[19][0] = "rubbish"
    
    #now the higher level vocabulary
    vocab[20][0] = "two"
    vocab[20][1] = RPMutils.normalize(RPMutils.cconv(vocab[9][1], vocab[10][1]))
    vocab[21][0] = "three"
    vocab[21][1] = RPMutils.normalize(RPMutils.cconv(vocab[20][1], vocab[10][1]))
    vocab[22][0] = "four"
    vocab[22][1] = RPMutils.normalize(RPMutils.cconv(vocab[21][1], vocab[10][1]))
    
    output = open(RPMutils.vocabFile(d, numwords, seed), "w")
    for i in range(len(vocab)):
        if vocab[i][0] != None:
            output.write(vocab[i][0] + " " + RPMutils.floatlist2str(vocab[i][1]) + "\n")
    output.close()
    
def vocabVal(name, vocab):
    """Returns the vector value for the given word."""
    
    for n,v in vocab:
        if n == name:
            return v
    return None 

def loadVocab(file):
    """Loads vocabulary from file."""
    
    input = open(file)
    vocab = []
    
    for line in input:
        splitline = line.split(" ", 1)
        vocab = vocab + [[splitline[0], RPMutils.str2floatlist(splitline[1])]]
    
    return(vocab)

def fillVectors(vocab, d, numwords, threshold = 1.0):
    """Fills the vocabulary with randomly generated vectors."""
    
    numgen = 0
    
    timeout = 1000
    
    while numgen < numwords:
        vec = RPMutils.genVector(d)
        
        unique = True
        for i in range(numgen):
            if(RPMutils.similarity(vocab[i][1], vec) > threshold):
                unique = False
        
        if unique:
            vocab[numgen][1] = vec
            numgen = numgen + 1
            timeout = 1000
        
        timeout = timeout - 1
        if timeout == 0:
            System.out.println("Timeout in fillVectors, using threshold of " + str(threshold+0.1))
            return(fillVectors(vocab, d, numwords, threshold+0.1))
    
    return(vocab)



def testSimilarity(vocab):
    """Calculates a measure of the vocabulary's similarity."""
    
    #we will count the average proportion of the vocabulary that is more than
    #threshold similar to each vector in the population
    threshold = 0.3
    vecs = [x[1] for x in vocab]
    
    sim = 0.0
    for i in range(len(vocab)):
        count = 0.0
        
        #count the number of vectors that exceed threshold
        for j in range(len(vocab)):
            if i != j:
                if RPMutils.similarity(vecs[i],vecs[j]) > threshold:
                    count += 1.0
                    
        count /= len(vocab)
        sim += count
    
    sim /= len(vocab)
    
    return sim