"""Functions to generate different sets of vectors."""

import math

from java.io import Serializable

from ca.nengo.util import VectorGenerator
from ca.nengo.math import PDFTools
from ca.nengo.math.impl import GaussianPDF
from ca.nengo.math.impl import IndicatorPDF

class CleanupVectorGenerator(VectorGenerator, Serializable):
    """Returns the vectors contained in a vocabulary, one at a time."""
    
    serialVersionUID = 1
    
    index = 0
    vocabulary = []
    variance = 0.01
    
    def genVectors(self, N, d):      
        if d != len(self.vocabulary[0]):
            print "Error, vocabulary d != requested d in CleanupVectorGenerator"
        
        if self.index >= len(self.vocabulary):
            return([None])
        
        max = 0
        for x in self.vocabulary[self.index]:
            if abs(x) > max:
                max = abs(x)
                
        
        vecs = []
        for i in range(N):
            vec = [0 for x in range(d)]
            length = 0
            for j in range(d):
                vec[j] = self.vocabulary[self.index][j] + PDFTools.sampleFloat(GaussianPDF(0.0, self.variance)) * max
                length = length + vec[j]**2
            length = math.sqrt(length)
            for j in range(d):
                vec[j] = vec[j] / length
            vecs = vecs + [vec]
        
        return(vecs)
    
    def setVocabulary(self, vocab):
        self.vocabulary = vocab
    
    def reset(self):
        self.index = 0
    
    def nextWord(self):
        self.index = self.index + 1    
        
class DirectedVectorGenerator(VectorGenerator, Serializable):  
    """Returns vectors pointed in a given direction."""
      
    serialVersionUID = 1
    
    def __init__(self, direction=None):
        self.dir = direction
    
    def setDirection(self, dir):
        self.dir = dir
    
    def getDirection(self):
        return(self.dir)
    
    def genVectors(self, N, d):
        if self.dir == None:
            print "Error, calling genVectors before setting direction"
        
        if len(self.dir) != d:
            print "Error, direction dimension not equal to requested dimension"

#        result = [self.dir for i in range(N)]
#        print result

        return([self.dir for i in range(N)])

class DirectedEvalPointGenerator(VectorGenerator, Serializable):
    """Returns vectors pointed in a given direction with length from 0 to 1."""
    serialVersionUID = 1
    
    def __init__(self, direction):
        self.dir = direction
    
    def genVectors(self, N, d):
        if d != len(self.dir):
            print "Error, direction dimension not equal to requested dimension in DirectedEvalPointGenerator"
            
        vectors = []
        for i in range(N):
            scale = PDFTools.sampleFloat(IndicatorPDF(0.0, 1.0))
            vec = [0.0 for i in range(d)]
            
            for j in range(d):
                vec[j] = float(self.dir[j]) * scale
            vectors = vectors + [vec]
            
        
        return(vectors)
    
class RangedEvalPointGenerator(VectorGenerator, Serializable):
    """Generates scalar values within a given range."""
    
    serialVersionUID = 1
    
    def __init__(self, ranges):
        self.ranges = ranges
        self.numranges = len(ranges)
        
    def genVectors(self, N, d):
        if d != 1:
            print "Error, trying to use RangedEvalPointGenerator when d > 1"
        
#        if not self.low2:
#            return [[PDFTools.sampleFloat(IndicatorPDF(self.low1, self.high1))] for i in range(N)]
#        
#        vecs = []
#        for i in range(N):
#            val = PDFTools.sampleFloat(IndicatorPDF(self.low1, self.high2))
#            while not ((val >= self.low1 and val <= self.high1) or (val >= self.low2 and val <= self.high2)):
#                val = PDFTools.sampleFloat(IndicatorPDF(self.low1, self.high2))
#            vecs = vecs + [[val]]
        Nperrange = int(float(N)/self.numranges)
        vecs = []
        for i in range(self.numranges):
            low = self.ranges[i][0]
            high = self.ranges[i][1]
            for j in range(Nperrange):
                val = PDFTools.sampleFloat(IndicatorPDF(low, high))
                vecs = vecs + [[val]]
        
        for j in range(N-(Nperrange*self.numranges)):
            val = PDFTools.sampleFloat(IndicatorPDF(low, high))
            vecs = vecs + [[val]]
        
        return vecs

class MultiplicationVectorGenerator(VectorGenerator, Serializable): 
    """Generates vectors at 45 degrees (good for multiplication encoding vectors)."""
    
    serialVersionUID = 1
    
    def genVectors(self, N, d):
        if d != 2:
            print "Error, d !=2 when generating custom EUVs"
            
        angle = math.pi/4
        vectors = []
        
        for i in range(N):
            vectors = vectors + [[math.cos(angle), math.sin(angle)]]
            angle = (angle + math.pi/2) % (2 * math.pi)
        
        return(vectors)
    
class NeuronVectorGenerator(VectorGenerator, Serializable):
    """Generates pre-specified encoding vectors for neurons."""
    
    serialVersionUID = 1
    
    def __init__(self, vectors):
        self.vectors = vectors
        
    def genVectors(self, N, d):
        vecs = []
        for i in range(N):
            if i < len(self.vectors):
                vecs.append(self.vectors[i])
            else:
                tmp = [PDFTools.sampleFloat(GaussianPDF()) for x in range(d)]
                length = math.sqrt(sum([x**2 for x in tmp]))
                tmp = [x / length for x in tmp]
                vecs.append(tmp)
        
        return vecs