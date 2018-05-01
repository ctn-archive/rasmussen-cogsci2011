"""Contains the information for one matrix, and can output it in the form the model needs."""

import math
from java.lang import System

from misc import RPMutils
from misc import vocabulary

class MatrixHandler:

    def __init__(self, matrixfilename, vocab):
        self.vocab = vocab
        
        self.matrix = [None for i in range(16)]
        self.numfeatures = [None for i in range(16)]
        
        #read in the matrix information
        matrixfile = open(matrixfilename)
        for i in range(16):
            self.numfeatures[i] = int(self.readMatrixLine(matrixfile))
            self.matrix[i] = [None for j in range(self.numfeatures[i])]
            for j in range(self.numfeatures[i]):
                pairs = self.readMatrixLine(matrixfile).split(";")
                self.matrix[i][j] = [pair.split() for pair in pairs]
        matrixfile.close()
            
    def readMatrixLine(self, matrixfile):
        """Reads in the next line from the matrix file (skipping any blank lines or lines that start with #)."""
        line = matrixfile.readline().strip()             
        
        while line == "" or line.startswith("#"):
            line = matrixfile.readline().strip()
        
        return line
    
    def getAttributes(self, feature):
        """Returns the attributes present in the given feature."""
        
        #note: this function assumes that the attributes are constant across the cells
        #(i.e. the attributes in feature 2 of cell 5 are the same as the attributes in feature 2 of cell 1)
        
        return [pair[0] for pair in self.matrix[0][feature]]
    
    def getVocabVal(self, word):
        """Returns the vector associated with the given word."""
        
        for line in self.vocab:
            if line[0] != None and line[0] == word:
                return line[1]
        System.err.println("Error, " + word + " not found in vocab")
        return None
    
    def getMatrix(self, feature=None, attribute=None):
        """Returns the (sub)matrix data (restricted to the given feature/attribute)."""
        
        if feature == None:
            return self.matrix[0:8]
        
        if attribute == None:
            return [[self.matrix[i][feature]] for i in range(8)]
        else:
            return [[[[attr,val] for attr,val in self.matrix[i][feature] if attr in attribute]] for i in range(8)]
       
    def getAnswers(self, feature=None, attribute=None):
        """Returns the (sub)answer data (restricted to the given feature/attribute)."""
        
        if feature == None:
            return self.matrix[8:]
        
        if attribute == None:
            return [[self.matrix[8+i][feature]] for i in range(8)]
        else: 
            return [[[[attr,val] for attr,val in self.matrix[8+i][feature] if attr in attribute]] for i in range(8)]
        
    
    def encodeMatrix(self, matrix):
        """Converts the matrix into vector form."""
        
        #check if there is more than one feature in the matrix
        #this is relevant because if there is more than one feature then
        #we will encode each feature as A + B + AxB, whereas if there is
        #only one feature we will just use A + B
        oneFeature = True
        for cell in matrix:
            if len(cell) > 1:
                oneFeature = False
        
        result = []
        #for each cell in the matrix
        for cell in matrix:
            vec = None #the vector representation for this cell
            
            #for each feature in the cell
            for feature in cell:
                featurevec = None #the vector representation for this feature (the A + B part)
                prodvec = None #the AxB part of the vector representation of this feature
                
                #for each attr in the feature
                for attr,val in feature:
                    attribute = self.getVocabVal(attr)
                    value = self.getVocabVal(val)
                    
                    pairword = RPMutils.normalize(RPMutils.cconv(attribute,value)) #vector for that attribute-value pair
                    
                    featurevec = RPMutils.vecsum(featurevec, pairword)
                    prodvec = RPMutils.cconv(prodvec,pairword)
                
                if oneFeature:
                    if vec != None:
                        System.err.println("oneFeature is true but more than one feature vector is being calculated!")
                    vec = featurevec #ignore the AxB part and vec=featurevec because there is only one feature
                else:
                    #add this feature (including AxB part) to previous features in the cell 
                    vec = RPMutils.vecsum(vec, RPMutils.normalize(RPMutils.vecsum(RPMutils.normalize(prodvec), RPMutils.normalize(featurevec))))
            if vec==None:
                vec = self.getVocabVal("null")
            result = result + [RPMutils.normalize(vec)]
        
        return result
    
    def getMatrixVocab(self): 
        """Returns all the vocabulary (in vector form) used in this matrix."""
        
        #first figure out all the words that are used
        #words in this case are attr x val pairs
        #also add the AxB tags we add to feature descriptions when more than one feature is being used
        v = []  #list of vocab words
        for cell in self.matrix[0:8]:
            for feature in cell:
                fv = "" #the AxB tag
                for attr,val in feature:   
                    v = v + [attr + " " + val] #add the word for this attr-val pair to the list
                    fv = fv + attr + " " + val + ";" #add this attr-val pair to the AxB tag
                
                if len(feature) > 1: 
                    #if more than one attribute in feature, add the tag to the word list (if there is
                    #only one attribute in feature then the tag is the same as the attr-val word we
                    #already added, so unnecessary)
                    v = v + [fv]
        
        #remove duplicates
        v.sort()
        v = [word for i,word in enumerate(v) if i == len(v)-1 or not v[i+1] == word]
        
#        print v
        
        #turn vocab into vectors
        vocab = []
        for word in v:
            vec = None
            prod = None
            for pair in word.split(";"):
                if pair:
                    attr,val = pair.split(" ")
                    
                    #the vector for this attr-val pair
                    pairword = RPMutils.normalize(RPMutils.cconv(self.getVocabVal(attr), self.getVocabVal(val)))
                     
                    vec = RPMutils.vecsum(vec, pairword)  #the non-tag part (vec should always be None, so vec=pairword)
                    prod = RPMutils.cconv(prod, pairword) #the tag part
            if ";" in word:
                vec = prod #then use the tag part
            vocab = vocab + [RPMutils.normalize(vec)]
        
        return vocab
    
    def printMatrix(self, matrix):
        for cell in matrix:
            print cell