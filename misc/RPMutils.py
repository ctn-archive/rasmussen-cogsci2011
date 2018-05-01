"""Utility functions and global parameters for the model."""

import time
import math
import os

from ca.nengo.model import SimulationMode
from ca.nengo.model import Units
from ca.nengo.model.impl import FunctionInput
from ca.nengo.model.nef.impl import NEFEnsembleFactoryImpl
from ca.nengo.math import PDFTools
from ca.nengo.math.impl import IndicatorPDF
from ca.nengo.math.impl import GaussianPDF
from ca.nengo.math.impl import ConstantFunction

from java.lang import System


#note: the following are all constants, in that they are set for a given model.
#however, this does not guarantee that their values will be the ones listed below.
#in particular, the batch module sets these constants to (possibly) different 
#values before starting each run. these values should be thought of as defaults.

#the number of dimensions to use in HRR vectors
VECTOR_DIMENSION = 30

#the number of neurons to use per dimension
NEURONS_PER_DIMENSION = 25

#random seed to use when generating vocabulary vectors
VOCABULARY_SEED = 100

#the minimum confidence value we will require in order to decide we have a match in cleanup memory
MIN_CONFIDENCE = 0.7

#the minimum value we will require to pick either the same or different result
#SAMEDIFF_CHOICE = 0.5

#the minimum score we will require in order to decide we have found a correct rule
CORRECTNESS_THRESHOLD_FIG = 0.8
CORRECTNESS_THRESHOLD_SEQ = 0.7
CORRECTNESS_THRESHOLD_SET = 0.9

#whether or not to add probes when building networks
USE_PROBES = True

#the time (in seconds) for which we present each input
STEP_SIZE = 0.2

#the size (number of base words) of vocabulary to use (we have different versions depending on how many base words are allowed)
VOCAB_SIZE = 80

#true if we are running the controller, false if we are just running the individual modules
RUN_WITH_CONTROLLER = True

#the maximum similarity we will allow when generating a set of vectors
VECTOR_SIMILARITY = 1.0

#if running jobs concurrently, use this to ensure they don't use overlapping data files
JOB_ID = 0

#the mode to run model in
SIMULATION_MODE = SimulationMode.DEFAULT

#whether or not to use cleanup memory
USE_CLEANUP = False

#whether or not to update the cleanup memory after a run
DYNAMIC_MEMORY = False

#the number of threads we want to run with
NUM_THREADS = 0

#whether or not to split n-dimensional populations into n 1-dimensional populations
SPLIT_DIMENSIONS = True

#the threshold to use when detecting same features in figure solver
SAME_THRESHOLD = 1.0

#the threshold to use when detecting different features in figure solver
DIFF_THRESHOLD = 0.9

#the minimum difference required to differentiate between matrix answers
SIMILARITY_THRESHOLD = 0.0

#the folder in which to read/write all files throughout the run
FOLDER_NAME = "test"

#scale on the total number of neurons
NEURON_SCALE = 1.0

#whether or not to do same/diff calculations in neurons
NEURO_SAMEDIFF = True

#whether or not to load rules from file
LOAD_RULES = False

#kill the given percentage of neurons after generation
KILL_NEURONS = 0.0

#returns the appropriate correctness threshold for each module
def correctnessThreshold(module):
    if module == "figuresolver":
        return CORRECTNESS_THRESHOLD_FIG
    if module == "sequencesolver":
        return CORRECTNESS_THRESHOLD_SEQ
    if module == "setsolver":
        return CORRECTNESS_THRESHOLD_SET

#returns a string containing the current value of all the parameters
def getParameterSettings():
    keys = getParameterSettings.func_globals.keys()
    values = getParameterSettings.func_globals.values()
    
    parms = [[keys[i],values[i]] for i,key in enumerate(keys) if key.isupper()]
    return ",".join(["=".join([str(x) for x in pair]) for pair in parms]) 

#output from origin (used to update cleanup memory)
def cleanupDataFile():
    return os.path.join(CURR_LOCATION, "data", FOLDER_NAME, "cleanupoutputdata_" + str(JOB_ID) + ".txt")

#file containing word-vector associations
def vocabFile(d, numwords, seed):
    return os.path.join(CURR_LOCATION, "data", FOLDER_NAME, "RPMvocab_" + str(numwords) + "x" + str(d) + "_" + str(seed) + ".txt")

#file containing vectors in cleanup memory
def cleanupFile(d, numwords):
    return os.path.join(CURR_LOCATION, "data", FOLDER_NAME, "cleanup_" + str(numwords) + "x" + str(d) + "_" + str(JOB_ID) + ".txt")

#rule output from neural module
def resultFile(modulename):
    return os.path.join(CURR_LOCATION, "data", FOLDER_NAME, modulename + "_result_" + str(JOB_ID) + ".txt")

#prediction of blank cell from neural module
def hypothesisFile(modulename):
    return os.path.join(CURR_LOCATION, "data", FOLDER_NAME, modulename + "_hypothesis_" + str(JOB_ID) + ".txt")

#file to record the rules used to solve a matrix
def ruleFile():
    return os.path.join(CURR_LOCATION, "data", FOLDER_NAME, "rules_" + str(JOB_ID) + ".txt")

#vocabulary present in the matrix
def matrixVocabFile():
    return os.path.join(CURR_LOCATION, "data", FOLDER_NAME, "matrixvocab_" + str(JOB_ID) + ".txt")

#returns a dxd matrix with the given value along the diagonal
def eye(d, val):
    identity = [[0 for x in range(d)] for x in range(d)] 
    for i in range(d):
        identity[i][i] = val
    return(identity)

def str2floatlist(str):
    return [float(word) for word in str.split()]
    
def floatlist2str(floatlist):
    return " ".join([str(x) for x in floatlist])

#generates a random d-dimensional vector
def genVector(d):
    result = [PDFTools.sampleFloat(GaussianPDF()) for i in range(d)]
    
    result = normalize(result)
    
    return result

#creates a function which outputs a random unit vector
def makeInputVector(name, d, randomSeed=None):
    vec = []
    
    if randomSeed == None:
        randomSeed = long(time.clock()*100000000000000000)
    
    if randomSeed > -1:
        PDFTools.setSeed(randomSeed)
    
    length = 0
    for i in range(d):
        tmp = PDFTools.sampleFloat(GaussianPDF())
        vec = vec + [tmp]
        length = length + tmp**2
    
    length = math.sqrt(length)
    
    f = []
    for i in range(d):
        vec[i] = vec[i] / length
        f = f + [ConstantFunction(1, vec[i])]
    
    if randomSeed > -1:
        PDFTools.setSeed(long(time.clock()*1000000000000000))
    
    print vec
    
    return(FunctionInput(name, f, Units.UNK))

#create function inputs, where each function outputs one of the given vectors
def makeInputVectors(names, vectors):
    return [FunctionInput(names[i], [ConstantFunction(1,x) for x in vec], Units.UNK) for i,vec in enumerate(vectors)]
    
#load vectors from a file and create corresponding output functions
def loadInputVectors(filename):
    file = open(filename)
    vectors = [str2floatlist(line) for line in file]
    file.close()

    return makeInputVectors(["vec_" + str(i) for i in range(len(vectors))], vectors)

#an NEF ensemble factory with more evaluation points than normal
class NEFMorePoints(NEFEnsembleFactoryImpl):
    def getNumEvalPoints(self, d):
        #add shortcut so that it doesn't waste time evaluating a bunch of points when its in direct mode
        if SIMULATION_MODE == SimulationMode.DIRECT:
            return 1
        
        pointsPerDim = [0, 1000, 2000]
        if d < 3:
            return(pointsPerDim[d])
        else:
            return(d*500)

#default ensemble factory used in the model
def defaultEnsembleFactory():
    ef=NEFMorePoints()
    ef.nodeFactory.tauRC = 0.02
    ef.nodeFactory.tauRef = 0.002
    ef.nodeFactory.maxRate=IndicatorPDF(200,500)
    ef.nodeFactory.intercept=IndicatorPDF(-1, 1)
    ef.beQuiet()
    return(ef)

#returns all the probes containing name
def findMatchingProbes(probes, name, subname=None):
    result = []
    for probe in probes:
        if name in probe.getTarget().getName() or ((probe.getEnsembleName() != None) and (probe.getEnsembleName().count(name) > 0)):
            result = result + [probe]
    
    if subname == None:
        return result
    else:
        return findMatchingProbes(result, subname)

#calculate circular convolution of vec1 and vec2
def cconv(vec1, vec2):
    if vec1 == None:
        return(vec2)
    if vec2 == None:
        return(vec1)
    
    d = len(vec1)
    result = [0 for i in range(d)]
    
    for i in range(d):
        for j in range(d):
            result[i] = result[i] + vec1[j] * vec2[(i - j) % d]
    
    return(result)

#calculate vector addition of vec1 and vec2
def vecsum(vec1, vec2):
    if vec1 == None:
        return(vec2)
    if vec2 == None:
        return(vec1)

    return [x+y for x,y in zip(vec1,vec2)]

#calculate length of vec
def length(vec):
    return math.sqrt(sum([x**2 for x in vec]))

#normalize vec
def normalize(vec):
    l = length(vec)
    if l == 0:
        return vec
    return [x/l for x in vec]

#calculate similarity between vec1 and vec2
def similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        System.err.println("vectors not the same length in RPMutils.similarity(), something is wrong")
        System.err.println(str(len(vec1)) + " " + str(len(vec2)))

    return sum([x*y for x,y in zip(vec1,vec2)])

def ainv(vec):
    newvec = []
    for i,val in enumerate(vec):
        newvec += [vec[-i % len(vec)]]
    
    return newvec

#calculate mean value of vec
def mean(vec):
    if len(vec) == 0:
        return 0.0
    return float(sum(vec)) / len(vec)

#calculate the words in vocab that vec1 and vec2 have in common
def calcSame(vec1, vec2, vocab, threshold, weight1, weight2):
    vec1 = [x*weight1 for x in vec1]
    vec2 = [x*weight2 for x in vec2]
    vec = vecsum(vec1,vec2)
    
    ans = [0 for i in range(len(vec))]
    for word in vocab:
        if similarity(vec,word) > threshold:
            ans = vecsum(ans,word)

    return normalize(ans)

#calculate the words in vocab that vec1 and vec2 have distinct
def calcDiff(vec1, vec2, vocab, threshold, weight1, weight2):
    vec1 = [x*weight1 for x in vec1]
    vec2 = [x*weight2 for x in vec2]
    vec = [x-y for x,y in zip(vec1,vec2)]
    
    ans = [0 for i in range(len(vec))]
    for word in vocab:
        if similarity(vec,word) > threshold or similarity(vec,word) < -threshold:
            ans = vecsum(ans,word)

    return normalize(ans)
    