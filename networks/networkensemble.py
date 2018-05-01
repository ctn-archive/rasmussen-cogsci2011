"""Network that mimics a single population but internally splits n dimensions into n 1-dimensional populations."""

from java.lang import System

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl

import math

from misc import RPMutils

class NetworkEnsemble(NetworkImpl):
    def __init__(self, ef):
        self.ef = ef
    
    def make(self, name, N, tauPSC, matrices, outputfuncs = None, splitoverride=None):
        """Create a network ensemble."""
        
        #network ensembles can be either split by dimension or not
        if splitoverride == None:
            splitoverride = RPMutils.SPLIT_DIMENSIONS
        if splitoverride:
            return self.makeNetwork(name, N, tauPSC, matrices, outputfuncs)
        else:
            return self.makePopulation(name, N, tauPSC, matrices, outputfuncs)
    
    def makePopulation(self, name, N, tauPSC, matrices, outputfuncs):
        """Create a network ensemble that doesn't split by dimension (just used to save memory when running
        in direct mode."""
        
        numin = len(matrices)
        d = len(matrices[0])
        
        pop = self.ef.make(name, N, d)
        
        Main = NetworkImpl()
        Main.name = name
        Main.addNode(pop)
        
        for i in range(numin):
            pop.addDecodedTermination("in_" + str(i), matrices[i], tauPSC, False)
            Main.exposeTermination(pop.getTermination("in_" + str(i)), "in_" + str(i))
                                   
        if outputfuncs != None:
            pop.addDecodedOrigin("output", outputfuncs, "AXON")
            Main.exposeOrigin(pop.getOrigin("output"), "X")
        else:
            Main.exposeOrigin(pop.getOrigin("X"), "X")
            
        return Main
    
    def makeNetwork(self, name, N, tauPSC, matrices, outputfuncs):
        """Create a network ensemble that splits by dimension."""
        
        Main = NetworkImpl()
        Main.name = name
        
        numin = len(matrices) #number of inputs
        din = [0 for i in range(numin)] #dimension of each input
        for i in range(numin):
            din[i] = len(matrices[i][0])
            
        dout = len(matrices[0]) #dimension of output
        
        smallN = int(math.ceil(float(N)/dout)) #neurons per population
        defef = RPMutils.defaultEnsembleFactory()
        
        #create input populations (just relay nodes)
        inputs = []
        for i in range(numin):
            inputs = inputs + [defef.make("in_" + str(i), 1, din[i])]
            inputs[i].addDecodedTermination("input", RPMutils.eye(din[i],1), 0.0001, False)
            Main.exposeTermination(inputs[i].getTermination("input"), "in_" + str(i))
            Main.addNode(inputs[i])
            inputs[i].setMode(SimulationMode.DIRECT)
            inputs[i].fixMode()
        
        #output population (another relay node)
        output = defef.make("output", 1, dout)
        Main.exposeOrigin(output.getOrigin("X"), "X")
        Main.addNode(output)
        output.setMode(SimulationMode.DIRECT)
        output.fixMode()
        
        resultTerm = [[0] for x in range(dout)]
        
        #create dimension populations
        for i in range(dout):
            pop = self.ef.make("mid_" + str(i), smallN, 1)
            Main.addNode(pop)
            
            for j in range(numin):
                pop.addDecodedTermination("in_" + str(j), [matrices[j][i]], tauPSC, False)
                Main.addProjection(inputs[j].getOrigin("X"), pop.getTermination("in_" + str(j)))
            
            resultTerm[i] = [1]
            output.addDecodedTermination("in_" + str(i), resultTerm, 0.0001, False)
            resultTerm[i] = [0]
            
            if outputfuncs == None:
                Main.addProjection(pop.getOrigin("X"), output.getTermination("in_" + str(i)))
            else:
                pop.addDecodedOrigin("output", [outputfuncs[i]], "AXON")
                Main.addProjection(pop.getOrigin("output"), output.getTermination("in_" + str(i)))
            
        return Main