"""Network to calculate sequence rules."""


import math
from java.io import File
from java.lang import System

from ca.nengo.model import SimulationMode
from ca.nengo.model import Units
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.model.impl import FunctionInput
from ca.nengo.math.impl import PiecewiseConstantFunction
from ca.nengo.math.impl import ConstantFunction

from misc import RPMutils
from misc import vectorgenerators
from networks import transform
from networks import similarity
from networks import cconv

class SequenceSolver(NetworkImpl):
    def __init__(self, N, d, matrix):
        NetworkImpl.__init__(self)
        self.name = "SequenceSolver"
        self.N = N
        self.d = d
        
        ef1=RPMutils.defaultEnsembleFactory()
        
        #load matrix data from file
        matrixData = self.loadSequenceMatrix(matrix)
        
        #the two input signals, A and B, representing the sequence of example pairs
        Ain = matrixData[0]
        Bin = matrixData[1]
        self.addNode(Ain)
        self.addNode(Bin)
        
        #the adaptive learning rate
#        lrate = matrixData[2]
#        self.addNode(lrate)
        
        #calculate the T for the current A and B
        calcT = transform.Transform("calcT", N, d)
        self.addNode(calcT)
        
        self.addProjection(Ain.getOrigin("origin"), calcT.getTermination("A"))
        self.addProjection(Bin.getOrigin("origin"), calcT.getTermination("B"))
#        self.addProjection(lrate.getOrigin("origin"), calcT.getTermination("lrate"))
        
        if RPMutils.USE_CLEANUP:
            #run T through cleanup memory
            cleanT = memory.Memory("cleanT", N, d)
            self.addNode(cleanT)
            
            self.addProjection(calcT.getOrigin("T"), cleanT.getTermination("dirty"))
        
        #calculate the result of applying T to the second last cell
        secondLast = matrixData[3]
        self.addNode(secondLast)
        
        calcLast = cconv.Cconv("calcLast", N, d)
        self.addNode(calcLast)
        
        self.addProjection(secondLast.getOrigin("origin"), calcLast.getTermination("A"))
        
        if RPMutils.USE_CLEANUP:
            self.addProjection(cleanT.getOrigin("clean"), calcLast.getTermination("B"))
        else:
            self.addProjection(calcT.getOrigin("T"), calcLast.getTermination("B"))
            
        if RPMutils.LOAD_RULES:
            self.removeProjection(calcLast.getTermination("B"))
            rulesig = matrixData[len(matrixData)-1]
            self.addNode(rulesig)
            self.addProjection(rulesig.getOrigin("origin"),calcLast.getTermination("B"))
        
        
        #compare the result to the possible answers to determine which is most similar
        if not RPMutils.RUN_WITH_CONTROLLER:
            testSimilarity = similarity.Similarity("testSimilarity", N, d, matrixData[4:])
            self.addNode(testSimilarity)
            
            self.addProjection(calcLast.getOrigin("X"), testSimilarity.getTermination("hypothesis"))
            self.simulator.addProbe("testSimilarity", "result", True)
        
        if RPMutils.USE_CLEANUP:
            Tprobe = self.simulator.addProbe("cleanT", "clean", True)
        else:
            Tprobe = self.simulator.addProbe("calcT", "T", True)
        answerprobe = self.simulator.addProbe("calcLast", "X", True)
        
        
        if RPMutils.USE_CLEANUP and RPMutils.DYNAMIC_MEMORY:
            self.simulator.addSimulatorListener(memorylistener.MemoryManagementListener(RPMutils.cleanupDataFile(), RPMutils.cleanupFile(d, RPMutils.VOCAB_SIZE)))
        
        if RPMutils.RUN_WITH_CONTROLLER:
            self.simulator.addSimulatorListener(proberecorder.ProbeRecorder(Tprobe, RPMutils.resultFile("sequencesolver"), 0.05))
            self.simulator.addSimulatorListener(proberecorder.ProbeRecorder(answerprobe, RPMutils.hypothesisFile("sequencesolver"), 0.05))
        
        self.setMode(RPMutils.SIMULATION_MODE)

    
    def loadSequenceMatrix(self, cell):
        """Load a matrix in HRR vector format from a file and create corresponding output functions."""
        
        d = len(cell[0])
        discontinuities = [RPMutils.STEP_SIZE, 2*RPMutils.STEP_SIZE, 3*RPMutils.STEP_SIZE, 4*RPMutils.STEP_SIZE, 5*RPMutils.STEP_SIZE]
        values1 = [[0 for x in range(6)] for x in range(d)]
        values2 = [[0 for x in range(6)] for x in range(d)]
        
        
        #         1.0       2.0      3.0      4.0
        #signal A
        #    cell1    cell2    cell4    cell5    cell7
        #signal B
        #    cell2    cell3    cell5    cell6    cell8
        
        #values for 0th timestep
        for i in range(d):
            values1[i][0] = cell[0][i]
        for i in range(d):
            values2[i][0] = cell[1][i]
            
        #values for 1st timestep
        for i in range(d):
            values1[i][1] = cell[1][i]
        for i in range(d):
            values2[i][1] = cell[2][i]
            
        #values for 2nd timestep
        for i in range(d):
            values1[i][2] = cell[3][i]
        for i in range(d):
            values2[i][2] = cell[4][i]
            
        #values for 3rd timestep
        for i in range(d):
            values1[i][3] = cell[4][i]
        for i in range(d):
            values2[i][3] = cell[5][i]
        
        #values for 4th timestep
        for i in range(d):
            values1[i][4] = cell[6][i]
        for i in range(d):
            values2[i][4] = cell[7][i]
            
        for i in range(d):
            values1[i][5] = 0
            values2[i][5] = 0
            
        #create signalA
        f = []
        for i in range(d):
            f = f + [PiecewiseConstantFunction(discontinuities, values1[i])]
        sigA = FunctionInput("sigA", f, Units.UNK)
        
        #create signal B
        f = []
        for i in range(d):
            f = f + [PiecewiseConstantFunction(discontinuities, values2[i])]
        sigB = FunctionInput("sigB", f, Units.UNK)
        
        #create signal for adaptive learning rate
        rates = [1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0, 1.0/5.0, 0.0]
        lrate = FunctionInput("lrate", [PiecewiseConstantFunction(discontinuities, rates)], Units.UNK)
        
        #create signal for second last cell
        f = []
        for i in range(d):
            f = f + [ConstantFunction(1, cell[7][i])]
        secondLast = FunctionInput("secondLast", f, Units.UNK)
        
        #load rule signal from file
        rulesig = []
        if RPMutils.LOAD_RULES:
            rulefile = open(RPMutils.ruleFile())
            lines = rulefile.readlines()
            rulefile.close()
            mod,rule = lines[0].split(":")
            if mod == "sequencesolver":
                rule = RPMutils.str2floatlist(rule.strip())
            else:
                rule = [0.0 for i in range(self.d)]
            rulesig = RPMutils.makeInputVectors("rulesig", [rule])
        
        if RPMutils.RUN_WITH_CONTROLLER:
            return([sigA, sigB, lrate, secondLast] + rulesig)
        else:
            #create signals for answers
            ans = []
            for i in range(8):
                ans = ans + [cell[8+i]]
                
            return([sigA, sigB, lrate, secondLast] + ans + rulesig)
    
    
    def reload(self, matrix):
        """Reload network with new matrix information."""
        
        N = self.N
        d = self.d
        
        if RPMutils.LOAD_RULES:
            System.out.println("Warning, calling reload when LOAD_RULES is True")
        
        #reload matrix data
        matrixData = self.loadSequenceMatrix(matrix)
        
        #remove and re-add input signals
        self.removeProjection(self.getNode("calcT").getTermination("A"))
        self.removeProjection(self.getNode("calcT").getTermination("B"))
#        self.removeProjection(self.getNode("calcT").getTermination("lrate"))
        self.removeNode("sigA")
        self.removeNode("sigB")
#        self.removeNode("lrate")
        
        Ain = matrixData[0]
        Bin = matrixData[1]
#        lrate = matrixData[2]
        self.addNode(Ain)
        self.addNode(Bin)
#        self.addNode(lrate)
        self.addProjection(Ain.getOrigin("origin"), self.getNode("calcT").getTermination("A"))
        self.addProjection(Bin.getOrigin("origin"), self.getNode("calcT").getTermination("B"))
#        self.addProjection(lrate.getOrigin("origin"), self.getNode("calcT").getTermination("lrate"))
        
        #remove and re-add secondLast signal
        self.removeProjection(self.getNode("calcLast").getTermination("A"))
        self.removeNode("secondLast")
        
        secondLast = matrixData[3]
        self.addNode(secondLast)
        self.addProjection(secondLast.getOrigin("origin"), self.getNode("calcLast").getTermination("A"))
        
        
        #remove and re-add similarity network
        if not RPMutils.RUN_WITH_CONTROLLER:
            self.removeProjection(self.getNode("testSimilarity").getTermination("hypothesis"))
            probes = RPMutils.findMatchingProbes(self.simulator.getProbes(), "testSimilarity")
            for probe in probes:
                self.simulator.removeProbe(probe)
            self.removeNode("testSimilarity")
            
            testSimilarity = similarity.Similarity("testSimilarity", N, d, matrixData[4:])
            self.addNode(testSimilarity)
            self.addProjection(self.getNode("calcLast").getOrigin("X"), testSimilarity.getTermination("hypothesis"))
            self.simulator.addProbe("testSimilarity", "result", True)
        
        if RPMutils.USE_CLEANUP:
            #call reload on memory network, which will reload cleanup memory
            probes = RPMutils.findMatchingProbes(self.simulator.getProbes(), "cleaner")
            for probe in probes:
                self.simulator.removeProbe(probe)
            
            self.getNode("cleanT").reload()
        
        #reset all probes
        self.simulator.resetProbes()
        
        self.setMode(RPMutils.SIMULATION_MODE)