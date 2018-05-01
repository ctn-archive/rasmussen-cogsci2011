"""Creates a network that will compare a given vector to 8 possible answers and return their 
similarity (dot product)."""

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl

import math

from misc import RPMutils

class Similarity(NetworkImpl):
    def __init__(self, name, N, d, vocab):
        NetworkImpl.__init__(self)
        self.name = name
        
        scaleFactor = 0.1
        smallN = int(math.ceil(float(N)/d))
        tauPSC = 0.007
        
        ef1 = RPMutils.defaultEnsembleFactory()
        ef1.nodeFactory.tauRef = 0.001
        
        test = ef1.make("hypothesis", 1, d)
        test.setMode(SimulationMode.DIRECT) #since this is just a relay ensemble for modularity
        test.fixMode()
        test.addDecodedTermination("input", RPMutils.eye(d,1), 0.0001, False)
        self.addNode(test)
        self.exposeTermination(test.getTermination("input"), "hypothesis")
        
        
        combine = ef1.make("combine", 800, 8)
#        combine.setMode(SimulationMode.DIRECT) #since this is just a relay ensemble for modularity
#        combine.fixMode()
#        combine.collectSpikes(True)
        self.addNode(combine)
        
        inputVec = [[0] for x in range(8)]
        
        #create a population for each possible answer
        for i in range(8):
            ans = ef1.make("ans_" + str(i), smallN, 1)
            ans.addDecodedTermination("input", [vocab[i]], tauPSC, False)
            self.addNode(ans)
            
            self.addProjection(test.getOrigin("X"), ans.getTermination("input"))
            
            inputVec[i] = [scaleFactor]
            combine.addDecodedTermination("in_" + str(i), inputVec, tauPSC, False)
            inputVec[i] = [0]
            
            self.addProjection(ans.getOrigin("X"), combine.getTermination("in_" + str(i)))
            
        
        self.exposeOrigin(combine.getOrigin("X"), "result")
        
        if RPMutils.USE_PROBES:
            self.simulator.addProbe("combine", "X", True)
        