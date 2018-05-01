"""Creates a network which calculates the transform vector given two input vectors."""

import math

from ca.nengo.model.impl import NetworkImpl

from misc import RPMutils
from networks import networkensemble
from networks import cconv
from networks import average

class Transform(NetworkImpl):
    def __init__(self, name, N, d):
        NetworkImpl.__init__(self)
        self.name = name
        
        tauPSC = 0.007
        
        ef = RPMutils.defaultEnsembleFactory()
        netef = networkensemble.NetworkEnsemble(ef)
        
        #create the approximate inverse matrix
        inv = RPMutils.eye(d, 1)
        for i in range(d/2):
            tmp = inv[i+1]
            inv[i+1] = inv[d-i-1]
            inv[d-i-1] = tmp
        
        #create the two input populations
        Ainv = netef.make("Ainv", N, tauPSC, [inv], None)
        self.addNode(Ainv)
        
        B = netef.make("B", N, tauPSC, [RPMutils.eye(d, 1)], None)
        self.addNode(B)
          
        #create circular convolution network
        corr = cconv.Cconv("corr", N, d)
        self.addNode(corr)
        
        self.addProjection(Ainv.getOrigin("X"), corr.getTermination("A"))
        self.addProjection(B.getOrigin("X"), corr.getTermination("B"))
        
        #average result
        T = average.Average("T", N, d)
        self.addNode(T)
        
        self.addProjection(corr.getOrigin("X"), T.getTermination("input"))
        
        if RPMutils.USE_PROBES:
            self.simulator.addProbe("T", "X", True)
            self.simulator.addProbe("corr", "X", True)
            self.simulator.addProbe("Ainv", "X", True)
            self.simulator.addProbe("B", "X", True)
        
        self.exposeOrigin(T.getOrigin("X"), "T")
        self.exposeTermination(Ainv.getTermination("in_0"), "A")
        self.exposeTermination(B.getTermination("in_0"), "B")
#        self.exposeTermination(T.getTermination("lrate"), "lrate")