"""Creates a multidimensional integrator network."""

from ca.nengo.model import *
from ca.nengo.model.impl import *
from ca.nengo.model.nef.impl import *
from ca.nengo.math.impl import *

import math

from misc import RPMutils
from networks import networkensemble

class Integrator(NetworkImpl):
    def __init__(self, name, N, d, inputScale=1.0, forgetRate = 0.0, stepsize=1.0):
        NetworkImpl.__init__(self)
        self.name = name
        
        tauPSC = 0.007
        intPSC = 0.1
        smallN = int(math.ceil((float(N)/d) * 2))
        inputWeight = intPSC * 1.0/stepsize * inputScale  
            #weight on input connection
            #we multiply by intPSC as in standard NEF integrator formulation
            #we multiply by 1/stepsize so that the integrator will reach its target value
            #in stepsize rather than the default 1 second
            #then we multiply by the scale on the input
        recurWeight = 1-(intPSC * 1.0/stepsize * forgetRate) #weight on recurrent connection
        
        ef=RPMutils.defaultEnsembleFactory()
        netef = networkensemble.NetworkEnsemble(ef)
        
        intef = RPMutils.NEFMorePoints()
        intef.nodeFactory.tauRC = 0.05
        intef.nodeFactory.tauRef = 0.002
        intef.nodeFactory.maxRate = IndicatorPDF(100, 200)
        intef.nodeFactory.intercept = IndicatorPDF(-1, 1)
        intef.beQuiet()
        
        input = netef.make("input", N, 0.05, [RPMutils.eye(d, 1)], None) #note we run this in non-direct mode to eliminate the "bumps"
        self.addNode(input)   
        
        output = ef.make("output", 1, d)
        output.setMode(SimulationMode.DIRECT) #since this is just a relay ensemble for modularity
        output.fixMode()
        self.addNode(output)
        
        if RPMutils.SPLIT_DIMENSIONS:
            invec = [0 for x in range(d)]
            resultTerm = [[0] for x in range(d)]
            
            for i in range(d):
                intpop = intef.make("intpop_" + str(i), smallN, 1)
                
                invec[i] = inputWeight
                intpop.addDecodedTermination("input", [invec], tauPSC, False)
                invec[i] = 0
                intpop.addDecodedTermination("feedback", [[recurWeight]], intPSC, False)
                self.addNode(intpop)
                
                self.addProjection(input.getOrigin("X"), intpop.getTermination("input"))
                self.addProjection(intpop.getOrigin("X"), intpop.getTermination("feedback"))
                
                resultTerm[i] = [1]
                output.addDecodedTermination("in_" + str(i), resultTerm, 0.0001, False)
                resultTerm[i] = [0]
                
                self.addProjection(intpop.getOrigin("X"), output.getTermination("in_" + str(i)))
        else:
            #do all the integration in one population rather than dividing it up by dimension. note that
            #this will only really work in direct mode.
            
            intpop = intef.make("intpop", N, d)
            intpop.addDecodedTermination("input", RPMutils.eye(d,inputWeight), tauPSC, False)
            intpop.addDecodedTermination("feedback", RPMutils.eye(d,recurWeight), intPSC, False)
            self.addNode(intpop)
            
            self.addProjection(input.getOrigin("X"), intpop.getTermination("input"))
            self.addProjection(intpop.getOrigin("X"), intpop.getTermination("feedback"))
            
            output.addDecodedTermination("in", RPMutils.eye(d,1), 0.0001, False)
            self.addProjection(intpop.getOrigin("X"), output.getTermination("in"))
            
        
        self.exposeTermination(input.getTermination("in_0"), "input")
        self.exposeOrigin(output.getOrigin("X"), "X")           
        