"""Network to calculate the (approximate) average of inputs over time."""

from ca.nengo.model.impl import NetworkImpl

from misc import RPMutils
from networks import networkensemble
from networks import integrator
from networks import eprod

class Average(NetworkImpl):
    def __init__(self, name, N, d):
        NetworkImpl.__init__(self)
        self.name = name
        
        tauPSC = 0.007
        
        ef = RPMutils.defaultEnsembleFactory()
        netef = networkensemble.NetworkEnsemble(ef)
    
        #scale input to integrator (adaptive learning rate)
#        scaler = eprod.Eprod("scaler", N, d, oneDinput=True)
#        self.addNode(scaler)

        #new idea, try a constant scale
        #constant scale on input: 0.4
        #constant scale on recurrent connection: 0.8 (forget rate 0.2)
        
        #create integrator
        #we scale the input by 1/stepsize because we want the integrator to reach the target value in stepsize*1s, not 1s
        int = integrator.Integrator("int", N, d, inputScale=0.4, forgetRate=0.2, stepsize=RPMutils.STEP_SIZE)
        self.addNode(int)
        
#        self.addProjection(scaler.getOrigin("X"), int.getTermination("input"))
        
        if RPMutils.USE_PROBES:
            self.simulator.addProbe("int", "X", True)
#            self.simulator.addProbe("scaler", "X", True)
        
        self.exposeOrigin(int.getOrigin("X"), "X")
#        self.exposeTermination(scaler.getTermination("A"), "input")
#        self.exposeTermination(scaler.getTermination("B"), "lrate")
        self.exposeTermination(int.getTermination("input"), "input")
    