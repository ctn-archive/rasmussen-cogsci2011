"""A network to perform circular convolution."""

import math

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl

from misc import RPMutils
from networks import networkensemble
from networks import eprod

class Cconv(NetworkImpl):
    #calculate real part of FFT matrix
    def calcWreal(self, d):
        W = [[0 for x in range(d)] for x in range(int(d/2)+1)]
        for i in range(int(d/2)+1):
            for j in range(d):
                W[i][j] = math.cos(-2 * math.pi * (i*j) / d) / math.sqrt(d)
        
        return(W)
    
    #calculate imaginary part of FFT matrix
    def calcWimag(self, d):
        W = [[0 for x in range(d)] for x in range(int(d/2)+1)]
        for i in range(int(d/2)+1):
            for j in range(d):
                W[i][j] = math.sin(-2 * math.pi * (i*j) / d) / math.sqrt(d)
                
        return(W)
    
    #calculate real part of IFFT matrix
    def calcInvWreal(self, d):
        W = [[0 for x in range(d)] for x in range(d)]
        for i in range(d):
            for j in range(d):
                W[i][j] = math.cos(-2 * math.pi * (i*j) / d)
        
        return(W)
    
    #calculate imaginary part of IFFT matrix
    def calcInvWimag(self, d):
        W = [[0 for x in range(d)] for x in range(d)]
        for i in range(d):
            for j in range(d):
                W[i][j] = -math.sin(-2 * math.pi * (i*j) / d)
    
        return(W)
    
    def __init__(self, name, N, d):
        NetworkImpl.__init__(self)
        self.name = name
        
        tauPSC = 0.007
        
        Wr = self.calcWreal(d)
        Wi = self.calcWimag(d)
            
        halfd = int(d/2)+1
        halfN = int(math.ceil(float(N) * halfd/d))
        
        ef = RPMutils.defaultEnsembleFactory()
        netef = networkensemble.NetworkEnsemble(ef)
        
        #create input populations
        A = ef.make("A", 1, d)
        A.addDecodedTermination("input", RPMutils.eye(d,1), 0.0001, False)
        A.setMode(SimulationMode.DIRECT) #since this is just a relay ensemble for modularity
        A.fixMode()
        self.addNode(A)
        
        B = ef.make("B", 1, d)
        B.addDecodedTermination("input", RPMutils.eye(d,1), 0.0001, False)
        B.setMode(SimulationMode.DIRECT) #since this is just a relay ensemble for modularity
        B.fixMode()
        self.addNode(B)

        #this is the new method, where we collapse the fft into the eprod
        #populations to calculate the element-wise product of our vectors so far
        eprods = []
        
        #note: we scale the output of the eprods by d/2, which we will undo at the
        #end, to keep the overall length of each dimension around 1
        #(the average value of each dimension of a normalized d dimensional vector is 1/sqrt(d), 
        #so 1/sqrt(d)*1/sqrt(d) = 1/d, so when we add the scale the resulting average dimension 
        #should be around d/2d i.e. 1/2)
        #the 2 is added to give us a bit of a buffer, better to have the dimensions too small
        #than too large and run into saturation problems
        multscale = float(d)/2.0
        eprods = eprods + [eprod.Eprod("eprod0", halfN, halfd, scale=multscale, weights=[Wr, Wr], maxinput=2.0/math.sqrt(d))]
        eprods = eprods + [eprod.Eprod("eprod1", halfN, halfd, scale=multscale, weights=[Wi, Wi], maxinput=2.0/math.sqrt(d))]
        eprods = eprods + [eprod.Eprod("eprod2", halfN, halfd, scale=multscale, weights=[Wi, Wr], maxinput=2.0/math.sqrt(d))]
        eprods = eprods + [eprod.Eprod("eprod3", halfN, halfd, scale=multscale, weights=[Wr, Wi], maxinput=2.0/math.sqrt(d))]
        
        for i in range(4):
            self.addNode(eprods[i])
                
            self.addProjection(A.getOrigin("X"), eprods[i].getTermination("A"))
            self.addProjection(B.getOrigin("X"), eprods[i].getTermination("B"))

    
        #negative identity matrix (for subtraction)
        negidentity = [[0 for x in range(d)] for x in range(d)]
        for i in range(d):
            negidentity[i][i] = -1 
        
        #note: all this halfd/expansion stuff is because the fft of a real value
        #is symmetrical, so we do all our computations on just one half and then
        #add in the symmetrical other half at the end
        
        #matrix for expanding real half-vectors (with negative for subtraction)
        expand = RPMutils.eye(halfd,1)
        negexpand = RPMutils.eye(halfd, -1)
        
        #matrix for expanding imaginary half-vectors
        imagexpand = RPMutils.eye(halfd,1)
       
        midpoint = halfd-1-(d+1)%2
        for i in range(int(math.ceil(d/2.0)-1)):
            expand = expand + [expand[midpoint - i]]
            negexpand = negexpand + [negexpand[midpoint - i]]
            
            imagexpand = imagexpand + [[-x for x in imagexpand[midpoint - i]]]
        
        #multiply real components
        rprod = netef.make("rprod", N, tauPSC, [expand, negexpand], None)
        self.addNode(rprod)    
        self.addProjection(eprods[0].getOrigin("X"), rprod.getTermination("in_0"))
        self.addProjection(eprods[1].getOrigin("X"), rprod.getTermination("in_1"))
        
        #multiply imaginary components
        iprod = netef.make("iprod", N, tauPSC, [imagexpand, imagexpand], None)
        self.addNode(iprod)
        self.addProjection(eprods[2].getOrigin("X"), iprod.getTermination("in_0"))
        self.addProjection(eprods[3].getOrigin("X"), iprod.getTermination("in_1"))
        
        #now calculate IFFT of Z = (rprod) + (iprod)i
        #we only need to calculate the real part, since we know the imaginary component is 0
        Winvr = self.calcInvWreal(d)
        Winvi = self.calcInvWimag(d)
        
        for i in range(d):
            for j in range(d):
                Winvr[i][j] = Winvr[i][j] * (1.0/multscale)
                Winvi[i][j] = Winvi[i][j] * (1.0/multscale)
            
        negWinvi = [[0 for x in range(d)] for x in range(d)]
        for i in range(d):
            for j in range(d):
                negWinvi[i][j] = -Winvi[i][j]
            
        result = netef.make("result", N, tauPSC, [Winvr, negWinvi], None)
        
        self.addNode(result)
        
        self.addProjection(rprod.getOrigin("X"), result.getTermination("in_0"))
        self.addProjection(iprod.getOrigin("X"), result.getTermination("in_1"))
        
        if RPMutils.USE_PROBES:
            self.simulator.addProbe("A", "X", True)
            self.simulator.addProbe("B", "X", True)
            self.simulator.addProbe("eprod0", "X", True)
            self.simulator.addProbe("eprod1", "X", True)
            self.simulator.addProbe("eprod2", "X", True)
            self.simulator.addProbe("eprod3", "X", True)
            self.simulator.addProbe("rprod", "X", True)
            self.simulator.addProbe("iprod", "X", True)
            self.simulator.addProbe("result", "X", True)
        
        self.exposeTermination(A.getTermination("input"), "A")
        self.exposeTermination(B.getTermination("input"), "B")
        self.exposeOrigin(result.getOrigin("X"), "X")