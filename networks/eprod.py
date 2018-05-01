"""A network to calculate the element-wise product between two inputs."""

from ca.nengo.model import *
from ca.nengo.model.impl import *
from ca.nengo.model.nef.impl import *
from ca.nengo.math.impl import *
from ca.nengo.util import *

import math
from java.lang import System

from misc import RPMutils
from misc.vectorgenerators import *

class Eprod(NetworkImpl):
    def __init__(self, name, N, d, scale=1.0, weights = None, maxinput=1.0, oneDinput=False):
        #scale is a scale on the output of the multiplication
        #output = (input1.*input2)*scale
        
        #weights are optional matrices applied to each input
        #output = (C1*input1 .* C2*input2)*scale
        
        #maxinput is the maximum expected value of any dimension of the inputs. this is used to
        #scale the inputs internally so that the length of the vectors in the intermediate populations are not
        #too small (which results in a lot of noise in the calculations)
        
        #oneDinput indicates that the second input is one dimensional, and is just a scale on the
        #first input rather than an element-wise product
        
        NetworkImpl.__init__(self)
        self.name = name
        
        smallN = int(math.ceil(float(N)/d)) #the size of the intermediate populations
        tauPSC = 0.007
        
        #the maximum value of the vectors represented by the intermediate populations.
        #the vector is at most [maxinput maxinput], so the length of that is
        #sqrt(maxinput**2 + maxinput**2)
        maxlength = math.sqrt(2*maxinput**2)
        
        if weights != None and len(weights) != 2:
            System.out.println("Warning, other than 2 matrices given to eprod")
        
        if weights == None:
            weights = [RPMutils.eye(d,1),RPMutils.eye(d,1)]
            
        inputd = len(weights[0][0])
            
    
        ef=RPMutils.defaultEnsembleFactory()
        
        #create input populations
        in1 = ef.make("in1", 1, inputd)
        in1.addDecodedTermination("input", RPMutils.eye(inputd, 1), 0.0001, False)
        self.addNode(in1)
        in1.setMode(SimulationMode.DIRECT) #since this is just a relay ensemble for modularity
        in1.fixMode()
        
        
        in2 = ef.make("in2", 1, inputd)
        if not oneDinput:
            in2.addDecodedTermination("input", RPMutils.eye(inputd, 1), 0.0001, False)
        else:
            #if it is a 1-D input we just expand it to a full vector of that value so that we
            #can treat it as an element-wise product
            in2.addDecodedTermination("input", [[1] for i in range(inputd)], 0.0001, False)
        self.addNode(in2)
        in2.setMode(SimulationMode.DIRECT) #since this is just a relay ensemble for modularity
        in2.fixMode()
            
        
        #ensemble for intermediate populations
        multef=NEFEnsembleFactoryImpl()
        multef.nodeFactory.tauRC = 0.05
        multef.nodeFactory.tauRef = 0.002
        multef.nodeFactory.maxRate=IndicatorPDF(200,500)
        multef.nodeFactory.intercept=IndicatorPDF(-1, 1)
        multef.encoderFactory = MultiplicationVectorGenerator()
        multef.beQuiet()
    
    
        result = ef.make("result", 1, d)
        result.setMode(SimulationMode.DIRECT) #since this is just a relay ensemble for modularity
        result.fixMode()
        self.addNode(result)
        
        if RPMutils.SPLIT_DIMENSIONS:
            resultTerm = [[0] for x in range(d)]
            zeros = [0 for x in range(inputd)]
            one = [0 for x in range(inputd)]
            
            mpop = []
            for e in range(d):
                #create a 2D population for each input dimension which will combine the components from
                #one dimension of each of the input populations
                mpop = multef.make('mpop_' + str(e), smallN, 2)
                
                #make two connection that will select one component from each of the input pops 
                #we divide by maxlength to ensure that the maximum length of the 2D vector is 1  
                #remember that (for some reason) the convention in Nengo is that the input matrices are transpose of what they should be mathematically  
                for i in range(inputd):
                    one[i] = (1.0 / maxlength) * weights[0][e][i]
                mpop.addDecodedTermination('a', [one, zeros], tauPSC, False)
                
                for i in range(inputd):
                    one[i] = (1.0 / maxlength) * weights[1][e][i]
                mpop.addDecodedTermination('b', [zeros, one], tauPSC, False) 
                one = [0 for x in range(inputd)]
                
                #multiply the two selected components together
                mpop.addDecodedOrigin("output", [PostfixFunction('x0*x1', 2)], "AXON")
                self.addNode(mpop)
                self.addProjection(in1.getOrigin('X'), mpop.getTermination('a'))
                self.addProjection(in2.getOrigin('X'), mpop.getTermination('b'))
                
                #combine the 1D results back into one vector
                resultTerm[e] = [maxlength**2 * scale]  #undo our maxlength manipulations and apply the scale
                        #we scaled each input by 1/maxlength, then multiplied them together for a total scale of
                        #1/maxlength**2, so to undo we multiply by maxlength**2
                result.addDecodedTermination('in_' + str(e), resultTerm, 0.0001, False)
                resultTerm[e] = [0]
                
                self.addProjection(mpop.getOrigin('output'), result.getTermination('in_' + str(e)))
        else:
            #do all the multiplication in one population rather than splitting it up by dimension. note that this will
            #only really work in direct mode.
            
            mpop = ef.make("mpop", N, 2*d)
            mpop.addDecodedTermination("a", weights[0] + [[0 for x in range(inputd)] for y in range(d)], tauPSC, False)
            mpop.addDecodedTermination("b", [[0 for x in range(inputd)] for y in range(d)] + weights[1], tauPSC, False)
            mpop.addDecodedOrigin("output", [PostfixFunction("x" + str(i) + "*x" + str(d+i), 2*d) for i in range(d)], "AXON")
            self.addNode(mpop)
            
            self.addProjection(in1.getOrigin("X"), mpop.getTermination("a"))
            self.addProjection(in2.getOrigin("X"), mpop.getTermination("b"))
            
            result.addDecodedTermination("input", RPMutils.eye(d,scale), tauPSC, False)
            self.addProjection(mpop.getOrigin("output"), result.getTermination("input"))
        
        
        self.exposeTermination(in1.getTermination("input"), "A")
        self.exposeTermination(in2.getTermination("input"), "B")
        self.exposeOrigin(result.getOrigin("X"), "X")
        