import os
import sys
sys.path.append(os.path.dirname(scriptname))

from misc import RPMutils
from misc import vocabulary
from misc import matrixhandler
from networks import sequencesolver

d = RPMutils.VECTOR_DIMENSION
N = d*RPMutils.NEURONS_PER_DIMENSION
numwords = RPMutils.VOCAB_SIZE
seed = 107
RPMutils.CURR_LOCATION = os.path.dirname(scriptname)
RPMutils.RUN_WITH_CONTROLLER = False


vocab = vocabulary.genVocab(d, numwords, seed)

mhandler = matrixhandler.MatrixHandler(RPMutils.CURR_LOCATION + "\sequencematrix_1.txt", vocab)

test = sequencesolver.SequenceSolver(N, d, mhandler.encodeMatrix(mhandler.getMatrix()) + mhandler.encodeMatrix(mhandler.getAnswers()))

world.add(test)