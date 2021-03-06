Sequence rule generation in Raven's Progressive Matrices
========================================================

This is one aspect of a model designed to generate the rules needed to
solve a popular test of intelligence, Raven's Progressive
Matrices. This component generates a particular type of rule, which we
call "sequences". These are patterns defined by an iterative
transformation (e.g., the sequence 9, 10, 11 is defined by the
iterative transformation +1). Given Raven's matrices represented in a
mathematical, vector-based form, this component can then generate the
sequence transformations that define those matrices and use them to
find an answer to the problem.

## Instructions

1. Start the [Nengo 1.4](https://www.nengo.ai/nengo-1.4/) simulation
   environment.
3. Select File->Open, and open the `runme.py` file.
4. When the model appears, right click on it and select "Run
   SequenceSolver"

It will take some time for the model to be built and run. To view the
results, select View->Toggle Data Viewer (if the data viewer doesn't
open automatically). The data viewer will display recordings from
various parts of the model as it ran. The most interesting outputs are
calcT:T, which contains the rule generated by the model, and
testSimilarity:result, which displays the model's confidence in the
eight possible answers given for the matrix (the correct answer is
#8). To view the data, right click on the field in the data viewer and
select "Plot w/ options", entering a value around 0.05.

The matrix used as input to the model is given in the file
`sequencematrix_1.txt`. You can try making your own matrices (the
vocabulary is described in the file `vocabulary.py` in the `misc`
folder of the repository) to see how the model does. Just change the
section of the `runme.py` file that refers to `sequencematrix_1.txt`
to the name of the file you created and repeat steps 3-4 above.
