import theano
a = theano.tensor.vector('a') # declare variable
b = a + a**10                 # build expression
f = theano.function([a], b)   # compile function
print `f([0,1,2])`            # call function
# prints 'array([0,2,1026])'
