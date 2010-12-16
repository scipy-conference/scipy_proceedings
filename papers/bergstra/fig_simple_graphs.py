import theano
a = theano.tensor.vector('a')  # declare variable
b = a + a**10               # build expression
f = theano.function([a], b) # compile function
print `f([0,1,2])`          # call function
# prints 'array([0,2,1026])'

# compile graph without optimizations
slowf = theano.function([a], b, mode="FAST_COMPILE")
# visualize the unoptimized and optimized expression graphs
theano.printing.pydotprint(slowf, "f_unoptimized", format="pdf")
theano.printing.pydotprint(f, "f_optimized", format="pdf")
