import numpy
import theano.tensor as T
from theano import shared, function

# 1. Declare Theano variables
x = T.matrix()
y = T.lvector()
w = shared(numpy.random.randn(100))
b = shared(numpy.zeros(()))
print "Initial model:"
print w.get_value(), b.get_value()
