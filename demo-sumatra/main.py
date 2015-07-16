import numpy
import sys

__version__ = "1.2.3a"

def get_version(): # version numbers are deliberately different, for testing purposes
    return (1, 2, "3b")

def run():
    parameter_file = sys.argv[1]
    parameters = {}
    execfile(parameter_file, parameters) # this way of reading parameters
                                         # is not necessarily recommended
    numpy.random.seed(parameters["seed"])
    distr = getattr(numpy.random, parameters["distr"])
    data = distr(size=parameters["n"])
        
    numpy.savetxt("Data/example2.dat", data)


if __name__ == "__main__":
    run()
