import dask.array as np
from dask.distributed import Client,wait
from os import getenv
import numpy

def main():
    client = Client(getenv("DASK_MASTER")+":8786")
    #client = Client()
    nt = numpy.sum([x for x in client.nthreads().values()])
    print (int(nt))

if __name__ == "__main__":
    main()


