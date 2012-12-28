from sys import exit
import numpy as np
import logging
from sys import _getframe, exc_info


def tests():
    try:        
        test_restrictions()
    except: showexception
    
    try:
        test_coarsen_A()
    except: showexception()
    
    try:
        test_iter()
    except: showexception()
    
    try: test_amg_cycle()
    except: showexception()
        

def restriction(N, shape):
    alpha = len(shape)  # number of dimensions
    R = np.zeros((N / (2 ** alpha), N))
    r = 0  # rows
    NX = shape[0]
    if alpha >= 2:
        NY = shape[1]
    each = 1.0 / (2 ** alpha)
    if alpha == 1:
        coarse_columns = np.array(range(N)).\
                        reshape(shape)\
                        [::2].ravel()
    elif alpha == 2:
        coarse_columns = np.array(range(N)).\
                        reshape(shape)\
                        [::2, ::2].ravel()
    elif alpha == 3:
        coarse_columns = np.array(range(N)).\
                        reshape(shape)\
                        [::2, ::2, ::2].ravel()
    else:
        print "> 3 dimensions is not implemented."
        exit()
    for c in coarse_columns:
        R[r, c] = each
        R[r, c + 1] = each
        if alpha >= 2:
            R[r, c + NX] = each
            R[r, c + NX + 1] = each
            if alpha == 3:
                R[r, c + NX * NY] = each
                R[r, c + NX * NY + 1] = each
                R[r, c + NX * NY + NX] = each
                R[r, c + NX * NY + NX + 1] = each
        r += 1
    return R


def restrictions(N, problemshape, coarsest_level,\
                dense=False, verbose=False):
    alpha = np.array(problemshape).size
    levels = coarsest_level + 1
    # We don't need R at the coarsest level:
    R = list(range(levels - 1))
    for level in range(levels - 1):
        newsize = N / (2 ** (alpha * level))
        R[level] = restriction(newsize,
                    tuple(np.array(problemshape)\
                        / (2 ** level)))
    return R


def test_restrictions():
    saytest()
    N = 64
    problemshape = (8,8)
    coarsest_level = 2
    verbose = False
    R = restrictions(N, problemshape, coarsest_level,\
                dense=False, verbose=verbose)
    
def coarsen_A(A_in, coarsest_level, R, dense=False):
    levels = coarsest_level + 1
    A = list(range(levels))
    A[0] = A_in
    for level in range(1, levels):
        A[level] = np.dot(np.dot(
                            R[level-1],
                            A[level-1]),
                        R[level-1].T)
    return A
    
def test_coarsen_A():
    saytest()
    N = 64
    coarsest_level = 2
    problemshape = (8,8)
    R = restrictions(N, problemshape,coarsest_level)
    A_in = np.empty((N,N))
    coarsen_A(A_in, coarsest_level, R, dense=False)
    

def iterative_solve(A, b, x, iterations):
    N = b.size
    iteration = 0
    for iteration in range(iterations):
        for i in range(N):  # [ 0 1 2 3 4 ... n-1 ]
            x[i] = x[i] + (b[i] - np.dot(
                                    A[i, :],
                                    x.reshape((N, 1)))
                        ) / A[i, i]
    return x
    

def test_iter():
    saytest()
    N = 64
    problemshape = (8,8)
    u_actual = np.random.random((N,))
    A = np.random.random((N,N))
    b = np.dot(A,u_actual)
    u_iter = iterative_solve(A, b, np.zeros((N,)), 10)
    error = u_actual - u_iter
    print 'norm is', np.linalg.norm(error)

    
def amg_cycle(A, b, level, \
            R, parameters, initial='None'):
    # Unpack parameters, such as pre_iterations
    exec ', '.join(parameters) +\
        ',  = parameters.values()'
    if initial == 'None':
        initial = np.zeros((b.size, ))
    coarsest_level = gridlevels - 1
    N = b.size
    if level < coarsest_level:
        u_apx = iterative_solve(\
                                A[level],\
                                b,\
                                initial,\
                                pre_iterations,)
        b_coarse = np.dot(R[level],\
                        b.reshape((N, 1)))
        NH = len(b_coarse)
        b_coarse.reshape((NH, ))
        residual = b - np.dot(A[level], u_apx)
        coarse_residual = np.dot(\
                            R[level],\
                            residual.reshape((N, 1))\
                            ).reshape((NH,))
        coarse_correction = amg_cycle(
                            A,
                            coarse_residual,
                            level + 1,
                            R,
                            parameters,
                            )
        correction = np.dot(\
                            R[level].transpose(),\
                            coarse_correction.\
                            reshape((NH, 1))\
                        ).reshape((N, ))
        u_out = u_apx + correction
        norm = np.linalg.norm(b - np.dot(
                                    A[level],
                                    u_out.\
                                    reshape((N,1))
                                    ))
    else:
        norm = 0
        u_out = np.linalg.solve(A[level],\
                            b.reshape((N, 1)))
    return u_out
    
def test_amg_cycle():
    saytest()
    N = 64
    problemshape = (8,8)
    coarsest_level = 3
    u_actual = np.random.random((N,))
    A_in = np.random.random((N,N))
    b = np.dot(A_in, u_actual)
    R = restrictions(N, problemshape,coarsest_level)
    parameters = {'gridlevels': coarsest_level, 'pre_iterations': 1}
    A_list = coarsen_A(A_in, coarsest_level, R, dense=False)
    cycle_result = amg_cycle(A_list, b, 0, R, parameters, initial='None')
    error = u_actual - cycle_result
    print 'norm is', np.linalg.norm(error), '(1 %i-cycle)' % (coarsest_level-1)

    
    
    
    
    
    
    
# Helper functions for testing.
def whoami(level=1):
    '''Return the name of the calling function. Specifying a level greater than
    1 will return the name of the calling function's caller.'''
    return _getframe(level).f_code.co_name
    

def showexception():
    logging.exception('error in %s:', str(whoami(2)))
    logging.debug("debug: Something awful happened!")
    errortext = exc_info()[0]


def saytest():
    '''Declare that a test is beginning!'''
    print ''
    print 'TEST >>>>', whoami(2)


tests()