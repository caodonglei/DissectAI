import copy
from functools import reduce

def shape_size(shape):
    return reduce(lambda a, b: a * b, shape)

def matrix_add(A, B, inplace=True):
    if not A or not B:
        raise errors.MatrixComputeError('invalid input')

    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise errors.MatrixComputeError('invalid input shape')

    if inplace:
        C = A
    else:
        C = copy.deepcopy(A)

    for row_C, row_B in zip(C, B):
        for i in range(len(row_C)):
            row_C[i] += row_B[i]
    return C

def matrix_adds(matrixs, inplace=True):
    C = matrixs[0]
    for i in range(1, len(matrixs)):
        C = matrix_add(C, matrixs[i], inplace)
    return C
