#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Simplex.py
# Description: library with linear programming and maths services
# -----------------------------------------------------------------------------
#
# Started on  <Sat Nov 14,  01:37:54 2015 Carlos Clavero Munoz>
# Last update <Tue July 12,  12:21:35 2016 Carlos Clavero Munoz>
# -----------------------------------------------------------------------------
#
# $Id:: $
# $Date:: $
# $Revision:: $
# -----------------------------------------------------------------------------
#
# Made by Carlos Clavero Munoz
# 
#

# -----------------------------------------------------------------------------
#     This file is part of PySimplex
#
#     PySimplex is free software: you can redistribute it and/or modify it under
#     the terms of the GNU General Public License as published by the Free
#     Software Foundation, either version 3 of the License, or (at your option)
#     any later version.
#
#     PySimplex is distributed in the hope that it will be useful, but WITHOUT ANY
#     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#     FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
#     details.
#
#     You should have received a copy of the GNU General Public License along
#     with PySimplex.  If not, see <http://www.gnu.org/licenses/>.
#
#     Copyright Carlos Clavero Munoz, 2016
# -----------------------------------------------------------------------------

import numpy as np

import sys
import os.path
import io as io
import types
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math
import operator
import warnings
from PySimplex.rational import rational
warnings.filterwarnings('ignore')


############################ Rational operations ############################


def convertStringToRational(number):
    '''This method receives a number in a string and returns it as a rational object. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(number) != str:
        return None

    # It is checked if the number is a fraction
    if "/" in number:
        # It returns the number as rational
        return rational(int(number.split("/")[0]), int(number.split("/")[1]))
    else:
         # It returns the number as rational
        return rational(int(number), 1)


def convertLineToRationalArray(line):
    ''' This method receives a line(e.g '3 2') in a string, and returns a numpy array with rational objects, that contains
    each element of the line. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(line) != str:
        return None

    # Elements of the line are separated
    coef = line.split(" ")

    # Some elements are deleted
    if "\n" in coef:
        coef.remove("\n")
    if '' in coef:
        coef.remove('')

    lis = []
    for i in coef:
        # It is converted every element
        lis.append(convertStringToRational(i))

    # It returns the array with the line elements
    return np.array(lis)


def rationalToFloat(rat):
    '''This method receives a rational object and returns its float value. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(rat) != rational and type(rat) != int and type(rat) != np.int32:
        return None

    if type(rat) == int or type(rat) == np.int32:
        return float(rat)

    return float(rat.numerator / rat.denominator)


def listPointsRationalToFloat(rationalList):
    '''This method receives a list of points with rational coordinates, and returns a list with the float values of the rational objects. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(rationalList) != list or not isAListOfRationalPoints(rationalList):
        return None

    # Float value is calculated
    floatList = [(float(i[0].numerator / i[0].denominator),
                  float(i[1].numerator / i[1].denominator)) for i in rationalList]
    # It returns the list with the float values
    return floatList


def isAListOfRationalPoints(lis):
    '''This method receives a list of points with rational coordinates, and returns True if all elements of the list are points(tuples) with rational
    coordinates or False if there is any element that is not a point with rational coordinates. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(lis) != list:
        return None

    # It is checked every element of the list
    for i in lis:
        if type(i[0]) != rational or type(i[1]) != rational:
            # It returns False if there is any point that is not a rational
            # point
            return False
    # It returns True if all elements are rational points
    return True


def isAListOfPoints(lis):
    '''This method receives a list of points, and returns True if all elements of the list are points(tuples) or False if
    there is any element that is not a point. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(lis) != list:
        return None

    # It is checked every element of the list
    for i in lis:
        if type(i) != tuple:
            # It returns False if there is any element that is not a point
            return False
    # It returns True if all elements are points
    return True


def isARationalMatrix(mat):
    '''This method receives a bidimensional numpy array or a numpy matrix, and returns True, if ervery element is a rational
    object, or False if there is any element that is not a rational object. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(mat) != np.matrix and type(mat) != np.ndarray:
        return None

    mat = np.asarray(mat)
    # Every element is checked
    for j in mat:
        for i in j:
            if type(i) != rational:

                # It returns False if there is any element that is rational
                return False
    # It returns True if all elements are rational points
    return True


def isARationalArray(arr):
    '''This method receives a numpy array, and returns True, if ervery element is a rational object, or False if
    there is any element that is not a rational object. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(arr) != np.ndarray:
        return None

    # Every element is checked
    for j in arr:
        if type(j) != rational:
                # It returns False if there is any element that is rational
            return False

    # It returns True if all elements are rational points
    return True

##############################################################################


############################ Matrix operations ############################


def determinant(matrix):
    '''This method receives a numpy matrix with rational objects and returns its determinant, in a rational object. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if not isARationalMatrix(matrix) or (type(matrix) != np.ndarray and type(matrix) != np.matrix):
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None
    # If dimensions are 1,1 the determinant is the only one number of the
    # matrix
    if len(matrix) == 1:

        return matrix[0][0]

    # Matrix must be squared
    if matrix.shape[0] != matrix.shape[1]:
        return None

    matrix = np.asarray(matrix)

    # Determinant is calculared
    if len(matrix) == 2:

        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    add = 0
    for i in range(len(matrix)):
        nm = initializeMatrix(len(matrix) - 1, len(matrix) - 1)

        for j in range(len(matrix)):
            if j != i:
                for k in range(1, len(matrix)):
                    index = -1
                    if j < i:
                        index = j
                    elif j > i:
                        index = j - 1
                    nm[index][k - 1] = matrix[j][k]

        if i % 2 == 0:

            add += matrix[i][0] * determinant(nm)
        else:
            add -= matrix[i][0] * determinant(nm)

    if type(add) != rational:
        return rational(int(add), 1)

    # It returns the determinant in a rational object
    return add


def coFactorMatrix(matrix):
    '''This method receives a numpy matrix with rational objects and returns the cofactor matrix. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    # Matrix must be squared
    if matrix.shape[0] != matrix.shape[1] or not isARationalMatrix(matrix) or type(matrix) != np.matrix:
        return None

    # New matrix is initialized
    matrix = np.asarray(matrix)
    nm = initializeMatrix(len(matrix), len(matrix))

    # Cofactor matrix is calculated
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            det = initializeMatrix(len(matrix) - 1, len(matrix) - 1)

            for k in range(len(matrix)):
                if k != i:
                    for l in range(len(matrix)):
                        if l != j:
                            if k < i:
                                index1 = k
                            else:
                                index1 = k - 1
                            if l < j:
                                index2 = l
                            else:
                                index2 = l - 1

                            det[index1][index2] = matrix[k][l]

            detValue = determinant(det)
            nm[i][j] = detValue * ((-1)**(i + j + 2))

    # It returns cofactor matrix
    return np.asmatrix(nm)


def adjMatrix(matrix):
    '''This method receives a numpy matrix with rational objects and returns the adjugate matrix. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    # Matrix must be squared
    if matrix.shape[0] != matrix.shape[1] or not isARationalMatrix(matrix) or type(matrix) != np.matrix:
        return None
    # It returns the adjugate matrix
    return coFactorMatrix(matrix).T


def invertMatrix(matrix):
    '''This method receives a numpy matrix and returns the matrix inverted.'''

    # Matrix must be squared
    if matrix.shape[0] != matrix.shape[1] or not isARationalMatrix(matrix) or type(matrix) != np.matrix:
        return None

    # Determinant is calculated
    det = rational(determinant(matrix).denominator,
                   determinant(matrix).numerator)
    # Adjugate matrix is calculated
    nmatrix = adjMatrix(matrix)

    # It returns the inverted matrix
    return multNumMatrix(det, nmatrix)


def initializeMatrix(rows, col):
    '''This method receives the number of rows and the number of columns of a new matrix, and create a new numpy matrix of
    rational objects which values are 0. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if (type(rows) != int and type(rows) != np.int32) or (type(col) != int and type(col) != np.int32):
        return None

    aux = []
    # List is created with rational objects
    for i in range(rows):
        for j in range(col):
            aux.append(rational(0, 1))

    # Matrix is created
    mat = np.array([aux])

    mat = mat.reshape(rows, col)
    # It returns the matrix with rational objects
    return mat


def createRationalIdentityMatrix(dim):
    '''This method receives a number, and creates a squared identity matrix of this dimensions,
    with rational objects. If the parameters are not correct,it returns None.'''

    # It is checked if parameters are correct

    if type(dim) != int and type(dim) != np.int32:
        return None

    # Matrix is initialized
    mat = initializeMatrix(dim, dim)
    # Pricipal diagonal is changed to 1 values
    for i in range(dim):
        for j in range(dim):
            if i == j:
                mat[i][j] = rational(1, 1)

    # It returns the identity matrix
    return np.asmatrix(mat)

'''
def matrixToRational(matrix):
    This mehtod receives a numpy matrix with int values, and changes them to rational objects. It returns
    a numpy matrix with the rational objects. If the parameters are not correct, it returns None.

    # It is checked if parameters are correct

    if type(matrix) != np.matrix and type(matrix) != np.ndarray:
        return None
    
    print(matrix)
    matrix = np.asarray(matrix)

    # Values are changed to rational
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if type(matrix[i][j]) != rational:
                matrix[i][j] = rational(int(matrix[i][j]), 1)
    # It returns the matrix with new values
    return np.asmatrix(matrix)
'''

def multNumMatrix(num, matrix):
    '''This method receives a rational object and a numpy matrix of rational objects, and returns the
    multiplication between the number and the matrix in a numpy matrix. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(num) != rational or type(matrix) != np.matrix or not isARationalMatrix(matrix):
        return None

    matrix = np.asarray(matrix)
    # Every element of the matrix is multiplicated by the number
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] = matrix[i][j] * num
    # It returns the matrix with the multiplication final values
    return np.asmatrix(matrix)


def twoMatrixEqual(matrix1, matrix2):
    '''This method receives two numpy matrix of rational objects and returns True if they are equal,
    or False if they are not. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(matrix1) != np.matrix or not isARationalMatrix(matrix1) or type(matrix2) != np.matrix or not isARationalMatrix(matrix2):
        return None

    # If dimensions are different, matrix are different
    if matrix1.shape != matrix2.shape:
        return False

    matrix1 = np.asarray(matrix1)
    matrix2 = np.asarray(matrix2)

    # It is compared every element of each matrix
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            if matrix1[i][j] != matrix2[i][j]:
                # It returns False if matrix are not equal
                return False
    # It returns True if matrix are equal
    return True


def printMatrix(matrix):
    '''This method receives a numpy matrix with rational objects and returns the matrix in str format. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if (type(matrix) != np.matrix and type(matrix) != np.ndarray) or not isARationalMatrix(matrix):
        return False

    m = ""
    matrix = np.asarray(matrix)
    for i in range(matrix.shape[0]):

        if i == 0:
            m += "["
        else:
            m += "\n"
        for j in range(matrix.shape[1]):
            if j != matrix.shape[1] - 1:
                m += str(matrix[i][j]) + " , "
            else:
                m += str(matrix[i][j])
    m += "]"
    return m


def multMatrix(matrix1, matrix2):
    '''This method receives two numpy matrix of rational objects and multiplicates them. It returns the result in a numpy matrix. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    # The number of columns of the first matrix must be equal to the number of
    # rows of the second matrix
    if matrix1.shape[1] != matrix2.shape[0] or type(matrix1) != np.matrix or type(matrix2) != np.matrix or not isARationalMatrix(matrix1) or not isARationalMatrix(matrix2):
        return None
    result = initializeMatrix(matrix1.shape[0], matrix2.shape[1])

    matrix1 = np.asarray(matrix1)
    matrix2 = np.asarray(matrix2)
    # Matrix are multiplied
    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[1]):
            for k in range(matrix1.shape[1]):

                result[i][j] += matrix1[i][k] * matrix2[k][j]

    # It returns the result of the multiplication
    return np.asmatrix(result)

##################################################################


############################ Simplex ############################

def variablesNoiteration(matrix, variablesIteration):
    '''This method receives a numpy matrix and a numpy array with the variables of the iteration,
    and returns an array with the variables which are not in the iteration.If the parameters are not
    correct, it returns None.'''
    # It is checked if parameters are correct
    if type(matrix) != np.matrix or type(variablesIteration) != np.ndarray or len(variablesIteration) > matrix.shape[1]:
        return None
    # set of all variables
    groupOfVariables = np.array(range(matrix.shape[1]))
    # variables which are not in the iteration
    varNoIter = np.array([])

    # It is checked which variables are not in the iteration
    for i in groupOfVariables:
        if i not in variablesIteration:
            varNoIter = np.append(varNoIter, i)

    # It returns variables which are not in the iteration
    return varNoIter


def calcMinNoNan(setOfVal):
    '''This method receives a numpy array, and returns the minimum no nan value of it or None if all values are
    nan values. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(setOfVal) != np.ndarray:
        return None

    auxiliar = np.array([])

    for i in setOfVal:
        # Only considered rational values(nan values are not considered)
        if(type(i) == rational):
            auxiliar = np.append(auxiliar, i)
    # It returns the minimum value of the selected values.
    if len(auxiliar) > 0:
        return min(auxiliar)
    else:
        # It returns None if the problem is not bounded
        return None


def calculateIndex(array, value):
    '''This method receive a numpy array and a value, and returns the position of this value in the array. If
    the value is not in the array, the method returns None. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(array) != np.ndarray or (type(value) != int and type(value) != float and type(value) != np.float64 and type(value) != np.int32 and type(value) != rational):
        return None

    for i in range(array.size):

        if(array[i] == value):
            # It returns the position of the value
            return i
    # It returns None
    return None


def calculateBaseIteration(totalMatrix, columnsOfIteration):
    '''This method returns the base of an iteration in a numpy matrix. It recieves a numpy matrix(this matrix should be the initial
    matrix of the problem) and a numpy array with the columns of iteration. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(totalMatrix) != np.matrix or type(columnsOfIteration) != np.ndarray or len(columnsOfIteration) > totalMatrix.shape[1]:
        return None

    # It is added the first column of iteration to the base matrix
    B = np.matrix(np.c_[totalMatrix[:, [columnsOfIteration[0]]]])
    # The others column of iteration are added to  base matrix
    for i in range(1, columnsOfIteration.size):
        B = np.matrix(np.c_[B, totalMatrix[:, [columnsOfIteration[i]]]])
    # It returns the base of iteration
    return B


def showBase(Base, name):
    '''This method receives the base of iteration in a numpy matrix with rational objects and prints it.If
    the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(Base) != np.matrix or not isARationalMatrix(Base) or type(name) != str:
        return None

    print(name + " = " + printMatrix(Base) + "\n")


def calculateIterationSolution(invertedBase, resourcesVector):
    '''This method receives the inverted base of iteration in a numpy matrix of rational objects and the resoruces vector in a
     numpy array of rational objects. It returns the solution of the iteration in a numpy matrix. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if not isARationalMatrix(invertedBase) or not isARationalArray(resourcesVector) or type(invertedBase) != np.matrix or type(resourcesVector) != np.ndarray or len(resourcesVector) != invertedBase.shape[0]:

        return None

    # The resources vector is casted to a matrix to perform the vector product properly
    # It returns the product between the inverted base and the resources vector,
    # in a numpy matrix

    return np.asarray(multMatrix(invertedBase, np.asmatrix(resourcesVector).T))


def showSolution(sol):
    '''This method receives the solution of iteration in a column numpy array of rational objects and prints it.If
    the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(sol) != np.ndarray or not isARationalMatrix(sol):
        return None

    print("x = inv(B)*b = " + printMatrix(np.asmatrix(sol)) + "\n")


def calculateCB(columnsOfIteration, functionVector):
    '''This method receives the columns of the iteration in a numpy array and the function vector in
    other numpy array. It returns the value of function vector for this iteration in a numpy array.If the
    parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(columnsOfIteration) != np.ndarray or type(functionVector) != np.ndarray or len(columnsOfIteration) > len(functionVector):
        return None

    CB = np.array([])
    # The coefficient of variables of the vector function which are in the
    # iteration are added
    for i in columnsOfIteration:
        CB = np.append(CB, functionVector[i])
    # It returns the value of function vector for this iteration in a numpy
    # array
    return CB


def showCB(CBValue):
    '''This method receives the value of function vector in a numpy array of rational objects and prints it. If the
    parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(CBValue) != np.ndarray or not isARationalArray(CBValue):
        return None
    print("CB = " + printMatrix(np.asmatrix(CBValue)) + "\n")


def calculateFunctionValueOfIteration(solution, CB):
    '''This method recieve the solution of the iteration in a column numpy array of rational objects and the value
    of function vector for this iteration in a numpy array of rational objects. It returns the value of the function in this iteration. If the
    parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(solution) != np.ndarray or type(CB) != np.ndarray or len(solution) != len(CB) or not isARationalMatrix(solution) or not isARationalArray(CB):
        return None

    # The solution is casted to a matrix to perform the vector product properly
   
    solution=np.asmatrix(solution)
    CB = np.asmatrix(CB)
    # It returns the value of the function for this iteration
    return multMatrix(CB, solution)


def showFunctionValue(functionValue):
    '''This method receives the the value of the function in a numpy array and prints it. If the
    parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(functionValue) != np.matrix:
        return None

    print("z = CB*x = " + str(np.asarray(functionValue)[0][0]) + "\n")


def calculateYValues(variablesNoIteration, iterationBase, totalMatrix):
    '''This method receives the variables which are not in the iteration in a numpy array,
    the iteration base in a numpy matrix and the initial matrix of the problem, in a numpy matrix.
    It returns the y values for this iteration. The elements of iteration base and total matrix must be rational objects.
    If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(totalMatrix) != np.matrix or type(iterationBase) != np.matrix or type(variablesNoIteration) != np.ndarray or iterationBase.shape[0] != totalMatrix.shape[0] or len(variablesNoIteration) > totalMatrix.shape[1] or not isARationalMatrix(totalMatrix) or not isARationalMatrix(iterationBase):
        return None

    y = np.array([])
    # They are only selected the columns of the intial matrix which variable are not in the iteration and
    # these columns are multiplied by inverted base of the iteration
    for i in variablesNoIteration:
        y = np.append(y, multMatrix(invertMatrix(iterationBase),
                                    totalMatrix[:, [i]]))

    # The values are appended in this way [1,1,1,1,1], so I have to separate
    # in differents columns
    y = y.reshape(variablesNoIteration.size, totalMatrix.shape[0])
    # It returns the y values in a numpy array
    return y


def showYValues(variablesNoIteration, y):
    '''This method receives the y values in a numpy array and prints it. It also receives the variables
    which are not in the iteration in a numpy array. Every element of the coefficent of the y values must be a
    rational object.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(y) != np.ndarray or type(variablesNoIteration) != np.ndarray or not isARationalMatrix(y):
        return None

    for i in range(variablesNoIteration.size):

        print("y" + str(int(variablesNoIteration[i])) + " = inv(B)*a" + str(
            int(variablesNoIteration[i])) + " = " + printMatrix(np.asmatrix(y[i])) + "\n")


def calculateZC(functionVector, variablesNoIteration, CB, y):
    '''This method receives the complete function vector in a numpy array, the variables which are
    not in the iteration in a numpy array, the function vector of the iteration in a numpy array and the
    y values in a numpy array. It returns the values of the input rule in a numpy array. The elements of function vector,
    the function vector of the iteration, and the y values, must be rational objects. If the parameters are
    not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(functionVector) != np.ndarray or type(variablesNoIteration) != np.ndarray or type(CB) != np.ndarray or type(y) != np.ndarray or len(CB) != y.shape[1] or len(variablesNoIteration) > len(functionVector) or len(CB) > len(functionVector) or not isARationalArray(functionVector) or not isARationalArray(CB) or not isARationalMatrix(y):
        return None

    # The CB is casted to a matrix to perform the vector product properly
    CB = np.asmatrix(CB)
    Z_C = np.array([])
    for i in range(variablesNoIteration.size):
        y[i] = np.asmatrix(y[i])
        # It is performed the product of the vector function of the iteration and each y value, and
        # it is subtracted the coefficient of the function variables which are
        # not in the iteration
        Z_C = np.append(Z_C, np.asarray(multMatrix(np.asmatrix(CB), np.asmatrix(y[i]).T)[
                        0][0]) - (functionVector[variablesNoIteration[i]]))
    # It returns the input rule values in a numpy array
    return Z_C


def showZCValues(variablesNoIteration, Z_C):
    '''This method receives the values of the input rule in a numpy array and prints it. It also receives the variables
    which are not in the iteration in numpy array.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(Z_C) != np.ndarray or type(variablesNoIteration) != np.ndarray or len(variablesNoIteration) != len(Z_C):
        return None
    print("Input rule:\n")
    for i in range(variablesNoIteration.size):
        print("Z_C" + str(int(variablesNoIteration[i])) + " = CB*y" + str(int(variablesNoIteration[
              i])) + "- C" + str(int(variablesNoIteration[i])) + " = " + str(Z_C[i]) + "\n")


def thereIsAnotherIteration(inputRuleValues):
    '''This method receive the values of the input rule in a numpy array. It returns if there is another iteration
    or not, and if the problem has many solutions.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(inputRuleValues) != np.ndarray:
        return None

    if(np.min(inputRuleValues) < 0):
        # It returns true if there is another iteration
        return True
    elif(np.min(inputRuleValues) == 0):
        # It returns -1 if the problem has many solutions
        return -1
    else:
        # It returns true if there is not another iteration
        return False


def showNextIteration(thereIsAnotherIter):
    '''This method receives if there is another iteration or not, and prints an explanation of it. It receives true
    if there is another iteration, false if not, and -1 if the problem has many solutions.If the parameters
    are not correct, it returns None.'''

    # It is checked if parameters are correct
    if thereIsAnotherIter != True and thereIsAnotherIter != False and thereIsAnotherIter != -1:
        return None

    if thereIsAnotherIter and thereIsAnotherIter != -1:
        print("The problem is not finished. There is another iteration,because there is at least one negative reduced cost.\n")
    elif thereIsAnotherIter == -1:
        print("The problem is finished. There are a large amount of solutions.\n")
    else:
        print("The problem is finished, because there is not any negative reduced cost.\n")


def calculateVarWhichEnter(variablesNoIteration, inputRuleValues):
    '''This method receives the variables which are not in the iteration in a numpy array and
    the input rule values in a numpy array. It returns the variable which enters in the next iteration. If the parameters
    are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(inputRuleValues) != np.ndarray or type(variablesNoIteration) != np.ndarray or len(variablesNoIteration) != len(inputRuleValues):
        return None
    # It returns the variable which has the minimum value in the input rule
    return variablesNoIteration[np.argmin(inputRuleValues)]


def showVarWhichEnter(variableWhichEnter):
    '''This method receives the value of variable which enters and prints it.If the parameters
    are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(variableWhichEnter) != np.float64 and type(variableWhichEnter) != int and type(variableWhichEnter) != float and type(variableWhichEnter) != np.int32:
        return None

    print("The variable which enters in the next iteration is " +
          str(int(variableWhichEnter)) + "\n")


def calculateExitValues(inputRuleValues, yValues, sol):
    '''This method receives the input rule values in a numpy array, the y values of the iteration in a numpy array
    and the solution of the iteration in a numpy array.It returns the values of the output rule in two numpy arrays. The first
    array contains the values wihtout negative o 0 denominators, and the second contains the values with them. All elements of the parameter arrays,
    must be rational objects. If the parameters are not correct, it returns None.'''
    # It is checked if parameters are correct
    if type(inputRuleValues) != np.ndarray or type(yValues) != np.ndarray or type(sol) != np.ndarray or len(yValues) != len(inputRuleValues) or yValues.shape[1] != len(sol) or not isARationalMatrix(sol) or not isARationalArray(inputRuleValues) or not isARationalMatrix(yValues):
        return None

    auxiliar = np.array([])
    auxiliar2 = np.array([])
    # The y column of the variable which are going to enter in the next
    # iteration is selected
    yp = yValues[np.argmin(inputRuleValues)]

    # It is divided the solution values of the iteration between the selected y values.
    # The negative and zero values of y are discarded.
    for i in range(sol.size):
        auxiliar2 = np.append(auxiliar2, sol[i] / yp[i])

        #if(yp[i].numerator * sol[i][0].denominator <= 0):
        if yp[i].numerator<= 0 or yp[i].denominator<0:    
            auxiliar = np.append(auxiliar, np.nan)
        else:
            auxiliar = np.append(auxiliar, sol[i] / yp[i])

    # It returns the values of the output rule in two numpy arrays

    return auxiliar, auxiliar2


def showExitValues(exitValues):
    '''This method receives the values of the output rule in a numpy array and prints it. All elements of the exit values must be rational objects.
    If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(exitValues) != np.ndarray or not isARationalArray(exitValues):

        return None

    print("Output rule:\n")
    print("O = min {" + str(printMatrix(np.asmatrix(exitValues))) + "}\n")


def calculateO(exitValues):
    '''This method receives the values of the output rule in a numpy array, with rational elements or Nan. It returns O value(the minimum output
    rule values) or None if the problem is not bounded. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(exitValues) != np.ndarray:
        return None

    O = calcMinNoNan(exitValues)
    # It returns the O value
    return O


def showOValue(O):
    '''This method receives the value of O and prints it. If the parameters
    are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(O) != int and type(O) != float and type(O) != np.float64 and type(O) != np.int32 and type(O) != rational:
        return None
    print("O = " + str(O) + "\n")


def calculateVarWhichExit(columnsOfIteration, outputRuleValues):
    '''This method receives the variables of the iteration in a numpy array and
    the values of the output rule in a numpy array. It returns the variable which
    exits in this iteration or None if the problem is not bounded. If the parameters
    are not correct, it returns None.'''

    # It is checked if parameters are correct

    if type(columnsOfIteration) != np.ndarray or type(outputRuleValues) != np.ndarray:

        return None

    O = calculateO(outputRuleValues)

    # It returns the variable which has the minimum no Nan value in the
    # output rule

    if O != None:

        return columnsOfIteration[calculateIndex(outputRuleValues, O)]
    else:
        # It returns None if the problem is not bounded
        return None


def showVarWhichExit(varWhichExit):
    '''This method receives the value of variable which exits and prints it. If the parameters
    are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(varWhichExit) != int and type(varWhichExit) != float and type(varWhichExit) != np.float64 and type(varWhichExit) != np.int32:
        return None

    print("The variable which exits in this iteration is " +
          str(int(varWhichExit)) + "\n")


def showIterCol(columnsOfIteration):
    '''This method receives the columns of the iteration in a numpy array and prints it. If the parameters
    are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(columnsOfIteration) != np.ndarray:
        return None

    print("The variables of this iteration are " +
          str(columnsOfIteration) + "\n")


def solveIteration(totalMatrix, b, functionVector, columnsOfIteration):
    '''This method receives the initial matrix of the problem in a numpy matrix, the resources vector
    in a numpy array, the complete function vector in a numpy array and the variables of the iteration
    in a numpy array. It returns the solution of the iteration, the value of the function for this iteration,
    the variable which enters in the next iteration, the variable which exits in this iteration and if there
    is another iteration. The elements of matrix, function vector and resources vector, must be rational objects.
    If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(totalMatrix) != np.matrix or type(columnsOfIteration) != np.ndarray or type(b) != np.ndarray or type(functionVector) != np.ndarray or len(b) != totalMatrix.shape[0] or len(functionVector) != totalMatrix.shape[1] or len(columnsOfIteration) != totalMatrix.shape[0] or not isARationalMatrix(totalMatrix) or not isARationalArray(b) or not isARationalArray(functionVector):
        return None

    nextIteration = False

    # The iteration base is calculated
    B = calculateBaseIteration(totalMatrix, columnsOfIteration)
    # The base of iteration is showed
    showBase(B, "B")

    # The iteration base is inverted
    invB = invertMatrix(B)
    # The inverted base of iteration is showed
    showBase(invB, "invB")
    # The solution of the iteration is calculated
    x = calculateIterationSolution(invB, b)

    # The solution of the iteration  is showed

    showSolution(x)
    #x = np.asarray(x)

    # The function vector for this iteration  is calculated
    CB = calculateCB(columnsOfIteration, functionVector)
    # The function vector is showed
    showCB(CB)
    # The function value for this iteration is calculated
    z0 = calculateFunctionValueOfIteration(x, CB)
    # Function value is showed
    showFunctionValue(z0)

    # y values are calculated
    y = calculateYValues(variablesNoiteration(
        totalMatrix, columnsOfIteration), B, totalMatrix)
    # y values are showed
    showYValues(variablesNoiteration(totalMatrix, columnsOfIteration), y)
    # Input rule values are calculated

    Z_C = calculateZC(functionVector, variablesNoiteration(
        totalMatrix, columnsOfIteration), CB, y)
    # Input rule values are showed
    showZCValues(variablesNoiteration(totalMatrix, columnsOfIteration), Z_C)

    # It is checked if there is another iteration
    nextIteration = thereIsAnotherIteration(Z_C)
    # It is showed if there is another iteration
    showNextIteration(nextIteration)
   
    variableWhichEnters = []
    variableWhichExits = []

    # If there is another iteration
    if nextIteration and nextIteration != -1:
        # Variable which enters is calculated
        variableWhichEnters = calculateVarWhichEnter(
            variablesNoiteration(totalMatrix, columnsOfIteration), Z_C)
        # O value is calculated
        O = calculateO(calculateExitValues(Z_C, y, x)[0])
        # Variable which exits is calculated

        # Variable which exits is calculated

        variableWhichExits = calculateVarWhichExit(
            columnsOfIteration, calculateExitValues(Z_C, y, x)[0])

        if O != None:
            # Variable which enters is showed
            showVarWhichEnter(variableWhichEnters)
            # Output rule values are showed
            showExitValues(calculateExitValues(Z_C, y, x)[1])
            # O value is showed
            showOValue(O)
            # Variable which exits is showed
            showVarWhichExit(variableWhichExits)

    # return the solution, the value of the function, the variable which enters,the variable which enters and
    # if there is another iteration
    return [x, z0, variableWhichEnters, variableWhichExits, nextIteration]


def identityColumnIsInMatrix(matrix, column):
    '''This method receives a numpy matrix of rational objects and a index of a column(this index is the index of
    an identity matrix column). It returns the index of the column in the matrix. If the parameters
    are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(matrix) != np.matrix or (type(column) != int and type(column) != float and type(column) != np.float64 and type(column) != np.int32) or column >= matrix.shape[1] or not isARationalMatrix(matrix):
        return None

    # Identity matrix with the properly size is generated
    identity = createRationalIdentityMatrix(int(matrix.shape[0]))

    for j in range(matrix.shape[1]):
        # It is checked if the column(of the identity matrix) introduced as parameter of the method
        # is in the matrix
        if twoMatrixEqual(matrix[:, j], identity[:, [column]]):
            # It returns the index of the column in the matrix.
            return j


def variablesFirstIteration(totalMatrix):
    '''This method receives the initial matrix with rational objects of the problem in a numpy matrix. It returns the
    variables of the first iteration of the problem in a numpy array.If the parameters are not
    correct, it returns None.'''

    # It is checked if parameters are correct
    if type(totalMatrix) != np.matrix or not isARationalMatrix(totalMatrix):
        return None

    aux = []
    # It is looked for where are the columns of the identity matrix in the
    # initial matrix of the problem
    for j in range(totalMatrix.shape[0]):
        aux.append(identityColumnIsInMatrix(totalMatrix, j))

    variablesFirstIteration = np.array(aux)
    # It returns the index of the columns of the initial matrix which are equal to
    # columns of the identity matrix
    return variablesFirstIteration


def calculateColumnsOfIteration(variableWhichEnters, variableWhichExits, previousVariables):
    '''This method receives the variable which enters in this iteration,the variable which exits
    in the previous iteration and the variables of the previous iteration in a numpy array. It returns
    the variables of the next iteration in a numpy array. If the parameters are not correct, it
    returns None.'''
    # It is checked if parameters are correct
    if type(previousVariables) != np.ndarray or (type(variableWhichEnters) != int and type(variableWhichEnters) != float and type(variableWhichEnters) != np.float64 and type(variableWhichEnters) != np.int32) or (type(variableWhichExits) != int and type(variableWhichExits) != float and type(variableWhichExits) != np.float64 and type(variableWhichExits) != np.int32):
        return None

    # It is deleted variable which exits
    variablesOfIteration = np.delete(previousVariables, np.where(
        previousVariables == variableWhichExits))
    # It is add variable which enters
    variablesOfIteration = np.append(variablesOfIteration, variableWhichEnters)
    # The variables of the iteration are sorted
    variablesOfIteration = np.sort(variablesOfIteration)
    # It returns the variables of the iteration in a numpy array
    return variablesOfIteration


def completeSolution(variablesOfIter, numberOfVariables, iterationSolution):
    '''This method receives the variables of the last iteration in a numpy array of rational objects, the total number of variables
    and the solution of the iteration in a numpy array. It returns the complete solution of the problem(with the
    variables which are not in the iteration). If the parameters are not correct, it returns None.'''
  
    # It is checked if parameters are correct
    if type(variablesOfIter) != np.ndarray or (type(numberOfVariables) != int and type(numberOfVariables) != float and type(numberOfVariables) != np.float64 and type(numberOfVariables) != np.int32) or type(iterationSolution) != np.ndarray or len(variablesOfIter) > numberOfVariables or len(variablesOfIter) > numberOfVariables or len(iterationSolution) != len(variablesOfIter) :
        return None
  
    # All values of the solution are initialized to zero value
    solution = initializeMatrix(numberOfVariables, 1)
    # The values of the iteration solution are introduced in the solution

    for i in range(numberOfVariables):

        if(i in variablesOfIter):

            solution[i] = iterationSolution[
                int(np.where(variablesOfIter == i)[0])]
   
    return solution


def addIdentityColumns(matrixInitial):
    '''This method receive a matrix in a numpy matrix of rational objects. It returns the identity columns 
    which are not in the matrix yet.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(matrixInitial) != np.matrix or not isARationalMatrix(matrixInitial):
        return None

    # An identity matrix is generated with the properly shape
    identity = createRationalIdentityMatrix(int(matrixInitial.shape[0]))
    lis = []
    # It is checked which are columns of the identity matrix that are in the
    # matrix
    for i in range(identity.shape[1]):
        for j in range(matrixInitial.shape[1]):
            if twoMatrixEqual(matrixInitial[:, j], identity[:, [i]]) and i not in lis:
                lis.append(i)

    # The columns of the identity matrix that are in the matriz, are deleted
    # of the identity matrix
    identity = np.delete(identity, lis, 1)
    # It returns the columns which are not in the matrix yet
    return identity


def isStringList(lis):
    '''This method receives a list. The method returns False if all elements of the list are strings or
    False if not. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(lis) != list:
        return None

    # If there is any value which is not a string, it return False
    for i in lis:
        if type(i) != str:
            # It returns False
            return False
    # return True, if all values are strings
    return True


def calculateArtificialValueInFunction(array):
    ''' This method receives the function vector in a numpy array,calculates the value of the
    coefficient in function for an artificial variable and returns it. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(array) != np.ndarray:
        return None

    value = 0
    # It is changed every negative value
    for i in array:
        if(i < 0):
            value += i * -1
        else:
            value += i

    value += rational(1, 1)
    value = int(rationalToFloat(value))
    # It is calculated the nearest ten
    while(value % 10 != 0):
        value += 1

    # It returns the value of the coefficent for an artificial variable
    return rational(value, 1)


def addArtificialVariablesToFunctionVector(vector, numOfArtificialVariables):
    '''This method receives the function vector in a numpy array of rational objects and the number of artificial variables
    of the problem. It returns the function vector with the coefficient of the articial variables added in
    a numpy array. If the parameters are not correct, it returns None.'''

  
    # It is checked if parameters are correct
    if type(vector) != np.ndarray or (type(numOfArtificialVariables) != int and type(numOfArtificialVariables) != float and type(numOfArtificialVariables) != np.float64 and type(numOfArtificialVariables) != np.int32) or not isARationalArray(vector):
        return None

    for i in range(numOfArtificialVariables):
        # Coefficient of the articial variables must be greater than the sum of
        # the rest of the coefficients and negative

        vector = np.append(
            vector, calculateArtificialValueInFunction(vector) * -1)
    # It returns the function vector with the coefficient of the articial
    # variables added in a numpy array

    return vector


def calculateWhichAreArtificialVariables(vector, numOfArtificialVariables):
    '''This method receives the vector function in a numpy array, and the number of the artificial
    variables(The artificial variables must be in the last positions of the function vector). It
    returns which are the artificial variables in a list. If the parameters are not correct, it
    returns None.'''

    # It is checked if parameters are correct
    if type(vector) != np.ndarray or (type(numOfArtificialVariables) != int and type(numOfArtificialVariables) != float and type(numOfArtificialVariables) != np.float64 and type(numOfArtificialVariables) != np.int32):
        return None

    varArtific = []
    k = len(vector)
    j = len(vector) + numOfArtificialVariables
    # It is calculated which are artificial variables
    while(k < j):
        varArtific.append(k)
        k += 1
    # return the artificial variables
    return varArtific


def checkValueOfArtificialVariables(varArtificial, solution):
    '''This method receives which are the artificial variables in a list and the solution of the iteration in
    a numpy array with rational objects.It returns a list which contains the artificial values which have a positive value. If the
    parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(varArtificial) != list or type(solution) != np.ndarray or not isARationalMatrix(solution):
        return None

    positiveResult = []

    varArtificial = [x - 1 for x in varArtificial]

    for i in range(len(solution)):
        if i in varArtificial and (solution[i][0].numerator > 0):
            positiveResult.append(i)
    # It returns the artificial values which have a positive value in a list
    return positiveResult


def omitComments(listOfstrings):
    '''This method receives a list of strings and deletes every comment that it has. The comments must start
    with "#" or "//". It returns the same list of strings without comments. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(listOfstrings) != list or not isStringList(listOfstrings):
        return None

    aux = []
    # Comments are looked for
    for i in range(len(listOfstrings)):
        if listOfstrings[i].strip().startswith("//") or listOfstrings[i].strip().startswith("#"):
            aux.append(i)
        elif "#" in listOfstrings[i] and not listOfstrings[i].strip().startswith("//"):
            listOfstrings[i] = listOfstrings[i].split("#")[0]
        elif "//" in listOfstrings[i] and not listOfstrings[i].strip().startswith("#"):
            listOfstrings[i] = listOfstrings[i].split("//")[0]
    # Comments are deleted
    for i in sorted(aux, reverse=True):
        del listOfstrings[i]
    # It returns the the list of strings without comments
    return listOfstrings


def proccessFile(file):
    '''This method receive the name of a file which contains a linnear programming problem. It returns
    the initial matrix of the problem in a numpy matrix with rational objects, the resources vector of
    the problem in a numpy array with rational objects,the signs of the restrictions in a string list
    and the function of the problem in a string. If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if type(file) != io.TextIOWrapper:
        return None

    file.seek(0, 0)
    # The lines of the file are readed
    data = file.readlines()
    data = omitComments(data)

    # Decimal number are not allowed
    for i in data:
        if "." in i:
            return None
    # file.close()
    matrix = []
    resources = []
    sign = []

    # The funtion is in the first line of the file
    func = data[0]
    # Each restriction is proccesed
    for i in range(1, len(data)):
        if "<" in data[i] and "<=" not in data[i]:
            matrix.append(convertLineToRationalArray(
                data[i].split("<")[0]))
            resources.append(convertStringToRational(
                data[i].split("<")[1].strip("/n")))
            sign.append("<")
        elif ">" in data[i] and ">=" not in data[i]:
            matrix.append(convertLineToRationalArray(
                data[i].split(">")[0]))
            resources.append(convertStringToRational(
                data[i].split(">")[1].strip("/n")))
            sign.append(">")
        elif ">=" in data[i]:
            matrix.append(convertLineToRationalArray(
                data[i].split(">=")[0]))
            resources.append(convertStringToRational(
                data[i].split(">=")[1].strip("/n")))
            sign.append(">=")
        elif "<=" in data[i]:
            matrix.append(convertLineToRationalArray(
                data[i].split("<=")[0]))
            resources.append(convertStringToRational(
                data[i].split("<=")[1].strip("/n")))
            sign.append("<=")
        elif "=" in data[i] and "<" not in data[i] and ">" not in data[i]:
            matrix.append(convertLineToRationalArray(
                data[i].split("=")[0]))
            resources.append(convertStringToRational(
                data[i].split("=")[1].strip("/n")))
            sign.append("=")

   # It returns the matrix,the resources vector, the signs of the restrictions
   # and the fuction of the problem
    return np.matrix(matrix), np.array(resources), sign, func


def convertFunctionToMax(function):
    '''This method receives the function vector in a string. It returns the function converted to a max function in
    a numpy array. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct

    if type(function) != str:
        return None

    # If the function is a min funcion is converted to max funcion and is
    # casted to a numpy array
    if "min" in function.lower():
        functionVector = convertLineToRationalArray(function.strip("min "))
        functionVector = [rational(i.numerator * -1, i.denominator)
                          for i in functionVector]
        functionVector = np.array(functionVector)
    # If the function is a max funcion is casted to a numpy array
    elif "max" in function.lower():
        functionVector = convertLineToRationalArray(function.strip("max "))
    # It returns the function vector

    return functionVector


def invertSign(previousSign):
    '''This method receives a string with a sign and returns other string with sign inverted. If the
    parameters are not correct, it returns None.'''

    # It is checked if parameters are correct

    if type(previousSign) != str:
        return None

    newSign = ""
    if previousSign == "<":
        newSign = ">"
    elif previousSign == ">":
        newSign = "<"
    elif previousSign == ">=":
        newSign = "<="
    elif previousSign == "<=":
        newSign = ">="
    elif previousSign == "=":
        newSign = "="
    # It returns the sign inverted
    return newSign


def negativeToPositiveResources(matrix, resources, sign):
    '''This method receives a matrix in a numpy matrix of rational objects, a vector resources in a numpy array of rational objects and a list of strings
    with the signs. It checks if there is any negative value of the resources, and changes it to a positive value. It
    also changes the restriction and inverted the sign. It returns the matrix in a numpy matrix of rational objects, the vector resources in a numpy array of rational objects
    and a list of strings with the signs. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(matrix) != np.matrix or type(resources) != np.ndarray or type(sign) != list or matrix.shape[0] != len(resources) or matrix.shape[0] != len(sign) or len(resources) != len(sign) or not isStringList(sign) or not isARationalMatrix(matrix) or not isARationalArray(resources):
        return None

    # If there are negative values in the resources, the restrictions, the
    # resources and the sign change
    matrix = np.asarray(matrix)
    for i in range(len(resources)):
        if resources[i].numerator < 0:

            resources[i] = rational(
                resources[i].numerator * -1, resources[i].denominator)

            matrix[i] = [rational(j.numerator * -1, j.denominator)
                         for j in matrix[i]]
            sign[i] = invertSign(sign[i])

    # It returns the matrix, the resources and the sign
    return np.asmatrix(matrix), resources, sign


def convertToStandardForm(matrix, resources, sign, function):
    '''This method receives a matrix with restricions in a numpy matrix of rational objects, a vector resources in a numpy array of rational objects,
    a list of strings with the signs and a string with the function. It returns the matrix in numpy matrix of rational objects, the vector resources in
    a numpy array of rational objects,a list of strings with the signs and a numpy array of rational objects with the coefficents of the function converted to standard form . If the
    parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(matrix) != np.matrix or type(resources) != np.ndarray or type(sign) != list or type(function) != str or matrix.shape[0] != len(resources) or matrix.shape[0] != len(sign) or len(resources) != len(sign) or not isStringList(sign) or not isARationalMatrix(matrix) or not isARationalArray(resources):
        return None
    # The function is converted to a max function
    function = convertFunctionToMax(function)

    # The negative resources are converted to positive resources
    positiveResources = negativeToPositiveResources(matrix, resources, sign)
    matrix = positiveResources[0]
    resources = positiveResources[1]
    sign = positiveResources[2]

   # Change the sign to equal and add variables to matrix and to the function
    for i in range(len(sign)):
        newColumn = initializeMatrix(len(sign), 1)

        if sign[i] != "=":
            if sign[i] == "<=":

                newColumn[i] = rational(1, 1)

            elif sign[i] == ">=":
                newColumn[i] = rational(-1, 1)

            matrix = np.c_[matrix, newColumn]
            sign[i] = "="
            function = np.append(function, rational(0, 1))

    # It returns the matrix, the resources, the sign and the function vector
    return matrix, resources, sign, function


def showStandarForm(matrix, resources, function):
    '''This method receives a numpy matrix of rational objects with the restrictions without sign and resources, a numpy array of rational objects
    with the resources and a numpy array of rational objects with the function vector, all of them are in standard form. The method shows
    them.'''

    if type(matrix) != np.matrix or type(resources) != np.ndarray or type(function) != np.ndarray or not isARationalMatrix(matrix) or not isARationalArray(resources) or not isARationalArray(function):
        return None

    print("max" + printMatrix(np.asmatrix(function)) + "\n")
    for i in range(len(resources)):
        print(printMatrix(np.asmatrix(matrix[i])) + " = " + str(resources[i]))
    print("\n")


def calculateSolutionOfDualProblem(colsOfIteration, function, totalMatrix):
    '''This method receives the columns of the last iteration in a numpy array, the complete function vector in a numpy array with rational objects and
    the initial matrix of the problem in a numpy matrix with rational objects. It returns the solution of the dual problem in a numpy array. If the parameters are not correct, it
    returns None.'''

    # It is checked if parameters are correct

    if type(totalMatrix) != np.matrix or type(colsOfIteration) != np.ndarray or type(function) != np.ndarray or totalMatrix.shape[1] != len(function) or totalMatrix.shape[1] < len(colsOfIteration) or len(function) < len(colsOfIteration) or not isARationalMatrix(totalMatrix) or not isARationalArray(function):
        return None

    # The iteration base is calculated
    B = calculateBaseIteration(totalMatrix, colsOfIteration)

    # The iteration base is inverted
    invB = invertMatrix(B)

    # The value of the funcion is calcaulated for this iteration
    CB = calculateCB(colsOfIteration, function)

    # Product between the function and the inverted base is calculated
    dualSol = np.asarray(multMatrix(np.asmatrix(CB.T), invB))
  
    # The value of all variables is calculated
    dualSol = completeSolution(
        colsOfIteration, totalMatrix.shape[1],dualSol[0])

    # It returns dual solution

    return dualSol


def solveProblem(matrix, resources, signs, function, solutionOfDualProblem):
    '''This method receives a matrix with restricions in a numpy matrix with rational objects, a vector resources in a numpy array with rational objects, a list of strings
    with the signs and a string with the function, and boolean value that indicates if user wants dual solution(True) or not(False).
    It returns the solution of the problem and the final value of the function. It also prints every steps of the
    problem. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(matrix) != np.matrix or type(resources) != np.ndarray or type(signs) != list or type(function) != str or matrix.shape[0] != len(resources) or matrix.shape[0] != len(signs) or len(resources) != len(signs) or type(solutionOfDualProblem) != bool or not isStringList(signs) or not isARationalMatrix(matrix) or not isARationalArray(resources):
        return None
    
    isMin=1
    if "min" in function:
        isMin=-1
        
    # The data of the file is converted to standard form
    standardForm = convertToStandardForm(matrix, resources, signs, function)
    totalMatrix = standardForm[0]
    b = standardForm[1]
    functionVector = standardForm[3]

    # It is checked if it is neccessary to add columns of the identity matrix
    identityColumnsToAdd = addIdentityColumns(totalMatrix)

    # The artificial variables are added to function vector
    functionVector = addArtificialVariablesToFunctionVector(
        functionVector, identityColumnsToAdd.shape[1])

    # It is calculated which are artificial variables
    artificialVariables = calculateWhichAreArtificialVariables(
        functionVector, identityColumnsToAdd.shape[1])

    if identityColumnsToAdd.shape[1] > 0:
        # The identity columns are added to initial matrix
        totalMatrix = np.c_[totalMatrix, identityColumnsToAdd]

    # The columns of first iteration are calculated
    columnsOfIteration = variablesFirstIteration(totalMatrix)

    expl = "There is a unique solution."

    # The problem in standard form is showed
    print("Problem in standard form:")
    showStandarForm(totalMatrix, b, functionVector)

    print("Iteration#0")
    # The columns of iteration are showed
    showIterCol(columnsOfIteration)
    # The first iteration is solve

    results = solveIteration(
        totalMatrix, b, functionVector, columnsOfIteration)
    cont = 1
    print(results[3])

    # While there is another iteration the problem continues
    while (results[4] and results[4] != -1 and results[3] != None):

        iteration = cont
        cont += 1
        print("Iteration#" + str(iteration))
        # The columns of the iteration are calculated
        columnsOfIteration = calculateColumnsOfIteration(
            results[2], results[3], columnsOfIteration)
        # The columns of iteration are showed
        showIterCol(columnsOfIteration)
        # The iteration is solve
        results = solveIteration(
            totalMatrix, b, functionVector, columnsOfIteration)

    # If it is a min function, the sign changes
    results[1]=results[1]*isMin
    # The complete solution is calculated
    finalSolution = completeSolution(
        columnsOfIteration, totalMatrix.shape[1], results[0])
    # The problem has a large amount of solution
    if results[4] == -1:
        expl = "There are a large amount of solutions."
    # The problem is not bounded
    if results[3] == None:
        expl = "The problem is not bounded, because it can not exit any variable."
        print(expl)
        # It returns the solution,of the last iterarion, the final value of the
        # function and the explanation
        artificialVariables=[]
        if solutionOfDualProblem:
            # Dual solution is not possible in a not bounded problem
            return finalSolution, results[1], expl,None
        

    if artificialVariables:
        # It is checked if there are artificial variables with positive values
        thereIsNotASolution = checkValueOfArtificialVariables(
            artificialVariables, finalSolution)
        # If there are artificial variables with positive values, the problem
        # does not have solution
        if thereIsNotASolution:
            thereIsNotASolution = [x + 1 for x in thereIsNotASolution]
            expl = "The problem does not have solution because, as we can see, the artificial variable/s " + \
                str(thereIsNotASolution) + " has/have postive value."
            
            if solutionOfDualProblem:
            # Dual solution is not possible if not possible if problem does not have solution
                return finalSolution, results[1], expl,None
            
        
           

    # If the user wants the solution of the dual problem, it is showed
    if solutionOfDualProblem:
        # It returns the solution,the final value of the function, the explanation and the solution
        # of the dual problem

        return finalSolution, results[1], expl, calculateSolutionOfDualProblem(columnsOfIteration, functionVector, totalMatrix)

    # It returns the solution,the final value of the function and the
    # explanation
    
    return finalSolution, results[1], expl


def dualProblem(matrix, resources, sign, function):
    '''This method receives a matrix with restrictions in a numpy matrix of rational objetcs, a vector resources in a numpy array of rational objetcs, a list of strings
    with the signs and a string with the function. It returns the dual problem of the problem introduced by parameter(in the same types) .
    If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(matrix) != np.matrix or type(resources) != np.ndarray or type(sign) != list or type(function) != str or matrix.shape[0] != len(resources) or matrix.shape[0] != len(sign) or len(resources) != len(sign) or not isStringList(sign) or not isARationalMatrix(matrix) or not isARationalArray(resources):
        return None
    matrix=np.asarray(matrix)
    # The problem is changed to simetric maximization form
    for i in range(len(sign)):
        if sign[i] == ">=":
            
            # sign[i]=invertSign(sign[i])
            matrix[i] = [rational(j.numerator * -1, j.denominator)
                         for j in matrix[i]]
            resources[i] = rational(resources[i].numerator * -1, resources[i].denominator)
                            
            
            
        elif sign[i] == "=":
            matrix=np.vstack((matrix,matrix[i]))
            resources=np.append(resources,resources[i])
          
            
            matrix[i] = [rational(j.numerator * -1, j.denominator)
                         for j in matrix[i]]
            resources[i] = rational(resources[i].numerator * -1, resources[i].denominator)
           
           
    
    vectorFunction = convertFunctionToMax(function)

    # The problem dual is calculated
    matrix = matrix.T
    aux = ""
    resources = list(resources)
    for i in range(len(resources)):
        aux += str(resources[i]) + " "
    function = "min " + aux
    resources = vectorFunction.reshape(int(vectorFunction.shape[0]), 1)
    sign=[]
    for i in range(matrix.shape[0]):
        sign.append("<=")
    # It returns the dual problem
    return np.matrix(matrix), resources, sign, function


###########################################################################

###############################graphic solution###########################


def convertToPlotFunction(lineOfMatrix, sign, resource, x):
    ''' This method receives a numpy array with the coefficients of the variables in a restriction, the sign of
    the restriction in a string and the resource of the restriction. The method returns a function which represents
    the restriction passed by parameter. It could return a flaot number if the y value of the restriction is 0(the function is x=n).
    The method also returns a string with the  the restriction. Coefficients and resources must be rational objects.If the parameters are not correct,
    it returns None.'''

    # It is checked if parameters are correct
    if not isARationalArray(lineOfMatrix) or type(lineOfMatrix) != np.ndarray or type(sign) != str or type(x) != np.ndarray or type(resource) != rational or len(lineOfMatrix) != 2:
        return None

    
    # the y value of the function is checked
    if lineOfMatrix[1] != rational(0, 1):
        # It returns a function if y value is not 0
        return lambda x: (rationalToFloat(resource) + (-1 * rationalToFloat(lineOfMatrix[0])) * x) / rationalToFloat(lineOfMatrix[1]),str(lineOfMatrix[0]) + "x + " + str(lineOfMatrix[1]) + "y " + sign + str(resource)
    else:
        # It returns a float if y value is 0
        return rationalToFloat(resource) / rationalToFloat(lineOfMatrix[0]),str(lineOfMatrix[0]) + "x" + sign + str(resource)


def showFunction(function,x,label):
    '''This method receives a function or a number(if the function is x=n),a numpy linespace and a string with the label of the function
    in the plot. The method prepares the representation of the function. It is neccesary to use plt.show() after
    this method to plot the function.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if (type(function) != np.float64 and type(function) != np.int32 and type(function) != int and type(function) != float and type(function) != types.FunctionType) or type(label) != str or type(x) != np.ndarray:

        return None

    # if the function is a number
    if type(function) == np.float64 or type(function) == np.int32 or type(function) == int or type(function) == float:
        # function is prepared to plot
        plt.plot(function * np.ones_like(x), x, lw=1, label=label)
    # if the function is not a number
    else:
        # function is prepared to plot
        y = function
        plt.plot(x, y(x), lw=1, label=label)
        

def eliminateRepeatedPoints(seq):
    '''This method receives a list with points and returns the list without the repeated points.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(seq) != list:
        return None

    checked = []
    # It is checked which points are repeated
    for e in seq:
        if e not in checked:
            checked.append(e)
    # It returns the list without repeated points
    return checked


def eliminatePoints(list1, list2):
    '''This method receives two lists of points and returns the first list without the points that are in the second
    list. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(list1) != list or type(list2) != list:
        return None

    aux = []

    # The points of list 1 that are in the list 2, are deleted
    for i in list2:
        if i in list1:
            list1.remove(i)

    # It returns the list 1 without the points of the list 2
    return list1


def calculatePointOfSolution(functionVector, points, solution):
    '''This method receives the vector of the function in its maximization form, the extreme points with rational coordinates of the feasible
    region in a list, and the value of the function optimized. The method returns the points of the list which optimizes
    the function. If there is not any point which achives the introduced value, it returns None. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(functionVector) != np.ndarray or type(points) != list or type(solution) != rational or not isAListOfRationalPoints(points):
        return None

    # The points are splitted in x and y components
    ss, ts = np.hsplit(np.array(points), 2)
    pointSolution = []
    # It is calculated the points of the list which optimizes the function
    for i in range(len(ss)):
        if functionVector[0] * ss[i][0] + functionVector[1] * ts[i][0] == solution:
            pointSolution.append((ss[i][0], ts[i][0]))
    # It returns the points of solution
    if pointSolution:
        return pointSolution
    else:
        return None


def calculateSolution(function, points):
    '''This method receives the function of the problem in a string and the extreme points with rational coordinates of the feasible
    region in a list. The method returns the optimized value of the function, and which are the points that optimize
    it, in a list.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(function) != str or type(points) != list or not isAListOfRationalPoints(points):
        return None

    if not points:
        return None
    # The points are splitted in x and y components
    ss, ts = np.hsplit(np.array(points), 2)
    # If the function is a min function
    if "min" in function.lower():
        # Function is converted to max funcion and is casted to a numpy array
        # of rational objects
        functionVector = convertLineToRationalArray(function.strip("min "))

        # it is calculated the value of the function for the points
        z = [functionVector[0] * ss[i][0] + functionVector[1] * ts[i][0]
             for i in range(len(ss))]
        minimum = min(z)
        # It returns the minimum value of the function and the points which
        # optimizes it
        return minimum, calculatePointOfSolution(functionVector, points, minimum)
    # If the function is a max funcion
    elif "max" in function.lower():
        # Function is casted to a numpy array
        functionVector = convertLineToRationalArray(function.strip("max "))

        # it is calculated the value of the function for the points
        z = [functionVector[0] * ss[i][0] + functionVector[1] * ts[i][0]
             for i in range(len(ss))]
        maximum = max(z)
        # It returns the maximunm value of the function and the points which
        # optimize it
        return maximum, calculatePointOfSolution(functionVector, points, maximum)


def intersectionPoint(line1, line2, resource1, resource2):
    '''This method receives two numpy arrays which are rows in the coefficent matrix and the resources of these two
    rows, so the method receives two restrictions.The method returns the intersection points between the two functions
    or none if there is no point of intersection between them. Every element of the coefficent matrix and the resources
    must be a rational object.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if not isARationalArray(line1) or not isARationalArray(line2) or type(line1) != np.ndarray or type(line2) != np.ndarray or type(resource1) != rational or type(resource2) != rational or len(line1) != 2 or len(line2) != 2:
        return None

    # A new matrix with the coefficents is created
    A = np.matrix([line1, line2])
    # the simultaneous equations are solved, and the result is the
    # intersection point
    if determinant(A) != rational(0, 1):

        invA = invertMatrix(A)
        res = np.array([[resource1], [resource2]])

        point = multMatrix(invA, np.asmatrix(res))
        # returns the intersection point between the restrictions
        return np.asarray(point)[0][0], np.asarray(point)[1][0]

    # If there is not point of intersection
    else:
        # It returns None
        return None


def eliminateNegativePoints(points):
    '''This method receives a list of points, and eliminates the points which have a negative coordinate. The
    method returns the list with the points eliminated. The cordinates of the points must be rational objects.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(points) != list or not isAListOfRationalPoints(points):
        return None

    aux = []

    for i in points:
        # It is added all points which have positive coordinates
        if i[0].numerator >= 0 and i[0].denominator >= 0 and i[1].numerator >= 0 and i[1].denominator >= 0:
            aux.append(i)
    # It returns the list of points
    return aux


def calculateAllIntersectionPoints(matrix, resources):
    '''This method receives the coefficient matrix of the problem in a bi-dimensional numpy array and the resources vector in
    a numpy array. The method returns a list of intersection points between the restrictions and positive axis.
    The points with negative coordinates are eliminated.Every element of the coefficent matrix and the resources
    must be a rational object.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if not isARationalMatrix(matrix) or not isARationalArray(resources) or type(matrix) != np.ndarray or type(resources) != np.ndarray or matrix.shape[0] != len(resources):
        return None

    intersectionPoints = []

    for i in range(len(matrix)):
        # The intersection points between the restrictions and positives axis
        # are calculated
        intersectionPoints.append(intersectionPoint(matrix[i], np.array(
            [rational(0, 1), rational(1, 1)]), resources[i], rational(0, 1)))
        intersectionPoints.append(intersectionPoint(matrix[i], np.array(
            [rational(1, 1), rational(0, 1)]), resources[i], rational(0, 1)))
        for j in range(len(matrix)):
            if j > i:
                # The intersection points between the restrictions are
                # calculated
                intersectionPoints.append(intersectionPoint(
                    matrix[i], matrix[j], resources[i], resources[j]))

    # The none elements are deleted
    while None in intersectionPoints:
        intersectionPoints.remove(None)

    # The points with negative coordinates are eliminated.
    intersectionPoints = eliminateNegativePoints(intersectionPoints)
    # Repeated points are eliminated
    intersectionPoints = eliminateRepeatedPoints(intersectionPoints)

    # The point 0,0 is added
    if (rational(0, 1), rational(0, 1)) not in intersectionPoints:
        intersectionPoints.append((rational(0, 1), rational(0, 1)))
    # It returns the list of intersection points
    return list(intersectionPoints)


def calculateNotBoundedIntersectionPoints(matrix, resources, constX, constY):
    '''This method receives the coefficient matrix of the problem in a numpy array,the resources vector in
    a numpy array and the maximum x and y coordinates plotted. The method returns a list of intersection points between
    the restrictions and the imaginary extreme axis of the plot. The points with negative coordinates are eliminated.Every
    element of the coefficent matrix,the resources and the maximum x and y coordinates must be rational objects.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if not isARationalMatrix(matrix) or not isARationalArray(resources) or type(matrix) != np.ndarray or type(resources) != np.ndarray or type(constX) != rational or type(constY) != rational or matrix.shape[0] != len(resources):
        return None

    intersectionPoints = []

    # The intersection points between the functions and the hypothetic axis
    # are calculated
    for i in range(len(matrix)):
        intersectionPoints.append(intersectionPoint(matrix[i], np.array(
            [rational(0, 1), rational(1, 1)]), resources[i], constY))
        intersectionPoints.append(intersectionPoint(matrix[i], np.array(
            [rational(1, 1), rational(0, 1)]), resources[i], constX))

    # The none elements are deleted
    while None in intersectionPoints:
        intersectionPoints.remove(None)

    # The points with negative coordinates are eliminated.
    intersectionPoints = eliminateNegativePoints(intersectionPoints)
    # Repeated points are eliminated
    intersectionPoints = eliminateRepeatedPoints(intersectionPoints)

    # The top left point of the plot is added
    topPoint = intersectionPoint(np.array([rational(1, 1), rational(0, 1)]), np.array(
        [rational(0, 1), rational(1, 1)]), constX, constY)
    if topPoint not in intersectionPoints:
        intersectionPoints.append(topPoint)

    # It returns the list of intersection points
    return list(intersectionPoints)


def checkIfIsSolution(inecuation, solution, sign, resource):
    '''This method receives a restriction without sign and resource in a numpy array, a point in a tuple, the sign of the restriction
    in a string and the resource of the restriction. The method returns True if the points the restriction satisfies or False if
    not.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(inecuation) != np.ndarray or len(inecuation) != 2 or type(solution) != tuple or type(sign) != str or (type(resource) != np.int32 and type(resource) != np.float64 and type(resource) != float and type(resource) != int and type(resource) != rational):
        return None
    aux = 0
    # It is calculated the value of the restriction for this point
    for i in range(len(inecuation)):

        aux += inecuation[i] * solution[i]

    ops = {"<": operator.lt, "<=": operator.le, "=": operator.eq, ">=": operator.ge,
           ">": operator.gt}

    # It is checked if the value calculated before, satifies the restriction
    if ops[sign](aux, resource):
        # It returns True if the value calculated satifies the restriction
        return True
    # It returns False
    # if the value calculated satifies the restriction
    return False


def calculateFeasibleRegion(points, inecuations, resources, sign):
    '''This method receives a list of points, the restrictions of the problem without sign and resource in a numpy array, the resources of the restrictions in a numpy array,a list of
    strings with the signs of the restrictions. The method calculates which points belong to feasible region
    and returns these points in a list.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(points) != list or type(inecuations) != np.ndarray or type(resources) != np.ndarray or not isStringList(sign) or len(inecuations) != len(resources) or len(inecuations) != len(sign) or len(sign) != len(resources):
        return None

    pointsOfFeasibleRegion = []

    # It is checked if every point satifies all restrictions
    for i in range(len(points)):
        cont = 0

        for j in range(len(resources)):

            if checkIfIsSolution(inecuations[j], points[i], sign[j], resources[j]):

                cont += 1
        # If a point satifies all restrictions, it is a point of the feasible
        # region
        if cont == len(resources):
            pointsOfFeasibleRegion.append(points[i])
    # It returns the points of the feasible region
    return pointsOfFeasibleRegion


def calculateMaxScale(points):
    '''This method receives a list of points, and returns the maximum x and y coordinates. It calculates which should be
    the value of the scale to plot all points properly.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(points) != list or not isAListOfPoints(points):
        return None

    x = []
    y = []
    if not points:
        return 10, 10
    for i in points:
        # It is added every x coordinates
        x.append(i[0])
    # It is added every y coordinates
        y.append(i[1])
    # It returns the maximum x and y coordinates
    return max(x), max(y)


def calculateMinScale(points):
    '''This method receives a list of points, and returns the minimum x and y coordinates. It calculates which should be
    the value of the scale to plot all points properly.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(points) != list or not isAListOfPoints(points):
        return None

    x = []
    y = []
    if not points:
        return 10, 10
    for i in points:
        # It is added every x coordinates
        x.append(i[0])
    # It is added every y coordinates
        y.append(i[1])
    # It returns the minimum x and y coordinates
    return min(x), min(y)


def checkIfPointInFeasibleRegion(point, inecuations, resources, sign):
    '''This method receives a point, the restrictions of the problem without sign and resource in a numpy array,
    the resources of the restrictions in a numpy array and a list of strings with the signs of the restrictions. The method calculates
    if the point belongs to feasible region and returns True or False.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(point) != tuple or type(inecuations) != np.ndarray or type(resources) != np.ndarray or not isStringList(sign) or len(inecuations) != len(resources) or len(inecuations) != len(sign) or len(sign) != len(resources):
        return None

    cont = 0
    # It is checked if the point satifies all restrictions
    for j in range(len(resources)):

        if checkIfIsSolution(inecuations[j], point, sign[j], resources[j]):
            cont += 1
        # If the point satifies all restrictions, it is a point of the feasible
        # region
        if cont == len(resources):
            # It returns true if the point belongs to the feasible region
            return True
    # It returns false if the point does not belong to the feasible region
    return False


def calculateIntegerPoints(inecuations, resources, sign,minpoints,scale):
    '''This method receives the restrictions of the problem without sign and resource in a numpy array with rational elements,
    the resources of the restrictions in a numpy array with rational elements,a list of strings with the signs of the restrictions and
    the scale of the problem(the max and the min point that should be calculated) in two tuples.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if not isARationalMatrix(inecuations) or not isARationalArray(resources) or type(scale) != tuple or type(inecuations) != np.ndarray or type(resources) != np.ndarray or not isStringList(sign) or len(inecuations) != len(resources) or len(inecuations) != len(sign) or len(sign) != len(resources):
        return None

    # It is calculated which integer points are in the feasible region
    pairs = [(rational(int(s), 1), rational(int(t), 1)) for s in np.arange(rationalToFloat(minpoints[0]),rationalToFloat(scale[0]) + 5)
             for t in np.arange(rationalToFloat(minpoints[1]),rationalToFloat(scale[1]) + 5)
             if checkIfPointInFeasibleRegion((rational(int(s), 1), rational(int(t), 1)), inecuations, resources, sign)]

    # It returns the points
    return pairs


def handle_close(evt):
    '''This method is used to close the plot window properly.'''
    sys.exit()


def centre(points):
    '''This method receives a list of points with rational coordinates and return the centre of the polygon that is formed by these points. The coordinates
    of the points must be rational objects.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(points) != list or not isAListOfRationalPoints(points):
        return None

    # The centre is calculated
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / rational(len(points), 1),
                sum(y) / rational(len(points), 1))
    # It returns the centre of the points
    return centroid


def isThePoint(listPoints, value, M):
    '''This method receives a list of points, a value which is the distance of one of the points to the centre and
    the centre of the points. The method returns the point of the list whose distance is equal to the value or None,
    if there is no point that satisfies this distance to the centre. The coordinates of the points must be rational objects, but the coordinates
    of the centre must be float or int.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if not isAListOfRationalPoints(listPoints) or type(listPoints) != list or type(M) != tuple or (type(value) != np.float64 and type(value) != float and type(value) != int and type(value) != np.int32 and type(value) != rational):
        return None

    # It is checked which point satisfies the distance to the centre
    for i in listPoints:
        point = (rationalToFloat(i[0]), rationalToFloat(i[1]))

        if math.atan2(point[1] - M[1], point[0] - M[0]) == value:

            # It returns the point
            return i
    # It returns None if there is no point that satisfies the distance
    return None


def calculateOrder(listPoints):
    ''' This method receives a list of points, and return the points ordered clockwise in a list.The coordinates
    of the points must be rational objects.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(listPoints) != list or not isAListOfRationalPoints(listPoints):
        return None
    a = []
    # The centre of the points is calculated
    M = centre(listPoints)

    # The points and the centre are transformed in float values
    M = (rationalToFloat(M[0]), rationalToFloat(M[1]))
    listPoints2 = listPointsRationalToFloat(listPoints)

    # It is calculate the distance between every point and the centre
    for i in listPoints2:

        a.append(math.atan2(i[1] - M[1], i[0] - M[0]))

    orderedPoints = []

    a.sort(reverse=True)

    # The points are ordered by its distance to the centre
    orderedPoints = [isThePoint(listPoints, i, M) for i in a]

    # It returns the ordered points
    return orderedPoints


def pointIsInALine(point, line, resource):
    ''' This method receives a point, a restriction in a numpy array without sign and resource and the resource of
    the restriction. This method returns True if the point is over the line that represents the restricition, and False, if it the point
    is not over the line. If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if type(point) != tuple or type(line) != np.ndarray or (type(resource) != np.float64 and type(resource) != float and type(resource) != int and type(resource) != np.int32 and type(resource) != rational):
        return None

    if point[0] * line[0] + point[1] * line[1] == resource or point[0] == rational(0, 1) or point[1] == rational(0, 1):
        return True
    else:
        return False


def deleteLinePointsOfList(listPoints, matrix, resources):
    '''This method receives a list of points with rational coordinates, a numpy array of rational elements with the restrictions without sign and resources and
    a numpy array of rational elements with the resources. The method returns the same list that receives without the points that are
    over the restriction lines, when it is plotted.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if not isAListOfRationalPoints(listPoints) or type(matrix) != np.ndarray or not isARationalMatrix(matrix) or type(resources) != np.ndarray or not isARationalArray(resources):
        return None
    aux = []
    # The points that satisfies any restriction are calculated
    for i in range(len(resources)):
        for j in range(len(listPoints)):
            if pointIsInALine(listPoints[j], matrix[i], resources[i]):
                aux.append(listPoints[j])

    # The points are deleted
    for i in aux:
        if i in listPoints:
            listPoints.remove(i)
    # It returns the list with the points
    return listPoints


def showProblemSolution(matrix, resources, signs, function, save):
    '''These method receives a linear programming problem. It receives the restrictions without signs and resources, in a numpy matrix,
    the resources in a numpy array, a list of strings with the signs, a string with the function and the name of the file which the function
    will be saved in or False, if the figure will be showed in a matplot window. The method shows the graphic solution of the problem. Every
    element of the coefficent matrix and the resources must be rational objects.If the parameters are not correct, it returns None.'''

    # It is checked if parameters are correct
    if not isARationalMatrix(matrix) or not isARationalArray(resources) or matrix.shape[1] != 2 or type(matrix) != np.matrix or type(resources) != np.ndarray or type(signs) != list or type(function) != str or matrix.shape[0] != len(resources) or matrix.shape[0] != len(signs) or len(resources) != len(signs) or (save != False and type(save) != str) or not isStringList(signs):
        return None

    matrix = np.asarray(matrix)
    functions = []
    sign = []
    labels = []
    x = np.linspace(0, 10)

    # The figure is created
    fig, ax = plt.subplots(figsize=(8, 8))
    # When the figure is closed the program finish
    fig.canvas.mpl_connect('close_event', handle_close)
   
    matrix = np.asarray(matrix)

    # Functions are proccessed
    for i in range(len(resources)):

        aux = convertToPlotFunction(matrix[i], signs[i], resources[i], x)

        functions.append(aux[0])
        labels.append(aux[1])

    # All intesection points between the functions are calculated
    po = calculateAllIntersectionPoints(matrix, resources)
    # The extreme points of the feasible region are calculated
    verts = calculateFeasibleRegion(po, matrix, resources, signs)

    # The scale of the representation is calculated
    scale = calculateMaxScale(verts)
    minPoints= calculateMinScale(verts)
    # Functions are plotted

    x = np.linspace(
        0, int(max(rationalToFloat(scale[0]), rationalToFloat(scale[1]))) + 5)

    for i in range(len(functions)):
       
        showFunction(functions[i],x, labels[i])
       

    # Axes are ploted
    plt.plot(np.zeros_like(x), x, lw=1)
    plt.plot(x, np.zeros_like(x), lw=1)
    
    # The solution of the problem is calculated
    solution = calculateSolution(function, verts)
    
    # Problem does not have solution
    if solution == None or len(solution[1]) == 0:
        fig.suptitle('There is not feasible region for this restrictions',
                     fontsize=10, fontweight='bold')
    
    # Problem has solution
    else:
        
        # The integer points of the feasible region are calculated
        pon = calculateIntegerPoints(matrix, resources, signs,minPoints,scale)
        
        # The extreme points are removed, because they have already been
        # plotted
        pon = eliminatePoints(pon, verts)

        # The integer points are plotted
        pon2 = listPointsRationalToFloat(pon)
        ss, ts = np.hsplit(np.array(pon2), 2)
        plt.scatter(ss, ts, color="blue", cmap='jet',
                        label='Integer points', zorder=3)

        # The extreme points of the feasible region, are plotted
        verts2 = listPointsRationalToFloat(verts)
        rl, pl = np.hsplit(np.array(verts2), 2)
        ax.scatter(rl, pl, s=50, color="red",
                      label="Extreme feasible region points")

        # These points are labeled
        for i, txt in enumerate(verts):

            ax.annotate(" (" + str(txt[0]) + "," +
                        str(txt[1]) + ")", (rl[i], pl[i]))

        # Possible not bounded extreme points are added,if the feasible region
        # is not bounded
        newPo = calculateNotBoundedIntersectionPoints(
            matrix, resources, scale[0] + rational(5, 1), scale[1] + rational(5, 1))
        newPo = calculateFeasibleRegion(newPo, matrix, resources, signs)

        verts = verts + newPo

        # Points of the feasible region are ordered
        verts = calculateOrder(verts)

        verts = eliminateRepeatedPoints(verts)
        # The points which are in a line, are deleted
        pon = deleteLinePointsOfList(pon, matrix, resources)

        pon = listPointsRationalToFloat(pon)
        verts = listPointsRationalToFloat(verts)

        # Feasible region is created
        path = Path(verts2)
        
        # It is calculated if the problem is not bounded
        NotBounded = []
        if len(pon) > 0:
            NotBounded = path.contains_points(pon)

        # The problem is not bounded
        if list(NotBounded).count(False) > 0 and "max" in function.lower():
            fig.suptitle('The problem is not bounded, because inifinite points improve the solution',
                         fontsize=10, fontweight='bold')

        # The problem is bounded
        
        else:
            # If there is only one solution(a point)
            if len(solution[1]) == 1:
                #Solution is plotted
                ax.scatter(rationalToFloat(solution[1][0][0]), rationalToFloat(solution[1][0][
                           1]), s=80, color="green", label="Solution: (" + str(solution[1][0][0]) + "," + str(solution[1][0][1]) + ")")
                fig.suptitle("The point which optimizes the function is (" + str(solution[1][0][0]) + "," + str(solution[1][0][1]) + ") and the value of the function is " +
                             str(solution[0]), fontsize=10, fontweight='bold')

            # If there are many soutions(a line)
            elif len(solution[1]) == 2:

                #Solution is plotted
                solutionAux = listPointsRationalToFloat(solution[1])
                vx, vy = np.hsplit(np.array(solutionAux), 2)
                plt.plot(vx, vy, lw="4", color="green", label="Solution")
                fig.suptitle('The line between the points (' + str(solution[1][0][0]) + ',' + str(solution[1][0][1]) + ') and (' + str(solution[1][1][
                             0]) + ',' + str(solution[1][1][1]) + ') ,optimizes the function with a value of ' + str(solution[0]), fontsize=10, fontweight='bold')

        # The feasible region is plotted

        path = Path(verts)
        patch = patches.PathPatch(
            path, label='Feasible region', facecolor='yellow', lw=1)
        ax.add_patch(patch)

    # The axis are labeled
    
    ax.set_xlim(0, int(rationalToFloat(scale[0])) + 5)
    ax.set_ylim(0, int(rationalToFloat(scale[1])) + 5)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid()
   
    # The figure is saved in a file
    if save:
        plt.savefig(str(save).split(".")[0] + "Graphic.png")
    
    # The figure is showed
    else:
        plt.show()
    
##############################################################################
