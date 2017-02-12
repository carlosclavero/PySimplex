#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SimplexSolver.py
# Description: Solver for linear programming problems
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
import argparse
import sys
import os
import matplotlib.pyplot as plt
import io
from PySimplex import Simplex

#add arguments to arg parse to use input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input',required=True, type=argparse.FileType('r'),help="File which contains the problem")
parser.add_argument('--output', type=argparse.FileType('w'),help="File where the solution is written")
parser.add_argument('--dual',action="store_true",help="Show the solution of dual problem")
parser.add_argument('--graphic',action="store_true",help="Show the graphic solution dual problem(only for two variables)")
parser.add_argument('--expl',action="store_true",help="Show the explanation and the development of the problem")
args = parser.parse_args()
save=False

# Input file is proccessed
fileProcess = Simplex.proccessFile(args.input)
# It is checked if there are any decimal in the input file
if fileProcess == None:
    print("Please introduce the problem in the correct format. Only fractions are allowed.")

else:

    
    if args.output:
        #The name of the output file is saved
        save=args.output.name
    #If user does not want explanation,console output is redirected to null 
    if not args.expl:
        f = open(os.devnull, 'w')
        sys.stdout = f
    
    # If user introduces output file and wants explanation, console output is redirected to the file    
    if args.output and args.expl:

        sys.stdout = args.output
    
    # Solution of the problem is calculated
    solution=Simplex.solveProblem(fileProcess[0],fileProcess[1],fileProcess[2],fileProcess[3],args.dual)
    
    # If user introduces output file and does not want explanation, console output is redirected to the file    
    if not args.expl and args.output:
        sys.stdout = args.output
    
    # If user does not want explanation and does not introduce output file, only solution is showed in the console
    elif not args.output:
       io.stdout = sys.stdout = sys.__stdout__
       
    # Solution is showed
    print("SOLUTION: x* ="+Simplex.printMatrix(np.asmatrix(solution[0]).T)+" ,  z* = "+str((np.asarray(solution[1])[0][0]))+" ."+solution[2])
    
    
    # Dual solution is showed
    if args.dual:
        if solution[3] == None:
            print("Dual problem solution is not available for this kind of primal problem.")
        else:
            print("DUAL SOLUTION: x* ="+ Simplex.printMatrix(np.asmatrix(solution[3]).T))
       
    if args.graphic:
        # Graphic solution is calculated
        fileProcess= Simplex.proccessFile(args.input)
        # It is checked if the number of variables is correct
        if fileProcess[0].shape[1]>2:
            print("Graphic solution is only available for problems with two variables.")
        else:
            Simplex.showProblemSolution(fileProcess[0],fileProcess[1],fileProcess[2],fileProcess[3],save)
    
    