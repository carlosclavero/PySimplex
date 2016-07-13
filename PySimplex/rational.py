#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# rational.py
# Description: implementation of rational numbers
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


class rational:
    '''This class implements rational numbers.They are formed by a numerator and a denominator. Dividers
    of numerator and denominator are also included in two list.'''

    def isPrime(self, n):
        '''This method receives a number and returns True if it is prime, or False if it is not.'''

        if n < 2:
            return False
        for i in range(2, n):
            if n % i == 0:
                return False
        return True

    def primeDividers(self, n):
        ''' This method receives a number and calculates all of its prime dividers. It returns the dividers in
        a list.'''
        div = []
        # The number is considered as positive
        n = abs(n)
        # If the number is prime, it is only divisible by itself
        if self.isPrime(n):
            div.append(n)
            return div

        # Prime dividers are calculated
        for i in range(2, n):
            if n % i == 0 and self.isPrime(i):
                div.append(i)
        # It returns the dividers
        return div

    # Rational object is intialized
    def __init__(self, numerator, denominator):
        # If numerator and denominator are negative, they both change to
        # positive
        if denominator < 0 and numerator < 0:
            numerator = numerator * -1
            denominator = denominator * -1

        elif numerator == 0 and denominator != 0:
            denominator = 1

        self.numerator = numerator
        self.denominator = denominator

        self.div_num = self.primeDividers(numerator)
        self.div_den = self.primeDividers(denominator)

    # Str is defined for rational objects
    def __str__(self):

        if self.denominator == 1:
            return str(self.numerator)

        else:
            return str(self.numerator) + "/" + str(self.denominator)

    def simplify(self):
        '''This method receives a rational number, and returns it simplified.'''

        # If numerator is equal to denominator, the result is 1
        if self.numerator == self.denominator:
            return rational(1, 1)

        common_div = []
        # Common dividers of numerator and denominator are calculated
        [[common_div.append(x)
          for x in self.div_num for y in self.div_den if x == y]]

        # Number is simplified
        for i in common_div:
            while self.numerator % i == 0 and self.denominator % i == 0:
                self.numerator = (int(self.numerator / i))
                self.denominator = (int(self.denominator / i))

        # It returns the number
        return self

    # Common operators are overloaded
    def __add__(self, other):
        return rational(self.numerator * other.denominator + self.denominator * other.numerator, self.denominator * other.denominator).simplify()

    def __radd__(self, other):
        other = rational(other, 1)
        return self + other

    def __sub__(self, other):
        return rational(self.numerator * other.denominator - self.denominator * other.numerator, self.denominator * other.denominator).simplify()

    def __rsub__(self, other):
        other = rational(other, 1)
        return other - self

    def __mul__(self, other):
        return rational(self.numerator * other.numerator, self.denominator * other.denominator).simplify()

    def __rmul__(self, other):
        other = rational(other, 1)
        return self * other

    def __truediv__(self, other):
        return rational(self.numerator * other.denominator, self.denominator * other.numerator).simplify()

    def __rtruediv__(self, other):
        other = rational(other, 1)
        return other / self

    def __eq__(self, other):
        if type(other) == int or type(other) == np.int32:
            other = rational(other, 1)

        if other == None or type(other) == float:
            return False

        return self.numerator * other.denominator == self.denominator * other.numerator

    def __ne__(self, other):
        if type(other) == int or type(other) == np.int32:
            other = rational(other, 1)
        if other == None:
            return True

        return self.numerator * other.denominator != self.denominator * other.numerator

    def __lt__(self, other):
        if type(other) == int or type(other) == np.int32:
            other = rational(other, 1)

        numA = self.numerator
        denA = self.denominator
        numB = other.numerator
        denB = other.denominator

        if numA > 0 and denA < 0:
            numA = self.numerator * -1
            denA = self.denominator * -1

        if numB > 0 and denB < 0:
            numB = other.numerator * -1
            denB = other.denominator * -1

        return numA * denB < denA * numB

    def __le__(self, other):
        if type(other) == int or type(other) == np.int32:
            other = rational(other, 1)

        self.simplify()
        other.simplify()

        numA = self.numerator
        denA = self.denominator
        numB = other.numerator
        denB = other.denominator

        if numA > 0 and denA < 0:
            numA = self.numerator * -1
            denA = self.denominator * -1

        if numB > 0 and denB < 0:
            numB = other.numerator * -1
            denB = other.denominator * -1

        return numA * denB <= denA * numB

    def __gt__(self, other):
        if type(other) == int or type(other) == np.int32:
            other = rational(other, 1)

        numA = self.numerator
        denA = self.denominator
        numB = other.numerator
        denB = other.denominator

        if numA > 0 and denA < 0:
            numA = self.numerator * -1
            denA = self.denominator * -1

        if numB > 0 and denB < 0:
            numB = other.numerator * -1
            denB = other.denominator * -1

        return numA * denB > denA * numB

    def __ge__(self, other):
        if type(other) == int or type(other) == np.int32:
            other = rational(other, 1)

        self.simplify()
        other.simplify()

        numA = self.numerator
        denA = self.denominator
        numB = other.numerator
        denB = other.denominator

        if numA > 0 and denA < 0:
            numA = self.numerator * -1
            denA = self.denominator * -1

        if numB > 0 and denB < 0:
            numB = other.numerator * -1
            denB = other.denominator * -1

        return numA * denB >= denA * numB
