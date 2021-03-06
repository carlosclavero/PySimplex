#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# setup.py
# Description: setup
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
from distutils.core import setup
setup(
  name = 'PySimplex',
  packages = ['PySimplex'], 
  version = '0.1',
  description = 'This module contains tools to solve linear programming problems.',
  author = 'Carlos Clavero Mu?oz',
  author_email = 'c.clavero74@gmail.com',
  url = 'https://github.com/carlosclavero/PySimplex',
  keywords = ['simplex', 'linear', 'programming','rational','maths'], 
  classifiers = [],
)