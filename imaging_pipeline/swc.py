#  Copyright (c) 2019. Martin Haesemeyer. All rights reserved.
#
#  Licensced under the MIT license. See LICENSE


"""
Module with simple parser of swc tracing files
"""

import numpy as np


class SWC:
    # TODO: init should take swc file name and optionally cmtk transformation file as parameters. swc information will
    #  be loaded from file and structured in matrix. Quick check for unique ID on first column. Matrix storage will be
    #  used for simple "find parent" (find row where val[0]==parent_id) and "find all children"
    #  (find rows where val[-1]==child_id) operations.
    #  Convenience function returns all cell body coordinates and all segment lines for passing to scatter and plot.
    #  If a cmtk transform file is passed during init, attempt to transform the swc x,y,z coordinates before populating
    #  the storage matrix.
    pass
