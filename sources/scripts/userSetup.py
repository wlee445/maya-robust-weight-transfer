import os
import sys
import traceback
from importlib import reload
try:

    import pymel.core as pm
    import pymel.core.datatypes as dt
    import pymel.core.nodetypes as nt

except Exception as e:
    traceback.print_exc()
    
import maya.cmds as mc
import maya.OpenMaya as om
import maya.utils
maya.utils.executeDeferred(exec("print('maya-robust-weight-transfer userSetup.py')"))