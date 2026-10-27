import sys
import os
from PyQt6 import QtCore

# Force software rendering to avoid Direct3D availability issues on clean PCs
os.environ.setdefault("QT_QPA_PLATFORM", "windows")
QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
