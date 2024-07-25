import warnings
import sys
if sys.version_info.major != 3 or sys.version_info.minor < 12:
    warnings.warn(f'Quetzal was updated to python 3.12. Please refer to README and update. {sys.version}')