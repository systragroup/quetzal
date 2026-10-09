import os
import warnings
import sys

if sys.version_info.major != 3 or sys.version_info.minor < 12:
    warnings.warn('Quetzal was updated to python 3.12. Please refer to README to update.')

if bool(os.environ.get('AWS_EXECUTION_ENV')):  # Cloud execution
    os.environ['TQDM_DISABLE'] = '1'
