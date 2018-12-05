import os
import shutil

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

os.system('rmdir /s /q build')
project_options = ' -H quetzal -A Systra'
make_options = ' -o source ../quetzal -s rst'
make_options += ' --full --separate' #  

os.system('sphinx-apidoc' + make_options + project_options)
os.system('sphinx-build -b html source build')
