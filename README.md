# quetzal
## What is it?
**quetzal** is a Python package providing flexible models for transport planning and traffic forecasting.
## Copyright 
(c) SYSTRA
## License
[CeCILL-B](LICENSE.md)
## Documentation
The official documentation is hosted on https://systragroup.github.io/quetzal
## Backward compatibility
In order to improve the ergonomics, the code may be re-factored and a few method calls may be re-designed. As a consequence, the backward compatibility of the library is not guaranteed. Therefore, the version of quetzal used for a project should be specified in its requirements. 
## Installation from sources
It is preferred to first create and use a virtual environment.
### For Linux
Virtual environment: `virtualenv .venv; source .venv/bin/activate` or any equivalent command.

```bash
pip install -e .
```
### For Windows
`Conda` is supposed to be installed.
#### To create quetzal_env automatically and install quetzal
```bash
(base) C:users\you\path\to\quetzal> windows-install.bat
```
press enter to accept default environment name
#### If you are facing SSL issues
```bash
(base) pip config set global.trusted-host "pypi.org files.pythonhosted.org"
(base) C:users\you\path\to\quetzal> windows-install.bat
```
press enter to accept default environment name

security warning: the host is added to pip.ini  
#### To install quetzal in active environment
```bash
(base) C:users\you\path\to\quetzal> windows-install-here.bat
```




