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
One should choose between Virtualenv and Pipenv or use Anaconda 3.
#### Virtualenv
Virtual environment: `virtualenv .venv -p python3.8; source .venv/bin/activate` or any equivalent command.

```bash
pip install -e .
```

#### Pipenv
```bash
pipenv install
```

#### Anaconda
In order to use python notebook, Anaconda 3 + Python 3.8 must be installed.
Then create + activate quetzal environment:
```bash
conda init
conda create -n quetzal_env -y python=3.8
conda activate quetzal_env
pip install -e . -r requirements.txt
python -m ipykernel install --user --name=quetzal_env
```

... Or use the `linus-install.sh` script.

### For Windows
`Anaconda 3 + Python 3.8` is supposed to be installed.

#### PIP with Wheels 
```bash
(base) C:users\you\path\to\quetzal>windows-install-whl.bat
```
press enter to accept default environment name
#### PIP and Anaconda 
To create quetzal_env automatically and install quetzal 
```bash
(base) C:users\you\path\to\quetzal> windows-install.bat
```
press enter to accept default environment name
#### If you are facing SSL issues
```bash
(base) pip config set global.trusted-host "pypi.org files.pythonhosted.org"
(base) C:users\you\path\to\quetzal> windows-install.bat
```
security warning: the host is added to pip.ini

#### If you are facing DLL or dependencies issues
Anaconda and Pip do not get along well, your Anaconda install may have been corrupted at some point.
- Remove your envs
- Uninstall Anaconda
- Delete your Python and Anaconda folders (users\you\Anaconda3, users\you\Appdata\Roaming\Python, ...etc)
- Install Anaconda 
