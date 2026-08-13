
# quetzal
## What is it?
**quetzal** is a Python package providing flexible models for transport planning and traffic forecasting. Quetzal is highly optimized to run fast on big cities.
## Copyright
(c) SYSTRA
## License
[CeCILL-B](LICENSE.md)
## Documentation
The official documentation is hosted on https://systragroup.github.io/quetzal
## Backward compatibility
In order to improve the ergonomics, the code may be re-factored and a few method calls may be re-designed. As a consequence, the backward compatibility of the library is not guaranteed. Therefore, the version of quetzal used for a project should be specified in its requirements.

# Installation

https://pypi.org/project/quetzal-transport/

```bash
pip install quetzal-transport
```

# Installation from sources
## For Linux

### uv
1) Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2) install dependencies (this will create a new virtualenv at `.venv/` and fetch Python 3.12 if needed)
```bash
uv sync
```
3) activate the env
```bash
source .venv/bin/activate
```
4) (optional) add the env to ipykernel (to use in jupyter)
```bash
python -m ipykernel install --user --name=quetzal_env
```

### poetry (alternative)
1) You may need to set the default (or local) python version in the project
```bash
pyenv local 3.12
```
2) install dependancies (this will create a new virtualenv)
```bash
poetry install
```
3) activate the env
```bash
poetry shell
```
4) (optional) add the env to ipykernel (to use in jupyter)
```bash
python -m ipykernel install --user --name=quetzal_env
```



## For Windows
#### uv (recommended)
No prior installation of Python or uv is required — the batch file handles both.
Open a command prompt and run the windows-install batch file
```bat
C:\users\you\path\to\quetzal> windows-install.bat
```
press enter to accept the default kernel name or enter a custom name
#### poetry (alternative)
Requires `Anaconda 3` with Python 3.12. Run the poetry batch file instead:
```bat
C:\users\you\path\to\quetzal> windows-install-poetry.bat
```
press enter to accept the default kernel name or enter a custom name
#### If you are facing DLL or dependencies issues
Anaconda and Pip do not get along well, your Anaconda install may have been corrupted at some point.
- Remove your envs
- Uninstall Anaconda
- Delete your Python and Anaconda folders (users\you\Anaconda3, users\you\Appdata\Roaming\Python, ...etc)
- Install Anaconda
#### If you are facing SSL issues
```bat
pip config set global.trusted-host "pypi.org files.pythonhosted.org"
C:\users\you\path\to\quetzal> windows-install.bat
```
security warning: the host is added to pip.ini
 
# Tests
to run unittest:

```bash
uv run python -W ignore -m unittest discover
```
Or with poetry:
```bash
poetry run python -W ignore -m unittest discover
```

# Deploying

1) change the version in **pyproject.toml**

```toml
[project]
name = "quetzal-transport"
version = "3.1.1"
```

2) edit **CHANGELOG.md** with the changes
```md
## [3.1.1] (2026-01-15)
## changes
* some changes
```


2) create a tag matching the version **(starting with v)**

```bash
git tag -a v3.1.1 -m 'description'
```

3) push the tag

```bash
git push origin v3.1.1
```

that's it. A Github action will 
* build
* create a release 
* update the package on pipy.