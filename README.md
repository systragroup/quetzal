
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

# Installation from sources
## For Linux
One should choose between 
- Poetry (recommended)
- Virtualenv 
- Anaconda

### poetry
1) May need to set the default (or local) python version in the project
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
4) add the env to ipykernel (to use in jupyter)
```bash
python -m ipykernel install --user --name=quetzal_env
```

### Virtualenv
Virtual environment: `virtualenv .venv -p python3.12; source .venv/bin/activate` or any equivalent command.

```bash
pip install -e .
```

#### Anaconda
In order to use python notebook, Anaconda 3 + Python 3.12 must be installed.
Then create + activate quetzal environment:
```bash
conda init
conda create -n quetzal_env -y python=3.12
conda activate quetzal_env
pip install -e . -r requirements_win.txt
python -m ipykernel install --user --name=quetzal_env
```



## For Windows
`Anaconda 3 + Python 3.12` is supposed to be installed
#### PIP and Anaconda (recommended)
To create quetzal_env automatically and install quetzal, open anaconda prompt and
run windows-install batch file
```bash
(base) C:users\you\path\to\quetzal> windows-install.bat
```
press enter to accept default environment name or enter a custom name 
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


## migration to 3.12
* pandas append was remove: 
```python
# before
sm.volumes = sm.volumes.append(vol)
#now
sm.volumes = pd.concat([sm.volumes, vol])
#or
sm.volumes = pd.concat([sm.volumes, pd.DataFrame(vol)])
```
filtering index with set was remove:
```python
# before
sm.volumes = sm.volumes.loc[od_set]
#now
sm.volumes = sm.volumes.loc[list(od_set)]
```

* shapely
```python
# before
hull = zones.unary_union.convex_hull
# now
hull = zones.union_all().convex_hull
```
* scikitlearn
```python
# add n_init='auto'
KMeans(n_clusters=num_zones,random_state=0,n_init='auto')
```
