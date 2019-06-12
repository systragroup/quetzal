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

Use `windows-install`

Alternatively, you can do these steps:
- Virtual environment: `conda create -n quetzal; conda activate quetzal`
- Then use `python get_conda_install_args.py` from the `quetzal` directory.
- Execute the given conda command to install required dependencies from conda repositories.
- Then `pip install -e .`
