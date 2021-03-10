#!/usr/bin/env python
# vim: et ai si sw=2 sts=2 ts=2:
from re import split


with open('requirements.txt') as f:
    deps = [split(r'([<>]=?|==)', l.strip()) for l in f if l and l[0] != '#']
no_wheels = {
    'ipykernel': True,
    'pytables': True,
    'bitarray': True,
    'geopandas': True,
    'shapely': True,
    'contextily': True,
}
conda_deps = []
for dep in deps:
    depname = dep[0]
    if depname in no_wheels:
        condaname = no_wheels[depname]
        if condaname:
            condaname = depname
        conda_deps.append([condaname] + dep[1:])
print('conda install -y -c conda-forge ' + ' '.join(['"%s"' % ''.join(dep) for dep in conda_deps]))
