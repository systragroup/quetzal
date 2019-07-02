#!/usr/bin/env python
# vim: et ai si sw=2 sts=2 ts=2:
from re import split


with open('requirements.txt') as f:
  deps = [split(r'([<>]=?|==)', l.strip()) for l in f if l and l[0] != '#']
no_wheels = {
  'bitarray': True,
  'geopandas': 'fiona',
  'shapely': True,
}
conda_deps = []
for dep in deps:
  depname = dep[0]
  if depname in no_wheels:
    condaname = no_wheels[depname]
    if condaname == True:
      condaname = depname
    conda_deps.append([condaname] + dep[1:])
print('conda install -y ' + ' '.join(['"%s"' % ''.join(dep) for dep in conda_deps]))
