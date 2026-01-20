## [3.1.2] (2026-01-20)
## changes
* road_pathfinder: moved default_brp and free_flow vdf from quetzal/engine/vdf.py to quetzal/engine/road_pathfinder.py
* road_pathfinder: delete the limit factory and the numba VDF function has we only support string now. (faster and simpler with polars)
* moved Api/MatrixRoadCaster Quenedi microservice to quetzal-backend/service.

## bug fixes
* road_pathfinder: Force VDF keys to be string (both vdf dict and dataframe col.) if. Mixed type in polar df is not support here. so cast all keys to string as it safer.
* os: call convertNotebook if jupyter nb convert failed
* When calling google map api for OD. set verify to True

## [3.1.1] (2026-01-15)
## changes
* add info to pypi. (github url, readme, etc)

## [3.1.0] (2026-01-15)
## changes
* add support for python 3.13
* Deploy to pypi as quetzal-transport with github action

