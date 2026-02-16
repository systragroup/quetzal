## [FUTURE] (2026-02-16)
## WIP
* engine.fast_utils.py
* start adding new way to compute paths and store it in LOS using numba, polars, jagged array and pyarrow.

## Changes
* road_pathfinder: add the option to add zone_to_road in the expanded links. its not use in the wrapper as of today.
* road_pathfinder: add cost in the los too.

 ## bug fixes
 * common_links: fix bug when common_list is empty and we want to skip. skip

## [3.3.0] (2026-02-04)
## Changes
 * pt_pathfinder: paths_from_edges return path as tuple instead of list. usefull to drop_duplicates whithout converting to tuple (not effective on large dataframe)

 ## bug fixes
 * read_var (excel): fix a bug when we specify a period and there is no period in the excel file
 * common_trips: there was a bug in the function filling missing links making duplicates.
 * common_trips: fix list extend function and expose kwargs

## [3.2.0] (2026-01-27)
## Features
 * road_pathfinder : can now specify a Cost function for each segment ex: {'car': 'jam_time + length + toll', 'truck': 'jam_time + 2*toll'}

## changes
* road_pathfinder: results will be slightly different because there was some changes and bug fixes in the computation of phi (step_size):
    - base_flow was considered in the computation of relgap, not anymore.
    - relgap now is for the Cost, and not the time ( was sum(flow x time) on links, now its flow x cost on links)
    - for BFW, when finding Beta, base_flow was considered in 'flow' but not in 'auxiliary_flow'. removed in both.
    - for BFW, the derivative of jam_time around flow was used. With cost we now use sum(cost_seg * flow_seg)

## bug fixes
 * common_links creation was failling for some circular lines

## [3.1.3] (2026-01-22)
## bug fixes
* pt_pathfinder: dont use column cost. recompute the cost (for boarding links) 
    =>  time + headway/2 + boarding_time
    
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

