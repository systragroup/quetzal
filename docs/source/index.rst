********************
Quetzal Overview
********************

**Quetzal** is a Python package providing flexible models for transport planning and traffic forecasting.
It offers tools to help us carry out modelling tasks in the fields of data pre-processing and integrity checks, data-management and transport-modelling.

.. image:: https://raw.githubusercontent.com/systragroup/quetzal/master/docs/source/_images/paris.png
   :width: 800
   :alt: paris.png

Free software
***************
Quetzal is free; you can redistribute it and/or modify it under the terms of the CeCILL-B License. 
We welcome contributions, join us on GitHub: https://github.com/systragroup/quetzal. 

History
**********
Quetzal was born in 2016. The first public release was in 2018.

***************
Documentation
***************
Quetzal is based on a main object called StepModel. 
It stores the data as attributes and makes the modelling procedures available as methods.

In terms of data storage, the models designed with Quetzal relies on light data formats.
They can be stored as zipped HDF5 which is hyper fast or as JSONs, which are human readable and easy to track for changes.
They can be stored, shared and versioned with Github which makes collaborative projects easier.

.. toctree::
   Read about stepmodel attributes <quetzal.model.stepmodel>

In the documentation, for every method, the required attributes and the expected products are listed.
The two main groups of methods are the preparation or pre-processing methods and the step methods.

Data Preparation
****************

Quetzal was designed around the use of open data and provides implementation of state-of-the-art
algorithms to process poorly georeferenced GTFS and Open Street Map data. It takes advantage of graph
analysis to build a consistent network-conscious timetable from a stop sequence-based,
network-naïve timetable and an extraneous road network.


.. toctree::
   Documentation of the preparation functions <quetzal.model.preparationmodel>
   Documentation of the GFTS import functions <quetzal.io.gtfs_reader.importer>
   Documentation of the integrity functions <quetzal.model.integritymodel>

To save computer time, it is often worth reducing the size of the input data. Quetzal makes extensive use of geomatics algorithms to build: 

   * aggregated zonings : **preparation_clusterize_zones**
   * station clusters : **preparation_clusterize_nodes**

Quetzal embeds automated consistency checks on the timetables and the road network that spot the most frequent error sources such as :

   * non-convex road networks : **integrity_fix_road_network**
   * incomplete public transport stop sequences : **integrity_fix_sequences**
   * namespace collisions (names shared between independent objects) : **integrity_fix_collision**
   * fix circular lines : **integrity_fix_circular_lines**
   * fix nodeset consistency (public transport and roads): **integrity_fix_nodeset_consistency** and **integrity_fix_road_nodeset_consistency**

Network casting : Public transport timetables and road networks are key inputs to many studies, but often come from independent sources.
It is necessary to link them in order to model their physical interactions such as the impact of private 
car traffic on the commercial speed of road-based public modes.
Function : **preparation_cast_network**

In order to have a functioning model, before the application of stepmodel functions and after data preparation,
the following functions will have to be used :

   * Prepare connectors : **preparation_ntlegs**
   * allow public transport connections : **preparation_footpaths**


Step Model
**********

Quetzal provides a complete algorithm suite to help design a standard four-stage model or more customized ones.
The downstream articulation between modal split and assignment is flexible since both use a utility-based nested Logit.
The modal split and the assignment (volume split between the many PT options) can be performed with a single nested logit model.
This approach helps enhance the consistency between the modal split and the assignment steps.

.. toctree::
   Documentation of the stepmodel functions <quetzal.model.transportmodel>

**Generation**

There is no real need for a specialized library to achieve the generation step and the Python language offers many options to do so.

**Distribution**

Once the productions and attractions are estimated, Quetzal offers a doubly constrained distribution based on
an impedance matrix that can be provided as an input or generated from a gravitational model : **step_distribution**
The growth of the OD matrix can also be performed with a Fratar.

**Pathfinders**

The goal of the public transport and car LOS estimation is to gather for all origin-destination pairs a collection of paths
and compute a voyager utility for each of them.

The levels of service (LOS) of the public transports (time, transfers, fares, …etc.) are estimated with graph algorithms.
Those algorithms can be parametrized through the function **step_pt_pathfinder**, based on frequency.

   * The fastest option is to compute only the best path
   * The most accurate one is to alternatively break sections of the Public Transport graph and compute the best paths in the broken graphs. 

The car travel time can be estimated with **step_road_pathfinder**.
This function performs road assigment by performing the pathfinder, because travel time depends on congestion.

   * a simple use of the Dijkstra Algorithm if the congestion is of no concern. 
   * Frank Wolfe Algorithm which leads to a car assignment in Wardrop equilibrium

There is the possibility to use Park and Ride with the dedicaded pathdfinder **step_pr_pathfinder**

Pathfinders based on timetables depend on other classes :

.. toctree::
   Read about connection scan models <quetzal.model.connectionscanmodel>
   Read about time expanded models <quetzal.model.timeexpandedmodel>

**Logit**

Use the following functions to perform the nested logit :

   * concatenate pt_los and car_los in an unique table self.los
   * Builds the necessary tables to perform the following functions : **preparation_logit**
   * Compute utilities per mode per segment based on logit parameters : **analysis_mode_utility**
   * Performs the nested logit, that is to say compute the probabilities per segment : **step_logit**

**Assignment**

Assignment of private transport is done with the function **step_pr_pathfinder**.
For public transport assignment, we will use :

   * Compute volume in the level of services table : **compute_los_volume**
   * Compute the volumes on the links of the public transport network,
     and the boardings and alightings on the nodes of the PT network : **step_assignment** 

  
Indices and tables
==================

* :ref:`genindex`

