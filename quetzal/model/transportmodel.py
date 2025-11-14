import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional
from quetzal.analysis import analysis
from quetzal.engine import engine, nested_logit
from quetzal.engine.park_and_ride_pathfinder import ParkRidePathFinder
from quetzal.engine.pathfinder import PublicPathFinder
from quetzal.engine.road_pathfinder import (
    init_network,
    init_volumes,
    aon_roadpathfinder,
    msa_roadpathfinder,
    expanded_roadpathfinder,
    get_car_los_time,
)
from quetzal.engine.msa_trackers.links_tracker import LinksTracker
from quetzal.engine.sampling import sample_od
from quetzal.model import model, optimalmodel, parkridemodel
from syspy.assignment import raw as raw_assignment
from syspy.assignment.raw import fast_assign as assign
from tqdm import tqdm


def read_hdf(filepath):
    """Read HDF format

    Parameters
    ----------
    filepath : string


    Returns
    -------
    _type_
        _description_
    """
    m = TransportModel()
    m.read_hdf(filepath)
    return m


def read_json(folder, **kwargs):
    """Read json format

    Parameters
    ----------
    folder : string


    Returns
    -------
    _type_
        _description_
    """
    m = TransportModel()
    m.read_json(folder, **kwargs)
    return m


track_args = model.track_args
log = model.log


class TransportModel(optimalmodel.OptimalModel, parkridemodel.ParkRideModel):
    @track_args
    def step_distribution(self, segmented=False, deterrence_matrix=None, **od_volume_from_zones_kwargs):
        """Function performing distribution of flows with doubly constrained algorithm,
        based on an impedance/deterrence matrix inversely proportional to the accessibility between two zones.

        Requires
        ----------
        self.zones


        Parameters
        ----------
        segmented : bool, optional
            if True: all parameters must be given in dict {segment: param}, by default False
        deterrence_matrix : unstacked dataframe, optional
            an OD unstaked dataframe representing the disincentive to
            travel as distance/time/cost increases, by default None
        od_volume_from_zones_kwargs :
            if the friction matrix is not
            provided, it will be automatically computed using a gravity distribution with
                - power: int, thee gravity exponent
                - intrazonal: (bool) if False set the intrazonal distance to 0, else compute a characteristic distance

        Builds
        ----------
        self.volumes :

        """
        if segmented:
            self.volumes = pd.DataFrame(columns=['origin', 'destination'])
            kwargs = od_volume_from_zones_kwargs.copy()
            if 'deterrence_matrix' not in kwargs.keys():
                kwargs['deterrence_matrix'] = deterrence_matrix if deterrence_matrix is not None else {}
            if 'power' not in kwargs.keys():
                kwargs['power'] = {}
            if 'intrazonal' not in kwargs.keys():
                kwargs['intrazonal'] = {}

            for segment in self.segments:
                print(segment)
                cols = ['geometry', (segment, 'emission'), (segment, 'attraction')]
                if 'area' in self.zones:
                    cols += ['area']
                segment_zones = self.zones[cols].rename(
                    columns={(segment, 'emission'): 'emission', (segment, 'attraction'): 'attraction'}
                )
                segment_volumes = engine.od_volume_from_zones(
                    segment_zones,
                    deterrence_matrix=kwargs['deterrence_matrix'].get(segment, None),
                    coordinates_unit=self.coordinates_unit,
                    power=kwargs['power'].get(segment, 2),
                    intrazonal=kwargs['intrazonal'].get(segment, False),
                )
                segment_volumes.rename(columns={'volume': segment}, inplace=True)

                self.volumes = self.volumes.merge(segment_volumes, on=['origin', 'destination'], how='outer')
            self.volumes['all'] = self.volumes[self.segments].T.sum()

        else:
            self.volumes = engine.od_volume_from_zones(
                self.zones, deterrence_matrix, coordinates_unit=self.coordinates_unit, **od_volume_from_zones_kwargs
            )

    def sample_volumes(
        self, bidimentional_sampling=False, fit_sums=True, sample_weight=1, sample_size=None, inplace=True, **kwargs
    ):
        od_volumes = self.volumes.set_index(['origin', 'destination'])
        # remove segments without volumes.
        columns = od_volumes.columns[od_volumes.sum(axis=0) > 0]

        series = {}
        for c in columns:
            try:
                sw = sample_weight[c]
            except TypeError:  # it is not a dict but a value
                sw = sample_weight
            try:
                ss = sample_size[c]
            except TypeError:
                ss = sample_size

            series[c] = sample_od(
                od_volumes[c],
                bidimentional_sampling=bidimentional_sampling,
                fit_sums=fit_sums,
                sample_weight=sw,
                sample_size=ss,
                **kwargs,
            )

        volumes = pd.DataFrame(series).reset_index()
        if inplace:
            self.volumes = volumes
        else:
            return volumes

    @track_args
    def step_road_pathfinder(
        self,
        method: Literal['bfw', 'fw', 'msa', 'aon'] = 'bfw',
        segments: List[str] = [],
        time_column: str = 'time',
        access_time: str = 'time',
        od_set: Optional[Dict] = None,
        tracker_plugin: LinksTracker = LinksTracker(),
        turn_penalties: Optional[Dict[str, List[str]]] = None,
        ntleg_penalty: float = 10e9,
        num_cores: int = 1,
        return_car_los=True,
        **kwargs,
    ) -> None:
        """Performs road assignment with or without capacity constraint, depending on the method used

        Requires
        ----------
        self.road_links
        self.zone_to_road
        self.volumes


        Parameters
        ----------
        method : ['bfw'|'fw'|'msa'|'aon'], optional
            Which method to use for pathfinder. Options are:

            'bfw'   -- Bi-conjugate Frank-Wolfe

            'fw'    -- Frank-Wolfe

            'msa'   -- Mean succesive average

            'aon'   -- all or nothing : shortest path pathfinder

        segments: list of segments in volumes to assign

        time_column: string, optional, defaut time
            name of the links free_flow time column in road_links

        access_time : string, optional, default 'time'
            column for time in zone_to_road for access time

        od_set : dict, optional, default None
            set of od to use - may be used to reduce computation time
            for example, the od_set is the set of od for which there is a volume in self.volumes

        ntleg_penalty : float, optional, default 1e9
            ntleg penality for access time
        turn_penalties : dict, optional, default None
            dictionary of turn penalties for the road links
            ex: {'rlink_0', ['rlink_4']}


        num_cores : integer, optional, default 1
            for parallelization

        tracker_plugin: LinksTracker() optional
            track OD using a selected links

        return_car_los:
            compute and return self.car_los

        **kwargs :  see msa_roadpathfinder()
            vdf={'default_bpr': default_bpr, 'free_flow': free_flow},
            method='bfw',
            maxiters=10,
            tolerance=0.01,
            log=False,

            turn_penality for expanded.

        Builds
        ----------
        self.car_los :
            create tables of car levels of services

        self.road_links :
            add columns flow (volume on the road links) and jam_time (time of the link with congestion)
            for each OD pair results of pathfinder with/without capacity restriction

        """
        # deprecation
        method = method.lower()
        if 'all_or_nothing' in kwargs:
            kwargs.pop('all_or_nothing', None)
            method = 'aon'
            print("all_or_nothing=True is deprecated. use method = 'aon' instead")
        if 'volume_column' in kwargs:
            volume_column = kwargs.pop('volume_column')
            segments = [volume_column]
            print("volume_column is depreacated, use segments instead (ex: segments=['car'])")
        if method not in ['msa', 'fw', 'bfw', 'aon']:
            print(method, ' not supported. use msa, fw, bfw or aon')
            return

        network = init_network(self, method, segments, time_column, access_time, ntleg_penalty)
        volumes = init_volumes(self, od_set)

        if method == 'aon':
            self.car_los = aon_roadpathfinder(network, volumes, time_column, num_cores)
            self.car_los = get_car_los_time(self.car_los, self.road_links, self.zone_to_road, 'time', 'time')
            return

        # elif method in ['msa', 'fw', 'bfw']:
        if turn_penalties is None:
            links, car_los, rel_gap = msa_roadpathfinder(
                network,
                volumes,
                segments=segments,
                method=method,
                time_col=time_column,
                tracker_plugin=tracker_plugin,
                num_cores=num_cores,
                return_car_los=return_car_los,
                **kwargs,
            )
        else:
            links, car_los, rel_gap = expanded_roadpathfinder(
                network,
                volumes,
                zones=self.zones,
                segments=segments,
                method=method,
                time_col=time_column,
                tracker_plugin=tracker_plugin,
                turn_penalties=turn_penalties,
                num_cores=num_cores,
                return_car_los=return_car_los,
                **kwargs,
            )

        time_dict = links['jam_time'].to_dict()
        self.road_links['jam_time'] = self.road_links.set_index(['a', 'b']).index.map(time_dict.get)
        self.road_links['jam_speed'] = self.road_links['length'] / self.road_links['jam_time'] * 3.6

        volume_dict = links['flow'].to_dict()
        self.road_links['flow'] = self.road_links.set_index(['a', 'b']).index.map(volume_dict.get)
        for seg in segments:
            volume_dict = links[(seg, 'flow')].to_dict()
            self.road_links[(seg, 'flow')] = self.road_links.set_index(['a', 'b']).index.map(volume_dict.get)
        if return_car_los:
            car_los = get_car_los_time(car_los, self.road_links, self.zone_to_road, 'jam_time', 'time')
            self.car_los = car_los
        self.relgap = rel_gap

    @track_args
    def step_pr_pathfinder(self, force=False, path_analysis=True, **kwargs):
        """Park and Ride pathfinder algorithm : shortest path algorithm on the graph
        built from links (public transport routes) and road_links considered as "public transport access links" (with car speed).

        Requires
        ----------
        self.zones
        self.links
        self.footpaths
        self.zone_to_road
        self.zone_to_transit
        self.transit_to_zone
        self.road_to_transit
        self.road_links

        Parameters
        ----------
        force : bool, optional, default False
            If True, will NOT perform integrity_test_collision on the 'nodes', 'links', 'zones','road_nodes', 'road_links'.

        path_analysis : bool, optional, default True
            Performs paths analysis, adds columns 'all_walk' and 'ntransfers' to the output pt_los

        od_set : dict, optional, default None
            set of od to use - may be used to reduce computation time (default None)
            for example, the od_set is the set of od for which there is a volume in self.volumes

        cutoff : default np.inf
            description

        Builds
        ----------
        self.pr_los

        """

        if not force:
            sets = ['nodes', 'links', 'zones', 'road_nodes', 'road_links']
            self.integrity_test_collision(sets)
        self.links = engine.graph_links(self.links)
        parkridepathfinder = ParkRidePathFinder(self)
        parkridepathfinder.find_best_path(**kwargs)

        self.pr_los = parkridepathfinder.paths

        if path_analysis:
            analysis_nodes = pd.concat([self.nodes, self.road_nodes])
            analysis_links = pd.concat([self.links, self.road_links])
            self.pr_los = analysis.path_analysis_od_matrix(
                od_matrix=self.pr_los, links=self.links, nodes=analysis_nodes, centroids=self.centroids
            )  # analyse non vérifiée, prise directement depuis pt_los

    @track_args
    def step_pt_pathfinder(
        self,
        broken_routes=True,
        broken_modes=True,
        route_column='route_id',
        mode_column='route_type',
        boarding_time=None,
        speedup=False,
        walk_on_road=False,
        # keep_graph=False,
        keep_pathfinder=False,
        force=False,
        path_analysis=True,
        **kwargs,
    ):
        """Performs public transport pathfinder.

        With :
            - all or nothing Diskjstra algorithm if broken_routes=False AND broken_modes=False
            - Prunning algorithm if broken_routes=True OR/AND broken_modes=True
        For optimal strategy pathfinder, use step_pt_pathfinder of the class OptimalModel

        For connection scan pathfinder algorithm (with time tables), use step_pt_pathfinder of the class ConnectionScanModel.

        Requires
        ----------
        self.zones
        self.links
        self.footpaths
        self.zone_to_road
        self.zone_to_transit

        Parameters
        ----------
        broken_routes : bool, optional, default True
            If True, will perform the route breaker of the pathfinder prunning algorithm
            with the different routes found in the route_column

        broken_modes : bool, optional, default True
            If True, will perform the mode breaker of the pathfinder prunning algorithm
            with the different modes found in the mode_column

        route_column : str, optional, default 'route_id'
            columns of the self.links containing the routes identifier (prunning algorithm)

        mode_column : str, optional, default 'route_type'
            columns of the self.links containing the modes identifier (prunning algorithm)

        boarding_time : float, optional, default None
            aditional boarding time

        alighting_time : float, optional, default None
            aditional alighting time

        speedup : bool, optional, default False
            Speed up the computation time, by

        walk_on_road : bool, optional, default False
            If True, will consider using the road network and zone_to_road for pedestrian paths.
            Warning : it will only compare those paths using the raod network with the paths using pedestrian links (footpaths, zone_to_transit)
            Force the use of road network for pedestrian paths by NOT defining footpaths and zone_to_transit

        keep_pathfinder : bool, optional, default False
            If True, keeps all computation steps of the pathfinder
            Use to performed advanced route and mode breaker (without the need to create submodel for route_breaker, for example)

        force : bool, optional, default False
            If True, will NOT perform integrity_test_collision on the 'nodes', 'links', 'zones','road_nodes', 'road_links'.

        path_analysis : bool, optional, default True
            Performs paths analysis, adds columns 'all_walk' and 'ntransfers' to the output pt_los


        Builds
        ----------
        self.pt_los :

        """
        sets = ['nodes', 'links', 'zones']
        if walk_on_road:
            sets += ['road_nodes', 'road_links']

        if not force:
            self.integrity_test_collision(sets)

        self.links = engine.graph_links(self.links)
        print('start publicpathfinder')
        publicpathfinder = PublicPathFinder(self, walk_on_road=walk_on_road)
        publicpathfinder.find_best_paths(
            broken_routes=broken_routes,
            broken_modes=broken_modes,
            route_column=route_column,
            mode_column=mode_column,
            boarding_time=boarding_time,
            **kwargs,
        )

        # if keep_graph:
        #     self.nx_graph=publicpathfinder.nx_graph

        if keep_pathfinder:
            self.publicpathfinder = publicpathfinder

        self.pt_los = publicpathfinder.paths
        analysis_nodes = pd.concat([self.nodes, self.road_nodes]) if walk_on_road else self.nodes
        print('path_analysis')
        if path_analysis:
            self.pt_los = analysis.path_analysis_od_matrix(
                od_matrix=self.pt_los, links=self.links, nodes=analysis_nodes, centroids=self.centroids
            )

    @track_args
    def step_modal_split(self, build_od_stack=True, **modal_split_kwargs):
        """Performs modal split. Use only for simple models with the modes Public Transport and Car,
        with few details : only based on duration and modal penalties.
        Based on modes demand and levels of services, it returns the volume by mode.
        Does not include price.
        For modal split, prefer the use of function step_logit

            * Utility(car) = alpha_car * 'duration_car' + beta_car
            * Utility(pt) = 'duration_pt'

        Requires
        ----------
        self.volumes :
            all mode origin->destination demand matrix
        self.los :
            levels of service. An od stack matrix with 'duration_pt' and 'duration_car'

        Parameters
        ----------
        build_od_stack : bool, optional
            _description_, by default True

        time_scale : float
            time scale of the logistic regression that compares utilities. Defines selectiveness.
            Defined as 1/(utility value of time, in seconds)

        alpha_car : float
            multiplicative penalty on 'duration_car' for the calculation of 'utility_car'

        beta_car : float
            additive penalty on 'duration_car' for the calculation of 'utility_car'


        Builds
        ----------
        self.od_stack :
        self.shared :

        Examples
        --------
        ::
            los = pd.merge(car_los, pt_los, on=['origin', 'destination'], suffixes=['_car', '_pt'])

            sm.step_modal_split(time_scale=1 / 1800, alpha_car=2, beta_car=600)
        """
        shared = engine.modal_split_from_volumes_and_los(self.volumes, self.los, **modal_split_kwargs)
        # shared['distance_car'] = shared['distance']
        if build_od_stack:
            self.od_stack = analysis.volume_analysis_od_matrix(shared)

        self.shared = shared

    def compute_los_volume(self, time_expanded=False, keep_segments=True):
        """Compute volumes in the level of services table from volumes and probabilities
        computed in the step_logit.

        Requires
        ----------
        self.los :
            concatenation of levels of services of the different modes (usually pt_los and car_los)
        self.volumes

        Parameters
        ----------
        time_expanded : bool, optional, default False
            Use True for models using timetables and ConnectionScan models
        keep_segments : bool, optional, default True
            True to use model segments - compute volumes per path per segment


        Builds
        ----------
        self.los :
            add volume column

        """
        los = self.los if not time_expanded else self.te_los

        segments = self.segments
        probabilities = [(segment, 'probability') for segment in segments]

        shared_cols = list(set(self.volumes.columns).intersection(set(los.columns)))
        on = [col for col in shared_cols if col in ['origin', 'destination', 'wished_departure_time']]

        left = los[on + probabilities]
        left['index'] = left.index
        df = pd.merge(left, self.volumes, on=on).set_index('index')
        df = df.reindex(los.index)
        values = df[probabilities].values * df[segments].values
        i = 0
        for segment in segments:
            los[segment] = values.T[i]
            i += 1
        los['volume'] = np.nansum(values, axis=1)

        if time_expanded:
            los_volumes = self.te_los.groupby('path_id')[['volume'] + segments].sum()
            path_id_list = list(self.los['path_id'])
            volume_values = los_volumes.reindex(path_id_list).fillna(0).values
            for c in los_volumes.columns:
                self.los[c] = np.nan  # create_columns
            self.los.loc[:, los_volumes.columns] = volume_values

    def step_assignment(
        self,
        road=False,
        boardings=False,
        boarding_links=False,
        alightings=False,
        alighting_links=False,
        transfers=False,
        segmented=False,
        time_expanded=False,
        compute_los_volume=True,
    ):
        """Performs assignment : compute the volumes on the links of the public transport network,
        and the boardings and alightings on the nodes of the PT network.

        Requires
        ----------
        self.los :
            concatenation of levels of services of the different modes (usually pt_los and car_los)
        self.volumes

        Parameters
        ----------
        road : bool, optional, default False
            Assign car volume.
            If road_link_list exists (columns of self.links computed with preparation_cast_network function)
            Add public transport volume on road_links.
            Requires
        boardings : bool, optional, default False
            If True, compute boardings to add to the nodes dataframe
        boarding_links : bool, optional, default False
            If True, compute boardings to add to the links dataframe
        alightings : bool, optional, default False
            If True, compute alightings to add to the nodes dataframe
        alighting_links : bool, optional, default False
            If True, compute alightings to add to the links dataframe
        transfers : bool, optional, default False
            If True, compute number of transfers to add to the nodes dataframe
        segmented : bool, optional, default False
            If True, use model segments - compute volumes on the links per segment
        time_expanded : bool, optional, default False
            Use True for models using timetables and ConnectionScan models
        compute_los_volume : bool, optional, default True
            True to add column volumes in los dataframe

        Builds
        ----------
        self.los :
            add volume column if compute_los_volume=True
        self.links :
            add volumes on public transport links, can add boardings, alightings,
            can be per segment, depending on parameters
        self.nodes :
            Depending on parameters, add boardings, alightings, transfers
        self.road_links :
            add public transport volumes on road_links if road=True
        """

        if compute_los_volume:
            self.compute_los_volume(time_expanded=time_expanded)
        los = self.los.copy()

        column = 'link_path'
        l = los.dropna(subset=[column])
        l = l.loc[l['volume'] > 0]
        self.links['volume'] = assign(l['volume'], l[column])

        if road:
            self.road_links[('volume', 'car')] = assign(l['volume'], l[column])
            if 'road_link_list' in self.links.columns:
                to_assign = self.links.dropna(subset=['volume', 'road_link_list'])
                self.road_links[('volume', 'pt')] = assign(to_assign['volume'], to_assign['road_link_list'])

        if boardings and not boarding_links:
            print('to assign boardings on links pass boarding_links=True')
        if boarding_links:
            column = 'boarding_links'
            l = los.dropna(subset=[column])
            self.links['boardings'] = assign(l['volume'], l[column])

        if boardings:
            column = 'boardings'
            l = los.dropna(subset=[column])
            self.nodes['boardings'] = assign(l['volume'], l[column])

        if alighting_links:
            column = 'alighting_links'
            l = los.dropna(subset=[column])
            self.links['alightings'] = assign(l['volume'], l[column])

        if alightings:
            column = 'alightings'
            l = los.dropna(subset=[column])
            self.nodes['alightings'] = assign(l['volume'], l[column])

        if transfers:
            column = 'transfers'
            l = los.dropna(subset=[column])
            self.nodes['transfers'] = assign(l['volume'], l[column])

        if segmented:
            self.segmented_assigment(
                road=road, boardings=boardings, alightings=alightings, transfers=transfers, aggregated_los=los
            )

    def segmented_assigment(self, road=False, boardings=False, alightings=False, transfers=False, aggregated_los=None):
        """Performs assignment per segment : compute the volumes on the links of the public transport network,
        and the boardings and alightings on the nodes of the PT network,
        keeping the memory of segments in the volumes.

        Requires
        ----------
        self.los :
            concatenation of levels of services of the different modes (usually pt_los and car_los)
        self.volumes

        Parameters
        ----------
        road : bool, optional, default False
            Add public transport volume on road_links.
            Requires road_link_list - columns of self.links computed with preparation_cast_network function.

        boardings : bool, optional, default False
            If True, compute boardings to add to the nodes dataframe

        alightings : bool, optional, default False
            If True, compute alightings to add to the nodes dataframe

        transfers : bool, optional, default False
            If True, compute number of transfers to add to the nodes dataframe

        aggregated_los : string, optional, default None
            Name of attributes containing model aggregated los


        Builds
        ----------
        self.los :
            add volume column if compute_los_volume=True
        self.links :
            add volumes on public transport links per segment,
            can add boardings, alightings,depending on parameters
        self.nodes :
            Depending on parameters, add boardings, alightings, transfers
        self.road_links :
            add public transport volumes on road_links if road=True

        """
        los = aggregated_los if aggregated_los is not None else self.los
        for segment in self.segments:
            column = 'link_path'
            l = los.dropna(subset=[column])
            self.links[segment] = assign(l[segment], l[column])
            if road:
                self.road_links[(segment, 'car')] = assign(l[segment], l[column])
                self.road_links[(segment, 'pt')] = assign(self.links[segment], self.links['road_link_list'])
            if boardings:
                column = 'boarding_links'
                l = los.dropna(subset=[column])
                self.links[(segment, 'boardings')] = assign(l[segment], l[column])

                column = 'boardings'
                l = los.dropna(subset=[column])
                self.nodes[(segment, 'boardings')] = assign(l[segment], l[column])

            if alightings:
                column = 'alighting_links'
                l = los.dropna(subset=[column])
                self.links[(segment, 'alightings')] = assign(l[segment], l[column])

                column = 'alightings'
                l = los.dropna(subset=[column])
                self.nodes[(segment, 'alightings')] = assign(l[segment], l[column])

            if transfers:
                column = 'transfers'
                l = los.dropna(subset=[column])
                self.nodes[(segment, 'transfers')] = assign(l[segment], l[column])

    @track_args
    def step_pt_assignment(self, volume_column=None, on_road_links=False, split_by=None, **kwargs):
        """Performs assignment of the indicated volume column: compute the volumes on the links of the public transport network,
        and the boardings and alightings on the nodes of the PT network.
        This function is older and slower, will soon be deprecated, if possible priviledge the use of function step_assigment.
        Compared to step_assignment :
            - uses pt_los (and not los)
            - Can specify the volume column
            - Can strack path categories

        Requires
        ----------
        self.links
        self.nodes
        self.pt_los :
            requires computed path probabilities in pt_los for each segment,
            they can be computed with funcions analysis_mode_utility + step_logit
        self.road_links
        self.volumes

        Parameters
        ----------
        volume_column : string, optional, default None
           volume column of self.volumes to assign. If none, all columns will be assigned
        on_road_links : bool, optional, default False
            if True, performs pt assignment on road_links as well
        split_by : string, optional, default None
            path categories to be tracked in the assignment. Must be a column of self.pt_los

        Builds
        ----------
        self.loaded_links
        self.loaded_nodes
        self.road_links :
            add public transport load column

        example:
        ::
            sm.step_assignment(
                    volume_column=None,
                    on_road_links=False,
                    split_by='route_type',
                    boardings=True,
                    alightings=True,
                    transfers=True
                }
            )
        """
        if volume_column is None:
            self.segmented_pt_assignment(on_road_links=on_road_links, split_by=split_by, **kwargs)
            return

        # When split_by is not None, this call could be replaced by a sum, provided
        # prior dumb definition of loaded_links and loaded_nodes
        self.loaded_links, self.loaded_nodes = engine.loaded_links_and_nodes(
            self.links,
            self.nodes,
            volumes=self.volumes,
            path_finder_stack=self.pt_los,
            volume_column=volume_column,
            **kwargs,
        )

        # Rename columns
        self.loaded_links.rename(columns={volume_column: ('load', volume_column)}, inplace=True)
        self.loaded_nodes.rename(columns={volume_column: ('load', volume_column)}, inplace=True)
        for col in list(set(['boardings', 'alightings', 'transfers']).intersection(kwargs.keys())):
            self.loaded_links.rename(columns={col: (col, volume_column)}, inplace=True)
            self.loaded_nodes.rename(columns={col: (col, volume_column)}, inplace=True)

        # Group assignment
        if split_by is not None:
            groups = self.pt_los[split_by].unique()
            for group in groups:
                # TODO remove rows with empty link_path
                group_pt_los = self.pt_los.loc[self.pt_los[split_by] == group]
                group_loaded_links, group_loaded_nodes = engine.loaded_links_and_nodes(
                    self.links,
                    self.nodes,
                    volumes=self.volumes,
                    path_finder_stack=group_pt_los,
                    volume_column=volume_column,
                    **kwargs,
                )
                # Append results columns
                self.loaded_links[('load', volume_column, group)] = group_loaded_links[volume_column]
                self.loaded_nodes[('load', volume_column, group)] = group_loaded_nodes[volume_column]
                for col in list(set(['boardings', 'alightings', 'transfers']).intersection(kwargs.keys())):
                    self.loaded_links[(col, volume_column, group)] = group_loaded_links[col]
                    self.loaded_nodes[(col, volume_column, group)] = group_loaded_nodes[col]

        # Assignment on road_links
        if on_road_links:
            if 'road_link_path' not in self.pt_los.columns:
                # create road_link_path column from networkcasted linkss if not already defined
                self._analysis_road_link_path()

            merged = pd.merge(self.pt_los, self.volumes, on=['origin', 'destination'])
            merged['to_assign'] = merged[(volume_column, 'probability')] * merged[volume_column].fillna(0)

            if split_by is not None:

                def assign_group(g):
                    x = g.reset_index()
                    result = raw_assignment.assign(x['to_assign'], x['road_link_path'])
                    return result

                group_assigned = merged.groupby(split_by).apply(assign_group)
                assigned = group_assigned.unstack().T.loc['volume'].fillna(0)
                # Add empty groups
                for empty in list(set(groups).difference(set(assigned.columns))):
                    assigned[empty] = 0
                self.road_links[[(volume_column, col) for col in groups]] = assigned[[col for col in groups]]
                self.road_links[volume_column] = assigned.T.sum()

            else:  # no groups
                assigned = raw_assignment.assign(merged['to_assign'], merged['road_link_path'])
                self.road_links[volume_column] = assigned['volume']

            # todo remove 'load' from analysis module:
            self.road_links['load'] = self.road_links[volume_column]

    def segmented_pt_assignment(self, split_by=None, on_road_links=False, *args, **kwargs):
        """Performs pt assignment for all demand segments. Function used in step_pt_assignment function,
        refer to this function for other args.

        Requires
        ----------
        self.links
        self.nodes
        self.pt_los :
            requires computed path probabilities in pt_los for each segment
        self.road_links
        self.volumes

        Parameters
        ----------
        on_road_links : bool, optional, default False
            if True, performs pt assignment on road_links as well
        split_by : string, optional, default None
            path categories to be tracked in the assignment. Must be a column of self.pt_los

        Builds
        ----------
        self.loaded_links
        self.loaded_nodes
        self.road_links :
            add public transport load column
        """
        segments = self.segments

        iterator = tqdm(segments)
        for segment in iterator:
            iterator.desc = str(segment)
            # Assign demand segment
            self.step_pt_assignment(
                volume_column=segment,
                path_pivot_column=(segment, 'probability'),
                split_by=split_by,
                on_road_links=on_road_links,
                **kwargs,
            )
            # Update links and nodes to keep results as loaded links and nodes
            # are erased at each call of step_pt_assignment
            self.links = self.loaded_links
            self.nodes = self.loaded_nodes

        # Group assignment results: sum over demand segments
        try:
            groups = self.pt_los[split_by].unique()
        except KeyError:
            groups = []
        cols = ['load']
        # Add boardings, alightings and transfers if processed
        cols += list(set(['boardings', 'alightings', 'transfers']).intersection(kwargs.keys()))
        for col in cols:
            for g in groups:
                columns = [tuple([col, s, g]) for s in segments]
                name = tuple([col, g])
                self.loaded_links[name] = self.loaded_links[columns].T.sum()
                self.loaded_links.drop(columns, axis=1, inplace=True)

                self.loaded_nodes[name] = self.loaded_nodes[columns].T.sum()
                self.loaded_nodes.drop(columns, axis=1, inplace=True)

            columns = [tuple([col, s]) for s in segments]
            self.loaded_links[col] = self.loaded_links[columns].T.sum()
            self.loaded_links.drop(columns, axis=1, inplace=True)

            self.loaded_nodes[col] = self.loaded_nodes[columns].T.sum()
            self.loaded_nodes.drop(columns, axis=1, inplace=True)

        if on_road_links:
            for group in groups:
                self.road_links[('all', group)] = self.road_links[[(s, group) for s in segments]].T.sum()
                self.road_links.drop([(s, group) for s in segments], axis=1, inplace=True)

            self.road_links['load'] = self.road_links[[s for s in segments]].T.sum()
            self.road_links.drop([s for s in segments], axis=1, inplace=True)

    def step_car_assignment(self, volume_column=None):
        """Performs car assignment of the indicated volume column: compute the volumes on the road_links of the private transport network.
        This function is older and slower, will soon be deprecated, priviledge the use of function step_assigment and step_road_pathfinder.
        Compared to step_assignment :
            - uses car_los (and not los)
            - Can specify the volume column

        Requires
        ----------
        self.road_links
        self.road_nodes
        self.car_los :
            requires computed path probabilities in car_los for each segment,
            they can be computed with funcions analysis_mode_utility + step_logit
        self.volumes

        Parameters
        ----------
        volume_column : string, optional, default None
           volume column of self.volumes to assign. If none, all columns will be assigned

        Builds
        ----------
        self.loaded_road_links
        self.loaded_road_nodes

        """
        if volume_column is None:
            self.segmented_car_assignment()
        else:
            merged = pd.merge(self.car_los, self.volumes, on=['origin', 'destination'])
            merged['to_assign'] = merged[(volume_column, 'probability')] * merged[volume_column].fillna(0)
            assigned = raw_assignment.assign(merged['to_assign'], merged['link_path']).fillna(0)
            self.road_links[(volume_column, 'car')] = assigned

    def segmented_car_assignment(self):
        """Performs car assignment for all demand segments. Function used in step_car_assignment function,
        refer to this function for other args and parameters.

        Requires
        ----------
        self.road_links
        self.road_nodes
        self.car_los :
            requires computed path probabilities in car_los for each segment
        self.volumes

        Builds
        ----------
        self.loaded_road_links
        self.loaded_road_nodes
        """

        segments = self.segments
        iterator = tqdm(segments)
        for segment in iterator:
            iterator.desc = str(segment)
            merged = pd.merge(self.car_los, self.volumes, on=['origin', 'destination'])
            merged['to_assign'] = merged[(segment, 'probability')] * merged[segment].fillna(0)
            assigned = raw_assignment.assign(merged['to_assign'], merged['link_path']).fillna(0)
            self.road_links[(segment, 'car')] = assigned

        columns = [(segment, 'car') for segment in self.segments]
        self.road_links[('all', 'car')] = self.road_links[columns].T.sum()
        # TODO Merge conflict: TO CHECK WITH ACCRA
        # self.road_links.drop(columns, 1, inplace=True)
        # if not 'load' in self.road_links.columns:
        #    self.road_links['load'] = 0
        # self.road_links['load'] += self.road_links[('all','car')]

    # TODO move all utility features to another object / file

    def analysis_mode_utility(self, how='min', segment=None, segments=None, time_expanded=False):
        """Compute utilities per mode per segment based on logit parameters.

        Before applying this function :
            -- the function preparation_logit allows to create the
                stepmodel attributes utility_values, logit_scales, mode_utility and mode_nests
                required in the analysis_mode_utility function
            -- the functions analysis_pt_route_type and analysis_car_route_type may also be needed
                to builds 'route_type' in pt_los (resp in car_los) based on 'route_types'
            -- concatenate pt_los and car_los

        The function analysis_mode_utility computes the utility based on the chosen variables of the utility
        functions. The coefficients of the variable in the utility fonctions must be found in the utility_values model attribute
        and in the columns of the model los (default : price, time, ntransfers). They can be defined by segment.
        The modal constant can also be defined per segment and must be found in the table mode_utility.

        Requires
        ----------
        self.mode_utility
        self.los
        self.utility_values
        self.segments

        Parameters
        ----------
        how : ['main'|'min'|'max'|'mean'|'sum'], optional, default 'min'
            What type of agregation for mode constant. Options are:

            'main'   --(default)  the utility (constant) of the 'route_type' is used

            'min'    -- minimum of the utilities (constant) of the modes in route_types

            'max'   -- maximum of the utilities (constant) of the modes in route_types

            'mean'   -- mean of the utilities (constant) of the modes in route_types

            'sum'   -- sum of the utilities (constant) of the modes in route_types

        segment : string, optional, default None
            Unique segment for which to compute utility. If None, will perform for all segments of self.segments
        segments : list, optional, default None
            List of segments of the model for which to compute utility
        time_expanded : bool, optional, default False
            Use True for models using timetables and ConnectionScan models


        Builds
        ----------
        self.los :
            add columns (segment, 'utility') - value of the utility per segment

        """
        if segment is None:
            for segment in tqdm(self.segments):
                self.analysis_mode_utility(how=how, segment=segment, time_expanded=time_expanded)
            return
        if time_expanded:
            logit_los = self.te_los
        else:
            logit_los = self.los
        mode_utility = self.mode_utility[segment].to_dict()

        if how == 'main':  # the utility of the 'route_type' is used
            logit_los['mode_utility'] = logit_los['route_type'].apply(mode_utility.get)
        else:  # how = 'min', 'max', 'mean', 'sum'
            # route type utilities
            rtu = {
                rt: get_combined_mode_utility(rt, how=how, mode_utility=mode_utility)
                for rt in logit_los['route_types'].unique()
            }
            logit_los['mode_utility'] = logit_los['route_types'].map(rtu.get)

        utility_values = self.utility_values[segment].to_dict()
        u = 0
        for key, value in utility_values.items():
            u += value * logit_los[key]

        logit_los[(segment, 'utility')] = u
        logit_los[(segment, 'utility')] = logit_los[(segment, 'utility')]

    def analysis_utility(self, segment='root', time_expanded=False, how='min'):
        """DEPRECATED USE analysis_mode_utility"""
        if segment is None:
            for segment in self.segments:
                print(segment)
                self.analysis_mode_utility(how=how, segment=segment, time_expanded=time_expanded)
            return
        if time_expanded:
            los = self.te_los
        else:
            los = self.los

        utility_values = self.utility_values[segment].to_dict()
        u = 0
        for key, value in utility_values.items():
            u += value * los[key]

        los[(segment, 'utility')] = u
        los[(segment, 'utility')] = los[(segment, 'utility')].astype(float)

    def initialize_logit(self):
        """Not necessary
        creates tables od_probabilities and od_utilities
        """
        zones = list(self.zones.index)
        od = pd.DataFrame(index=pd.MultiIndex.from_product([zones, zones]))
        self.od_probabilities = od.copy()
        self.od_utilities = od.copy()

    def step_logit(
        self,
        time_expanded=False,
        decimals=None,
        n_paths_max=None,
        nchunks=10,
        workers=1,
        keep_od_tables=True,
        symmetric=False,
    ):
        """Performs the nested logit : compute the probabilities per segment of the paths in self.los
        after having computed the utilities with function analysis_mode_utility.
        prob(mode k)= exp(utility mode k)/sum(utilities, all modes).
        Parametrize the nested logit with model attribute mode_nests and logit_scales
        created with function preparation_logit.

        Requires
        ----------
        self.mode_nests
        self.logit_scales
        self.los

        Parameters
        ----------
        time_expanded : bool, optional, default False
            Use True for models using timetables and ConnectionScan models
        decimals : float, optional, default None
            minimum probability (avoid very small volumes)
        n_paths_max : int, optional, default None
            Maximum number of paths to keep per OD (avoid very small volumes)
        nchunks : int, optional, default 10
            Parameter to speed up computation time (division of calculation)
        workers : int, optional, default 1
            Parameter to speed up computation time (division of calculation)
        keep_od_tables : bool, optional, default True
            _description_, by default True
        symmetric

        Builds
        ----------
        self.los
            Add columns (segment, 'probability') of probabilities per segment
        self.od_utilities
        self.od_probabilities
        self.path_utilities
        self.path_probabilities

        """
        # concatenate paths
        od_cols = ['origin', 'destination']
        if time_expanded:
            od_cols.append('wished_departure_time')
        if symmetric & (nchunks > 1):
            raise Exception('symmetric utility unspported for nchunks > 1')
        to_concat = []
        for segment in self.segments:
            keep_columns = od_cols + ['route_type', (segment, 'utility')]
            if time_expanded:
                paths = self.te_los[keep_columns]
            else:
                paths = self.los[keep_columns]

            paths.rename(columns={(segment, 'utility'): 'utility'}, inplace=True)
            paths = paths.dropna(subset=['utility'])
            paths['segment'] = segment
            to_concat.append(paths)
        segmented_paths = pd.concat(to_concat)

        try:
            # all the segments can be proccessed together
            # assert all logit scales are the same and pick one
            logit_scales = self.logit_scales.T.drop_duplicates().T
            assert len(logit_scales.columns) == 1
            logit_scales.columns = ['root']
            nls = logit_scales['root'].to_dict()

            # assert all mode_nests are the same and pick one
            mode_nests = self.mode_nests.T.drop_duplicates().T
            assert len(mode_nests.columns) == 1
            mode_nests.columns = ['root']
            nests = mode_nests.reset_index().groupby('root')['route_type'].agg(lambda s: list(s)).to_dict()

            p, mu, mp = nested_logit.nested_logit_from_paths(
                segmented_paths,
                od_cols,
                mode_nests=nests,
                phi=nls,
                verbose=False,
                decimals=decimals,
                n_paths_max=n_paths_max,
                nchunks=nchunks,
                workers=workers,
                return_od_tables=keep_od_tables,
                symmetric=symmetric,
            )

        except AssertionError:
            p_list = []
            mu_list = []
            mp_list = []
            for segment in self.segments:
                mode_nests = (
                    self.mode_nests.reset_index().groupby(segment)['route_type'].agg(lambda s: list(s)).to_dict()
                )
                nls = self.logit_scales[segment].to_dict()
                paths = segmented_paths.loc[segmented_paths['segment'] == segment]
                p, mu, mp = nested_logit.nested_logit_from_paths(
                    paths,
                    mode_nests=mode_nests,
                    phi=nls,
                    od_cols=od_cols,
                    decimals=decimals,
                    n_paths_max=n_paths_max,
                    symmetric=symmetric,
                )
                p_list.append(p)
                mu_list.append(mu)
                mp_list.append(mp)

            p = pd.concat(p_list)
            mu = pd.concat(mu_list, ignore_index=True)
            mp = pd.concat(mp_list, ignore_index=True)

        p.reset_index(inplace=True)
        p.set_index(['segment', 'index'], inplace=True)
        for segment in self.segments:
            if time_expanded:
                self.te_los[(segment, 'probability')] = p.loc[segment]['probability']
            else:
                assert self.los.index.is_unique, 'los must have unique index'
                self.los[(segment, 'probability')] = p.loc[segment]['probability']

        self.probabilities = mp
        self.utilities = mu


def get_combined_mode_utility(route_types, mode_utility, how='min'):
    """Agregate modal constants

    Parameters
    ----------
    route_types : list
        List of the route_types in the path
    mode_utility : dataframe
        Modal constants
    how : ['main'|'min'|'max'|'mean'|'sum'], optional, default 'min'
            What type of agregation for mode constant. Options are:

            'main'   --(default)  the utility (constant) of the 'route_type' is used

            'min'    -- minimum of the utilities (constant) of the modes in route_types

            'max'   -- maximum of the utilities (constant) of the modes in route_types

            'mean'   -- mean of the utilities (constant) of the modes in route_types

            'sum'   -- sum of the utilities (constant) of the modes in route_types

    """
    utilities = [mode_utility[mode] for mode in route_types]
    if not len(utilities):
        return 0
    if how == 'min':  # worse mode
        return min(utilities)
    elif how == 'max':  # best mode
        return max(utilities)
    elif how == 'sum':
        return sum(utilities)
    elif how == 'mean':
        return sum(utilities) / len(utilities)
