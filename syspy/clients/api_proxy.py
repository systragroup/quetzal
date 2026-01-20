import json
import time
from time import sleep
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString


def google_maps_encoder(
    latitude_origin,
    longitude_origin,
    latitude_destination,
    longitude_destination,
    timestamp=int(time.time()) + 1,
    token=None,
    mode='driving',
):
    def in_url(coordinates):
        """
        :param coordinates: list of coordinates [longitude, latitude]
        :type coordinates: list
        :param mode: in {'driving', 'walking', 'bicicling', 'transit'}
        :return: in_url_coordninates
        :rtype: str
        """
        return str(coordinates[1]) + ',' + str(coordinates[0])

    api_url = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
    proto_url = api_url + 'origins={0}&destinations={1}'
    proto_url += '&mode={2}&language=en-EN&sensor=false&departure_time={3}&trafic_model=pessimistic&key={4}'

    url = proto_url.format(
        in_url([longitude_origin, latitude_origin]),
        in_url([longitude_destination, latitude_destination]),
        mode,
        int(timestamp),
        token,
    )
    # timestamp must be an int!
    return url


class Proxy:
    """
    This proxy is used to store and reuse requests made to an external API
    (for instance Google Maps, which is the default configuration).
    These computed results are stored in a local dataframe and then to a
    database.

    :ivar encoder: the function used to format the url called from the
        parameters.
    :ivar conn: the connexion to the database where the responses are stored
    :ivar table: the table name
    :ivar parameters: the parameters used to create the index in the database.
    :ivar tolerance: the tolerance to reuse an already computed request.
    :ivar auto_populate: automatically populates the local dataframe and the
        database if True. Otherwise the adding process must be done by the user.

    example:
    ::
        # using it with Google Maps (provided by default)
        # Create proxy instance
        data_path = r'api_proxy/'
        conn = sqlite3.connect(data_path + 'google_maps_itineraries.db')
        proxy = api_proxy.Proxy(conn=conn)
        # Make a request
        data = {
            'latitude_origin': 48.833405,
            'longitude_origin': 2.269831,
            'latitude_destination': 48.422811,
            'longitude_destination': 2.585593,
            'timestamp': time.time(),
            'token': token,  # You need a valid token here
        }
        resp = proxy.get(**data)
    """

    def __init__(
        self,
        conn,
        encoder=google_maps_encoder,
        table='json',
        parameters=('latitude_origin', 'longitude_origin', 'latitude_destination', 'longitude_destination'),
        tolerance=1e-4,
        auto_populate=False,
        verbose=False,
    ):
        """
        Initialise the proxy:
        - initialize parameters and default attributes
        - create or load the database.
        """

        self.populated = 0
        self.conn = conn
        self.table = table
        self.encoder = encoder
        self.parameters = parameters
        self.verbose = verbose
        self.auto_populate = auto_populate

        sql = 'SELECT * FROM ' + table
        try:
            self.local = pd.read_sql(sql, conn)
        except Exception:  # no such table:
            print('create table')
            empty = pd.DataFrame(index=list(parameters) + ['current', 'json']).T
            empty.to_sql(table, conn, index=False, if_exists='fail')
            self.local = pd.read_sql(sql, conn)
            self.local = self.local.astype(np.float)  # todo: not type safe

        self.local['current'] = False
        self.local.set_index(list(parameters + ('current',)), inplace=True)
        self.local.sort_index(inplace=True)
        self.local = self.local['json']

        self.tolerance = {}
        try:
            self.tolerance.update(tolerance)
        # ...object is not iterable, if tolerance is not a dict
        except TypeError:
            self.tolerance = {p: tolerance for p in parameters}

        self.get_status = 0

    def indexer(self, kwargs):
        """
        Create index from parameters and request arguments.
        """
        return tuple(kwargs[p] for p in self.parameters)

    def slicer(self, kwargs):
        """
        Create a slicer from the request arguments and the tolerance parameter.
        """
        return tuple(slice(kwargs[p] - self.tolerance[p], kwargs[p] + self.tolerance[p]) for p in self.parameters)

    def request_api(self, kwargs):
        """
        Send a request to the api with the given arguments
        """
        assert self.get_status == 0
        if self.verbose:
            print('Sending request to API')
        response = requests.get(self.encoder(**kwargs))
        self.get_status = response.status_code
        to_return = json.dumps(response.json())  # a json string
        return to_return

    def populate(self, resp, **kwargs):
        """
        Add response to the local proxy database.
        """
        if self.verbose:
            print('Populate')
        self.populated += 1
        self.local.loc[self.indexer(kwargs) + (True,)] = resp
        self.clean_index()

    def get(self, **kwargs):
        """
        Method GET filtered by the proxy. Will return the api response if no
        similar result has been found in the db.
        """
        self.get_status = 0
        # Try to find a similar result in the db:
        try:
            # add 'current' naive slicer
            slices = self.slicer(kwargs) if self.tolerance else self.indexer(kwargs)
            responses = list(self.local.loc[slices + (slice(None),)])
            assert len(responses)
            if self.verbose:
                print('Result found!')

            # Return the first matching result
            return responses[0]

        except (KeyError, AssertionError, TypeError):
            # We send a request to the API
            resp = self.request_api(kwargs)
            # We populate the db if required
            if self.auto_populate:
                self.populate(resp=resp, **kwargs)
            return resp

    def clean_index(self):
        reset = self.local.reset_index()
        self.local = reset.set_index(list(self.parameters + ('current',)))['json']
        self.local.sort_index(inplace=True)

    def actual_requests(self):
        return self.local.reset_index('current')['current'].sum()

    def insert(self):
        """
        Insert results stored in the proxy local database into the real db.
        """
        if self.verbose:
            print('Insert')
        df = self.local.reset_index()
        df.loc[df['current']].to_sql(self.table, self.conn, if_exists='append', index=False)

        df['current'] = False
        self.local = df.set_index(list(self.parameters + ('current',)))['json']


def get_distance_matrix(
    origins, destinations=None, apiKey='', api='here', mode='car', region='polygon', time=None, buffer=0.1, verify=False
):
    """
    wrapper that return the time matrix (in seconds) for each OD
    with the Here matrix api or the google matrix api.

    parameters
    ----------
    origins (GeoDataframe) = geopandas dataframe with index and geometry (epsg:4326)

    destinations (None | GeoDataframe) = geopandas dataframe with index and geometry (epsg:4326)

    api (str) ='here', 'google'

    apiKey (str) : api key

    mode (str) = here : "car" "truck" "pedestrian" "bicycle" "taxi" "scooter" "bus" "privateBus".
    google : driving", "walking", "transit" "bicycling"

    region (str) = here : polygon or world. world is use for long distance call (>400km diamaters)

    time (None|str) = here : Time of departure at all origins, in ISO 8601 format: the time zone offset is required.
    datetime.datetime.now().astimezone().isoformat() for example ('2022-11-16T11:19:21.944095-05:00')
    google : timestamp in sec. ex: datetime.datetime(2023,2,7,7,0).timestamp(),
             no timezone (local timezone used. 7am in montreal is 7am in paris ). must be in the future.

    buffer (float) = here : 0.1 rad stating value, buffer will be increase (+0.1rad) while all origins & destinations are
    not included in the polygon around them. if it fail, you can provide a big buffer! (ex:1)

    returns
    ----------
    pd.dataframe index: origin, columns: destination. values: time in seconds
    """
    if origins.crs != 'EPSG:4326':
        origins = origins.to_crs(4326)
    df = origins.copy()
    origins_index = df.index.values
    # format geometry to here api format
    origins = list(df['geometry'].apply(lambda p: {'lat': p.y, 'lng': p.x}).values)
    # if destination, format them, else: use destination == origin
    if type(destinations) != type(None):
        if destinations.crs != 'EPSG:4326':
            destinations = destinations.to_crs(4326)
        df2 = destinations.copy()
        destinations_index = df2.index.values
        destinations = list(df2['geometry'].apply(lambda p: {'lat': p.y, 'lng': p.x}).values)
        df = pd.concat([df, df2])
    else:
        destinations = origins
        destinations_index = origins_index
    if api == 'here':
        # get centroid for the here region
        centroid = LineString(df['geometry'].values).centroid
        # center = {"lat":centroid.y,"lng":centroid.x}
        # create a polygon around the points. find a buffer for the centroid that include every od

        while not centroid.buffer(buffer).contains(LineString(df['geometry'].values)):
            buffer += 0.1
            if buffer > 1.7:
                break
        polygon = [
            {'lat': np.round(y, 5), 'lng': np.round(x, 5)}
            for x, y in list(zip(*centroid.buffer(buffer).exterior.coords.xy))
        ]
        polygon = _remove_duplicates(polygon)
        if region == 'polygon':
            regionDefinition = {'type': 'polygon', 'outer': polygon}
        elif region == 'world':
            regionDefinition = {'type': 'world'}
        else:
            raise Exception('{r} is not a valid region. use world or polygon.'.format(r=region))
        if buffer >= 1.7:
            print('buffer larger than 1.7. region definition set to world as the polygon is most likely')
            regionDefinition = {'type': 'world'}
        # departureTime : Time of departure at all origins, in ISO 8601 (RFC 3339)

        url = 'https://matrix.router.hereapi.com/v8/matrix?apiKey=' + apiKey + '&async=false'
        body = {
            'origins': origins,
            'destinations': destinations,
            'departureTime': time,
            'transportMode': mode,
            'regionDefinition': regionDefinition,
        }
        try:
            x = requests.post(url, json=body, verify=verify)
            resp = json.loads(x.text)
            if x.status_code != 200:
                raise Exception(resp)
        except:
            sleep(5)
            x = requests.post(url, json=body, verify=verify)
            resp = json.loads(x.text)
            if x.status_code != 200:
                raise Exception(resp)

        error_index = None
        if resp['matrix'].get('errorCodes') != None:
            if set(resp['matrix']['errorCodes']) != set([0, 3]):
                errors = np.array([err for err in resp['matrix']['errorCodes']]).reshape(
                    len(origins), len(destinations)
                )
                print('errors', errors)
                error_index = np.where((errors != 0) & (errors != 3))
        # format response to a dataframa OD with time in secs.
        mat = np.array([time for time in resp['matrix']['travelTimes']]).reshape(len(origins), len(destinations))
        if error_index != None:
            mat[error_index] = -9999
            print('times', mat)

    elif api == 'google':
        mode = {'car': 'driving', 'pedestrian': 'walking', 'bicycle': 'bicycling'}.get(mode, mode)
        api_url = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
        proto_url = api_url + 'origins={0}&destinations={1}'
        proto_url += '&mode={2}&language=en-EN&sensor=false&departure_time={3}&trafic_model=pessimistic&key={4}'
        url = proto_url.format(
            '|'.join([str(g['lat']) + ',' + str(g['lng']) for g in origins]),
            '|'.join([str(g['lat']) + ',' + str(g['lng']) for g in destinations]),
            mode,
            int(time),
            apiKey,
        )

        try:
            x = requests.get(url, verify=verify)
            resp = json.loads(x.text)
            if x.status_code != 200:
                raise Exception(resp)
        except:
            sleep(5)
            x = requests.get(url, verify=verify)
            resp = json.loads(x.text)
            if x.status_code != 200:
                raise Exception(resp)
        mat = []
        for i, origin in enumerate(resp['rows']):
            for j, destination in enumerate(origin['elements']):
                mat.append((destination['duration_in_traffic']['value']))
        mat = np.array(mat).reshape(len(origins), len(destinations))
    else:
        raise Exception('api should be here or google.')

    od = pd.DataFrame(mat, index=origins_index, columns=destinations_index)
    od.index.name = 'origin'
    od.columns.name = 'destination'

    return od


def get_batches(df, max_od=15):
    """
    gives index to divide dataframe into dataframes of length 15 max
    ex: [[0,15],[15,30],[30,35]] for a df of length 30.
    """
    div = divmod(len(df), max_od)
    batches = [[max_od * i, max_od * (i + 1)] for i in range(div[0])]
    if div[1] > 0:
        if div[0] == 0:
            batches.append([0, div[1]])
        else:
            batches.append([batches[-1][-1], batches[-1][-1] + div[1]])
    return batches


def multi_get_distance_matrix(origins, destinations, batch_size=(15, 15), api='here', **kwargs):
    """
    batch api call with 15x15 OD batches (limit per call for here)

    parameters
    ----------
    origins (GeoDataframe) = geopandas dataframe with index and geometry (epsg:4326)
    destination (GeoDataframe) = geopandas dataframe with index and geometry (epsg:4326)
    batch_size (tuple) : (origin, destination) batch sizes. (100,1) if 100 ori and 1 des . choices: (15,15), (100,1)
    api (str) : 'here' or 'google'
    **kwargs: get_distance_matrix(origins, destinations, **kwargs)

    returns
    ----------
    pd.dataframe index: origin, columns: destination. values: time in seconds
    """
    if api == 'here':
        assert len(origins) * len(destinations) <= 250_000, 'max 250 000 OD for free HERE api'

    elif api == 'google':
        assert len(origins) * len(destinations) <= 40_000, 'max 40 000 OD for free HERE api'
    else:
        raise Exception('api should be here or google')
    batches_origins = get_batches(origins, batch_size[0])
    batches_destinations = get_batches(destinations, batch_size[1])
    mat = pd.DataFrame()
    for batch_o in batches_origins:
        ori = origins.iloc[batch_o[0] : batch_o[1]]
        temp_mat = pd.DataFrame()
        for batch_d in batches_destinations:
            des = destinations.iloc[batch_d[0] : batch_d[1]]
            try:
                res = get_distance_matrix(origins=ori, destinations=des, api=api, **kwargs)
            except:
                sleep(3)
                res = get_distance_matrix(origins=ori, destinations=des, api=api, **kwargs)
            temp_mat = pd.concat([temp_mat, res], axis=1)
            sleep(0.2)
        mat = pd.concat([mat, temp_mat], axis=0)
    return mat


def _remove_duplicates(ls):
    seen = {}
    result = []

    for item in ls:
        if str(item) not in seen:
            result.append(item)
            seen[str(item)] = True
    return result
