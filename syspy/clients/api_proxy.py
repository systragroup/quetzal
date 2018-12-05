# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import requests
import json
import time


def google_maps_encoder(
        latitude_origin, 
        longitude_origin, 
        latitude_destination,
        longitude_destination, 
        timestamp=int(time.time())+1, 
        token=None,
        mode='driving'
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

    api_url = "https://maps.googleapis.com/maps/api/distancematrix/json?"
    proto_url = api_url + "origins={0}&destinations={1}"
    proto_url += "&mode={2}&language=en-EN&sensor=false&departure_time={3}&trafic_model=pessimistic&key={4}"

    url = proto_url.format(
        in_url([longitude_origin, latitude_origin]),
        in_url([longitude_destination, latitude_destination]),
        mode,
        int(timestamp),
        token
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
        # using it with Google Maps (provided by default)
        # Create proxy instance
        data_path = r'api_proxy/'
        conn = sqlite3.connect(data_path + 'google_maps_itineraries.db')
        proxy = api_proxy.Proxy(
            conn=conn
        )
        # Make a request
        data = {
            'latitude_origin': 48.833405,
            'longitude_origin': 2.269831,
            'latitude_destination': 48.422811,
            'longitude_destination': 2.585593,
            'timestamp': time.time(),
            'token': token  # You need a valid token here
        }
        resp = proxy.get(**data)
    """
    def __init__(
        self,
        conn,
        encoder=google_maps_encoder,
        table='json',
        parameters=(
            'latitude_origin',
            'longitude_origin',
            'latitude_destination',
            'longitude_destination'
        ),
        tolerance=1e-4,
        auto_populate=False,
        verbose=False
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
        except:  # no such table:
            print('create table')
            empty = pd.DataFrame(
                index=list(parameters) + ['current', 'json']
            ).T
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
        return tuple(
            slice(
                kwargs[p] - self.tolerance[p],
                kwargs[p] + self.tolerance[p]
            ) for p in self.parameters
        )

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
        self.local.loc[self.indexer(kwargs) + (True, )] = resp
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
            responses = list(self.local.loc[slices + (slice(None), )])
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
        self.local = reset.set_index(
            list(self.parameters + ('current',))
        )['json']
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
        df.loc[df['current']].to_sql(
            self.table,
            self.conn,
            if_exists='append',
            index=False
        )

        df['current'] = False
        self.local = df.set_index(list(self.parameters + ('current',)))['json']
