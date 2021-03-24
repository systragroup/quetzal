import json
from urllib import parse

import pandas as pd
import requests
import urllib3
from shapely import geometry
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ItsimClient:
    def __init__(
        self,
        api_root="https://sws-itsim.systrasaas.rec/api/",
        browser=None,
        project_id=None,
        scenario_id=None,
        username=None,
        password=None
    ):
        self.api_root = api_root
        self.browser = browser
        self.project_id = project_id
        self.scenario_id = scenario_id

        self.username = username
        self.password = password

        self.user_token = get_user_token(browser, username=username, password=password)

        scenario_url = api_root + 'projects/{}/scenarios/{}'.format(project_id, scenario_id)
        scenario = get(scenario_url, self.user_token).json()
        self.feed_id = scenario['data']['attributes']['feed_id']
        self.scenario_feed_url = scenario_url + r'/feeds/{}'.format(self.feed_id)

    def refresh_token(self):
        self.user_token = get_user_token(
            self.browser,
            username=self.username,
            password=self.password
        )

    def get_stops(self, **kwargs):
        self.stops = get_stops(self.scenario_feed_url, self.user_token, **kwargs)

    def get_routes(self, **kwargs):
        self.routes = get_routes(self.scenario_feed_url, self.user_token, **kwargs)

    def get_patterns(self, **kwargs):
        self.patterns = get_patterns(
            self.scenario_feed_url,
            self.user_token,
            routes_df=self.routes,
            stops_df=self.stops,
            **kwargs
        )

    def get_trips(self, **kwargs):
        self.refresh_token()
        self.trips = get_trips(
            self.scenario_feed_url,
            self.user_token,
            routes_df=self.routes,
            **kwargs
        )

    def get_timetables(self, **kwargs):
        self.refresh_token()
        self.timetables = get_timetables(
            self.scenario_feed_url,
            self.user_token,
            routes_df=self.routes,
            trips_df=self.trips,
            **kwargs
        )

    def get_services(self, **kwargs):
        self.refresh_token()
        self.services = get_services(
            self.scenario_feed_url,
            self.user_token,
            **kwargs
        )

    def get_stop_times(self, **kwargs):
        self.refresh_token()
        self.stop_times = get_stop_times(
            self.scenario_feed_url,
            self.user_token,
            routes_df=self.routes,
            trips_df=self.trips,
            **kwargs
        )

    def get_timetables(self, **kwargs):
        self.timetables = get_timetables(
            scenario_feed_url=self.scenario_feed_url,
            user_token=self.user_token,
            routes_df=self.routes,
            trips_df=self.trips,
            **kwargs
        )

    def get_services(self, **kwargs):
        self.services = get_services(
            scenario_feed_url=self.scenario_feed_url,
            user_token=self.user_token,
            **kwargs
        )

    def get_trips(self, **kwargs):
        self.trips = get_trips(
            scenario_feed_url=self.scenario_feed_url,
            user_token=self.user_token,
            routes_df=self.routes,
            **kwargs
        )


def get_user_token(browser, username=None, password=None):
    browser.get('https://itsim.systra.com')
    try:
        user_token = browser.get_cookie('user-token')['value']
    except TypeError:
        # 'NoneType' object is not subscriptable
        # browser is not logged in,
        usernameform = browser.find_element_by_id("loginForm:username")
        usernameform.send_keys(username)  # will fail if username is None
        passwordform = browser.find_element_by_id("loginForm:password")
        passwordform.send_keys(password)
        submit = browser.find_element_by_id("loginForm:loginButton")
        submit.click()
        user_token = browser.get_cookie('user-token')['value']
    return parse.unquote(user_token)


def after_request(request):
    if request.status_code // 100 != 2:
        print(request.url)
        print('Response has not 2XX status code')
        print(request.status_code)
        print(request.text)
    return request


def get(url, user_token=None):
    headers = {
        'Content-Type': 'application/json',
        'user-token': user_token
    }

    request = requests.get(url, headers=headers, verify=False)
    return after_request(request)


def put(url, user_token=None, data={}):
    headers = {
        'Content-Type': 'application/json',
        'user-token': user_token
    }
    request = requests.put(url, headers=headers, data=json.dumps(data), verify=False)
    return after_request(request)


def get_feed_tables(
    project_id,
    scenario_id,
    add_geometry=False,
    user_token=None,
    api_root="https://sws-itsim.systrasaas.rec/api/"
):
    scenario_url = api_root + 'projects/{}/scenarios/{}'.format(project_id, scenario_id)
    # print(scenario_url)
    scenario = get(scenario_url, user_token).json()
    feed_id = scenario['data']['attributes']['feed_id']
    scenario_feed_url = scenario_url + r'/feeds/{}'.format(feed_id)

    # stops and routes
    # print('Getting stops and routes')
    routes_df = get_routes(scenario_feed_url, user_token)
    stops_df = get_stops(scenario_feed_url, user_token, add_geometry=add_geometry)

    # Get patterns
    patterns_df = get_patterns(
        scenario_feed_url,
        user_token,
        routes_df,
        stops_df,
        add_geometry=add_geometry
    )

    # Get trips:
    trips_df = get_trips(scenario_feed_url, user_token, routes_df)

    # Get stop_times
    stop_times_df = get_stop_times(scenario_feed_url, user_token, routes_df, trips_df)

    # Get services
    services_df = get_services(scenario_feed_url, user_token)

    # Get timetables
    timetables_df = get_timetables(scenario_feed_url, user_token, routes_df, trips_df)
    return stops_df, routes_df, patterns_df, trips_df, stop_times_df, services_df, timetables_df


def get_routes(scenario_feed_url, user_token):
    routes = get(scenario_feed_url + '/routes', user_token).json()
    routes_df = pd.DataFrame({s['id']: s['attributes'] for s in routes['data']}).T
    return routes_df


def get_stops(scenario_feed_url, user_token, add_geometry=False):
    stops = get(scenario_feed_url + '/stops', user_token).json()
    stops_df = pd.DataFrame({s['id']: s['attributes'] for s in stops['data']}).T
    if add_geometry:
        stops_df['geometry'] = stops_df.apply(
            lambda r: geometry.Point(r['stop_lon'], r['stop_lat']),
            axis=1
        )
    return stops_df


def get_patterns(scenario_feed_url, user_token, routes_df, stops_df, add_geometry=False):
    pattern_data = []
    for route_id in tqdm(routes_df.index, desc='patterns'):
        r = get(scenario_feed_url + '/routes/{}/patterns'.format(route_id), user_token)
        try:
            pattern_data += r.json()['data']
        except KeyError:
            pass
    patterns_df = pd.DataFrame({s['id']: s['attributes'] for s in pattern_data}).T

    if add_geometry:
        def straight_geometry(pattern_stops):
            return geometry.LineString(list(stops_df.loc[pattern_stops]['geometry']))
        patterns_df['geometry'] = patterns_df['pattern_stops'].apply(straight_geometry)
    return patterns_df


def get_trips(scenario_feed_url, user_token, routes_df):
    trip_data = []
    for route_id in tqdm(routes_df.index, desc='trips'):
        r = get(scenario_feed_url + '/routes/{}/trips'.format(route_id), user_token)
        try:
            trip_data += r.json()['data']
        except KeyError:
            pass
    key_tables = {'patterns': 'pattern_uid', 'routes': 'route_uid'}
    for trip in trip_data:
        relationships = trip['relationships'].copy()
        for key, value in relationships.items():
            trip['attributes'][key_tables[key]] = value['data']['id']
            trip['relationships'].pop(key)
    trips_df = pd.DataFrame({s['id']: s['attributes'] for s in trip_data}).T
    return trips_df


def get_services(scenario_feed_url, user_token):
    services = get(scenario_feed_url + r'/services', user_token).json()
    services_df = pd.DataFrame({s['id']: s['attributes'] for s in services['data']}).T
    return services_df


def get_stop_times(scenario_feed_url, user_token, routes_df, trips_df):
    stop_times_data = []
    for route_id in tqdm(routes_df.index, desc='stop_times'):
        for trip_id in trips_df.loc[trips_df['route_uid'] == route_id].index:
            r = get(
                scenario_feed_url + '/routes/{}/trips/{}/stop_times'.format(route_id, trip_id), user_token
            )
            try:
                stop_times_data += r.json()['data']
            except KeyError:
                pass
    stop_times_df = pd.DataFrame({s['id']: s['attributes'] for s in stop_times_data}).T
    return stop_times_df


def get_timetables(scenario_feed_url, user_token, routes_df, trips_df):
    timetables_data = []
    for route_id in tqdm(routes_df.index, desc='timetables'):
        for trip_id in trips_df.loc[trips_df['route_uid'] == route_id].index:
            r = get(
                scenario_feed_url + '/routes/{}/trips/{}/timetables'.format(route_id, trip_id), user_token
            )
            try:
                timetables_data += r.json()['data']
            except KeyError:
                pass
    timetables_df = pd.DataFrame({s['id']: s['attributes'] for s in timetables_data}).T
    return timetables_df


def request_average_headway(scenario_feed_url, user_token, route_id, pattern_id, headway_request_dict):
    """
    Example of headway_request_dict:
    {
        "data":[
            {
                "type":"headway_request",
                "attributes":
                {
                    "typical_day": 'SCO',
                    "time_ranges": [[70000, 82929]]
                }
            }
        ]
    }
    """
    resp = put(
        scenario_feed_url + r'/routes/{}/patterns/{}/actions/avg_headway/invoke'.format(route_id, pattern_id),
        user_token=user_token,
        data=headway_request_dict
    )
    return resp.json()
