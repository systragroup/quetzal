# Adapted and simplified from https://github.com/GeographicaGS/GeoSmoothing

import numpy as np
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, Polygon


class Splines():
    def compSplineKnots(self, x, y, s, k, nest=-1):
        """
        Computed with Scipy splprep. Find the B-spline representation of
        an N-dimensional curve.
        Spline parameters:
        :s - smoothness parameter
        :k - spline order
        :nest - estimate of number of knots needed (-1 = maximal)
        """

        tck_u, fp, ier, msg = splprep([x, y], s=s, k=k, nest=nest, full_output=1)

        if ier > 0:
            print("{}. ier={}".format(msg, ier))
        return(tck_u, fp)

    def compSplineEv(self, x, tck, zoom=10):
        """
        Computed with Scipy splev. Given the knots and coefficients of
        a B-spline representation, evaluate the value of the smoothing
        polynomial and its derivatives
        Parameters:
        :tck - A tuple (t,c,k) containing the vector of knots,
             the B-spline coefficients, and the degree of the spline.
        """
        n_coords = len(x)
        n_len = n_coords * zoom
        x_ip, y_ip = splev(np.linspace(0, 1, n_len), tck)

        return(x_ip, y_ip)


class GeoSmoothing():
    def __init__(self, spl_smpar=0, spl_order=2, verbose=True):
        """
        spl_smpar: smoothness parameter
        spl_order: spline order
        """
        self.spl_smpar = spl_smpar
        self.spl_order = spl_order
        self.verbose = verbose

    def get_coordinates(self, geom):
        """
        Getting x,y coordinates from geometry...
        """
        if isinstance(geom, LineString):
            x = np.array(geom.coords.xy[0])
            y = np.array(geom.coords.xy[1])

        elif isinstance(geom, Polygon):
            x = np.array(geom.exterior.coords.xy[0])
            y = np.array(geom.exterior.coords.xy[1])
        return(x, y)

    def geom_from_coords(self, coords_ip, geom):
        """
        """
        if isinstance(geom, LineString):
            geom_ip = LineString(coords_ip.T)

        elif isinstance(geom, Polygon):
            geom_ip = Polygon(coords_ip.T)
        return geom_ip

    def smooth_geom(self, geom):
        """
        Run smoothing geometries
        """
        x, y = self.get_coordinates(geom)

        spl = Splines()

        tck_u, fp = spl.compSplineKnots(x, y, self.spl_smpar, self.spl_order)
        x_ip, y_ip = spl.compSplineEv(x, tck_u[0])

        coords_ip = np.array([x_ip, y_ip])
        return self.geom_from_coords(coords_ip, geom)
