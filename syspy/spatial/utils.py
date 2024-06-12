import math
def haversine(coord1: object, coord2: object)->float:
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    
    return meters

def get_acf_distance(x:list,reverse:bool=False)->list:
    # return dist in meters between 2 points in CRS 4326 (degs)
    # inputs : [(lat,lon), (lat,lon)]. or [(y,x),(y,x)]
    # if reverse : [(lon,lat), (lon,lat)]. or [(x,y),(x,y)]
    # ex: df['geometry'].apply(lambda x: get_acf_distance([x.coords[0],x.coords[-1]],True))
    if reverse:
        return haversine(x[0][::-1],x[1][::-1])
    else:
        return haversine(x[0],x[1])
    

def get_epsg(lat: float, lon: float) -> int:
    '''
    lat, lon or y, x
    return EPSG in meter for a given (lat,lon)
    lat is north south 
    lon is est west
    '''
    return int(32700 - round((45 + lat) / 90, 0) * 100 + round((183 + lon) / 6, 0))