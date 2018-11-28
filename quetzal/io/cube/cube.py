def head_string(links, trip_id):
    return  'LINE NAME="%s", ONEWAY=T, ' % trip_id

def stop_and_node_string(links, trip_id):
    line = links.loc[links['trip_id'] == trip_id]

    dicts = []
    for nodes in list(line['road_node_list']):
        dicts.append({'nodes': nodes[1:-1], 'stop': nodes[-1]})

    s = 'N=%s' %( str(line['road_a'].iloc[0]))
    for chunk in dicts:
        for node in chunk['nodes']:
            s += ', ' +  '-' + str(node)
        s += ', ' + str(chunk['stop']) 
    return s

def lin_string(links, trip_id, custom_head=head_string):
    return custom_head(links, trip_id) +  stop_and_node_string(links, trip_id)

class cubeModel():
    def __init__(self):
        pass

def to_lin(self, path_or_buf):
    lines = set(self.links['trip_id'])
    lin = ''
    for trip_id in lines:
        lin += lin_string(self.links, trip_id)
        lin += ' \n'
        
    with open(path_or_buf, 'w') as file:
        file.write(lin)
