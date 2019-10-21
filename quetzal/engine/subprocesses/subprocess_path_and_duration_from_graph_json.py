import sys
from pathlib import Path
path = Path(__file__)
libdir = str(path.parent.parent.parent.parent)
print(libdir)

sys.path.insert(0, libdir)

import json
from quetzal.engine.pathfinder import path_and_duration_from_graph_json
io_string = sys.argv[1]
kwargs = json.loads(io_string)
path_and_duration_from_graph_json(**kwargs)