FROM public.ecr.aws/lambda/python:3.8
COPY api/GTFS_importer/requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY api/GTFS_importer/ ./
#COPY quetzal/engine/road_model.py ./road_model.py
COPY quetzal ./quetzal
COPY syspy ./syspy


CMD ["main.handler"]