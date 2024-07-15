FROM public.ecr.aws/lambda/python:3.12
COPY api/ML_MatrixRoadCaster/requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY api/ML_MatrixRoadCaster/ ./
#COPY quetzal/engine/road_model.py ./road_model.py
COPY quetzal ./quetzal
COPY syspy ./syspy

CMD ["main.handler"]