FROM public.ecr.aws/lambda/python:3.12
COPY api/MapMatching/requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY api/MapMatching/ ./
COPY quetzal ./quetzal
COPY syspy ./syspy


CMD ["main.handler"]

