FROM public.ecr.aws/lambda/python:3.11
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
COPY ./ ./
CMD ["auth.handler"]
