FROM public.ecr.aws/lambda/python:3.11.2023.08.17.16
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
COPY ./ ./
CMD ["auth.handler"]
