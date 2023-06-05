
# quetzal
## Copyright
(c) SYSTRA
## License
[CeCILL-B](LICENSE.md)
## Deploy

see instruction on AWS ECR
for step 2 however:
```bash
docker build -f api/ML_MatrixRoadCaster/Dockerfile -t ml_matrixroadcaster .
```
## TEST

1) create a test.env file at the root of this folder (with the DockerFile)
```bash
AWS_ACCESS_KEY_ID=[your access key]
AWS_SECRET_ACCESS_KEY=[your secret key]
AWS_REGION=ca-central
BUCKET_NAME=matrixroadcaster
AWS_LAMBDA_FUNCTION_MEMORY_SIZE=3000
```
2) Buld the Docker from quetzal root directory (not this one)
```bash
docker build -f api/ML_MatrixRoadCaster/Dockerfile -t ml_matrixroadcaster:latest .
```
3) run the docker with the environment variable
```bash
docker run -p 9000:8080 --env-file 'api/ML_MatrixRoadCaster/test.env' ml_matrixroadcaster 
```
4) from another terminal window:
```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"callID":"test"}'
```
