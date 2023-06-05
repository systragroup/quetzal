
# quetzal
## Copyright
(c) SYSTRA
## License
[CeCILL-B](LICENSE.md)
## Create docker

from quetzal root directory (not this one)

docker build -f api/ML_MatrixRoadCaster/Dockerfile -t ml_matrixroadcaster:latest .



## TEST

1) create a test.env file
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=ca-central
BUCKET_NAME=matrixroadcaster
AWS_LAMBDA_FUNCTION_MEMORY_SIZE=3000

docker run -p 9000:8080 --env-file 'api/ML_MatrixRoadCaster/test.env' ml_matrixroadcaster 

from another terminal window:

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"callID":"test"}'