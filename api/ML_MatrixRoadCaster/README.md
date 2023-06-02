
# quetzal
## Copyright
(c) SYSTRA
## License
[CeCILL-B](LICENSE.md)
## Create docker

from quetzal root directory (not this one)

docker build -f api/ML_MatrixRoadCaster/Dockerfile -t ml_matrixroadcaster:latest .


## TEST

docker run -p 9000:8080  ml_matrixroadcaster

from another terminal window:

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'