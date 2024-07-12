declare QUETZAL_ROOT=../..

cd $QUETZAL_ROOT
# Build docker image
docker build -f api/GTFS_importer/Dockerfile  -t quetzal-gtfs-api:test .

echo ready

docker run -p 9000:8080 --env-file 'api/GTFS_importer/test.env' quetzal-gtfs-api:test


