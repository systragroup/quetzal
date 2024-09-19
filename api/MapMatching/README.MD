 
to test:


1) create a teste.env file with this.
```bash
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=ca-central
BUCKET_NAME=quetzal-api-bucket
AWS_LAMBDA_FUNCTION_MEMORY_SIZE=3000
```

2) put file on s3 quetzal-api-bucket :: test/

3) run this to build and run docker locally
```bash
./test_lambda.sh

```
4) in another terminal

```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"step": "preparation"}'

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"step": "mapmatching"}'

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"step": "merge"}'
```