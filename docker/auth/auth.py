'''
This lambnda function takes a cognito JWT token and a model as argument
And confirm that the user have access to this model
1) verify that the token is Valid and not expired with cognito
2) verify that the model (s3 bucket) is allowed in the iam role policies of the iam role.
'''

import jwt
import requests
import json
import os
import boto3
iam_client = boto3.client('iam')
# environment variable defined in Lambda function
USER_POOL_ID = os.environ['USER_POOL_ID']
APP_CLIENT_ID = os.environ['APP_CLIENT_ID']
REGION = os.environ['REGION']

# Decode and verify the Cognito JWT token
def verify_cognito_token(token):
    # Your Cognito User Pool ID

    # Fetch the JSON Web Key Set (JWKS) from Cognito
    jwks_url = f'https://cognito-idp.{REGION}.amazonaws.com/{USER_POOL_ID}/.well-known/jwks.json'
    jwks_response = requests.get(jwks_url)
    jwks_data = jwks_response.json()
    public_key = None
    for key in jwks_data['keys']:
        if key['kid'] == jwt.get_unverified_header(token)['kid']:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
            break
    if public_key:
        decoded_token = jwt.decode(
            token,
            public_key,
            algorithms=['RS256'],
            audience=APP_CLIENT_ID,
            issuer=f'https://cognito-idp.{REGION}.amazonaws.com/{USER_POOL_ID}',
            options={'verify_exp': True}
        )
        return decoded_token
    else:
        raise ValueError('Public key for token not found')

def get_policy_document(policy_arn):

    policy = iam_client.get_policy(PolicyArn=policy_arn)
    policy_version = iam_client.get_policy_version(
        PolicyArn = policy_arn, 
        VersionId = policy['Policy']['DefaultVersionId']
    )

    return policy_version['PolicyVersion']['Document']['Statement']
def get_policies_from_role(role_name):

     # Get the role's policies
    response = iam_client.list_attached_role_policies(RoleName=role_name)

    # Extract and print the attached policies
    attached_policies = response['AttachedPolicies']
    

    policies = []
    for pol in attached_policies:
        res = get_policy_document(pol['PolicyArn'])
        policies.append(res)
    return policies

def handler(event, context):
    print("event")
    print(event)
    token = event['authorization']
    model = event['model']

    '''
    Validate the incoming token and produce the principal user identifier
    associated with the token. This can be accomplished in a number of ways:

    1. Call out to the OAuth provider
    2. Decode a JWT token inline
    3. Lookup in a self-managed DB
    '''
        
    try:
        claims = verify_cognito_token(token)
        print('Token is valid:', claims)
    except jwt.ExpiredSignatureError:
          raise Exception('Token has expired')
    except Exception as e:
        raise Exception('Token validation failed:', e)

    role_arn = claims['cognito:roles'][0]
    role_name = role_arn.split('/')[-1]

    try:
        policies = get_policies_from_role(role_name)
    except Exception as e:
        raise Exception('error listing role policies:', e)
    for policy in policies:
        print(policy[1]['Resource'])
        if (policy[1]['Effect'] == 'Allow') and (model in str(policy[1]['Resource'])):
            print('Allowed')
            break

    else:
        raise Exception('Access Denied')
    
    return event
