import os
import sys
import boto3
import json


session = boto3.Session()


def main():
    with open('.env') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

    # Create a Step Functions client
    stepfunctions_client = session.client('stepfunctions')

    # List state machines and find the ARN by name
    state_machine_name = os.environ["AWS_LAMBDA_FUNCTION_NAME"]
    response = stepfunctions_client.list_state_machines()
    state_machines = response['stateMachines']
    state_machine_arn = None
    for state_machine in state_machines:
        if state_machine['name'] == state_machine_name:
            state_machine_arn = state_machine['stateMachineArn']
            break
    if state_machine_arn:
        with open('step-functions.json', 'r') as f:
            json_data = json.load(f)
        # Define the updated state machine definition
        updated_definition = json_data

        # Convert the definition to a JSON string
        updated_definition_json = json.dumps(updated_definition)
        # Update the state machine definition
        response = stepfunctions_client.update_state_machine(
            stateMachineArn=state_machine_arn,
            definition=updated_definition_json
        )
        # Print the response
        if (response['ResponseMetadata']['HTTPStatusCode']):
            print('Done!')
        else:
            print(response['ResponseMetadata']['HTTPStatusCode'], 'something when wrong')

    else:
        print(f"No state machine found with the name '{state_machine_name}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: At least one argument is required.")
        print("Usage: python {name} model_folder".format(name=sys.argv[0]))
        sys.exit(1)

    source = os.path.dirname(os.path.abspath(__file__))
    quetzal_root = os.path.abspath(os.path.join(source, '../../..'))
    os.chdir(os.path.abspath(os.path.join(quetzal_root, sys.argv[1])))
    main()
