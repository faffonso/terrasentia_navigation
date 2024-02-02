import json
import os

os.chdir('..')
def read_params_from_json(filename='./models/params.json', query_id=None):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)

            print(data)
            if query_id is not None and data['id'] != query_id:
                return None  # Return None if the queried id does not match

            result_dict = {
                'id': data['id'],
                'mean0': data['mean0'],
                'mean1': data['mean1'],
                'mean2': data['mean2'],
                'mean3': data['mean3'],
                'std0': data['std0'],
                'std1': data['std1'],
                'std2': data['std2'],
                'std3': data['std3']
            }

            return result_dict

    except (FileNotFoundError, json.decoder.JSONDecodeError, KeyError):
        return None  # Return None if there is an issue with file reading or data structure

# Example usage:
query_id = "02-02-2024_00-45-55"
result = read_params_from_json(query_id=query_id)

if result is not None:
    print("Query Result:")
    print(json.dumps(result, indent=4))
    print('-'*10)
    print(result['std1'])
else:
    print("No data found for the specified id.")
