import yaml

def find_parameters_by_id(target_id):
    with open('../models/config.yaml', 'r') as file:
        data = list(yaml.safe_load_all(file))

    if data:
        for entry in data:
            if entry.get('id') == target_id:
                return {
                    'mean0': entry.get('mean0'),
                    'mean1': entry.get('mean1'),
                    'mean2': entry.get('mean2'),
                    'mean3': entry.get('mean3'),
                    'std0': entry.get('std0'),
                    'std1': entry.get('std1'),
                    'std2': entry.get('std2'),
                    'std3': entry.get('std3'),
                }

    return None  # ID not found

# Example usage
target_id = '01-02-2024_15-03-08'
result = find_parameters_by_id(target_id)

if result:
    print(f"Parameters for ID '{target_id}':")
    for key, value in result.items():
        print(f"{key}: {value}")
else:
    print(f"ID '{target_id}' not found in the YAML file.")
