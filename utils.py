import json 

def load_jsonl_file(file_path):
    """
    Loads a JSONL file into a list of Python dictionaries.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary
              represents a JSON object from a line in the file.
    """
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Stripping leading/trailing whitespace is a good practice
                line = line.strip()
                if line:  # Ensure the line is not empty
                    data_list.append(json.loads(line))
        return data_list
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from a line. Details: {e}")
        return None