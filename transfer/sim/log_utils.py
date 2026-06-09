# Copyright 2026 Zachary Olkin. All rights reserved.

import csv
import os
import re
from datetime import datetime

import numpy as np


def find_most_recent_timestamped_folder(base_path):
    """
    Finds the path of the most recent folder named with a YYYY-MM-DD-HH-MM-SS timestamp
    within a specified base path.

    Args:
      base_path (str): The directory to search within.

    Returns:
      str: The full path to the most recent timestamped folder, or None if none found.
    """
    most_recent_folder = None
    latest_timestamp = None

    # Regular expression to match the YYYY-MM-DD-HH-MM-SS format
    timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

    try:
        # List all entries in the base directory
        entries = os.listdir(base_path)

        for entry in entries:
            entry_path = os.path.join(base_path, entry)

            # Check if the entry is a directory and matches the timestamp pattern
            if os.path.isdir(entry_path) and timestamp_pattern.match(entry):
                try:
                    # Parse the timestamp from the folder name
                    folder_timestamp = datetime.strptime(entry, "%Y-%m-%d-%H-%M-%S")

                    # If this is the first timestamped folder found, or if it's more recent
                    if latest_timestamp is None or folder_timestamp > latest_timestamp:
                        latest_timestamp = folder_timestamp
                        most_recent_folder = entry_path

                except ValueError:
                    # This handles cases where a folder name matches the pattern but isn't
                    # a valid date/time string (unlikely with the previous script, but good practice)
                    print(f"Warning: Directory '{entry}' matches pattern but has invalid timestamp.")
                    pass  # Skip this directory

    except FileNotFoundError:
        print(f"Error: Base path '{base_path}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return most_recent_folder


def extract_data(filepath, config):
    data_structure = config.get("data_structure")
    print("\nData structure from config:")
    for item in data_structure:
        print(f"  {item['name']}: length {item['length']}")

    extracted_data_lists = {item["name"]: [] for item in data_structure if "name" in item}
    print("\nInitialized data lists:")
    for name in extracted_data_lists:
        print(f"  {name}")

    with open(filepath) as f:
        csv_reader = csv.reader(f)

        for row_count, row in enumerate(csv_reader):
            numeric_row = []
            for item in row:
                numeric_row.append(float(item))

            current_index = 0
            for item in data_structure:
                name = item.get("name")
                length = item.get("length")
                component_data = numeric_row[current_index : current_index + length]
                extracted_data_lists[name].append(component_data)
                current_index += length

        # Convert lists of data to NumPy arrays
        extracted_data_arrays = {}
        for name, data_list in extracted_data_lists.items():
            if data_list:  # Only create array if there is data
                extracted_data_arrays[name] = np.array(data_list)
                print(f"\nLoaded data for {name}:")
                print(f"  Shape: {extracted_data_arrays[name].shape}")
            else:  # Create empty array if no data was collected for this component
                # Determine the shape based on the config length
                component_length = next((item["length"] for item in data_structure if item.get("name") == name), 0)
                extracted_data_arrays[name] = np.empty((0, component_length))
                print(f"\nNo data found for {name}, created empty array with shape (0, {component_length})")

        return extracted_data_arrays
