# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:56:33 2024

@author: Frank
"""

import numpy as np
import csv

from pathlib import Path

# Specify the path to the folder containing CSV files
folder_path = Path('/path/to/your/folder')

# Iterate through CSV files in the folder
for file_path in folder_path.glob('*.csv'):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)

        # Process each row in the CSV file
        for row in csv_reader:
            print(row)

def read_csv_with_two_headers(file_path):
    with open(file_path, 'r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        
        # Read the first two rows as headers
        headers_row1 = next(csv_reader)
        headers_row2 = next(csv_reader)
        
        # Combine headers from both rows as tuples
        combined_headers = [(header1, header2) for header1, header2 in zip(headers_row1, headers_row2)]
        
        # Read the rest of the data
        data = list(csv_reader)
        
        # Create a dictionary where keys are tuples of combined headers,
        # and values are lists of NumPy arrays
        result = {}
        
        for row in data:
            for i, value in enumerate(row):
                header_tuple = combined_headers[i]
                if header_tuple not in result:
                    result[header_tuple] = []
                result[header_tuple].append(np.array(value, dtype=np.float64))
        
        # Convert lists of arrays to arrays
        for key, value in result.items():
            result[key] = np.stack(value)
        
        return result

# Example usage
file_path = 'test_1.csv'
result = read_csv_with_two_headers(file_path)

# Convert dictionary values to a single multi-dimensional NumPy array
result_array = np.stack(list(result.values()), axis=-1)

print(result_array)
print(list(result.keys()))
print(list(result.keys())[0])
