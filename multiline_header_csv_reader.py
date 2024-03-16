# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:38:33 2024

@author: Frank
"""

import csv
import numpy as np
import os
import re

# Define the file path
file_name = "examplary_multiheader_csv1.csv"
file_path = os.path.join(os.getcwd(), file_name)


def read_csv_file(file_path, delimiter=',', country='US'):
    data = {}
    
    # Set delimiter and number format based on country
    if country == 'US':
        number_format = float
    elif country == 'EU':
        delimiter = ';'  # European CSV files often use semicolons as delimiters
        number_format = lambda x: float(x.replace(',', '.'))  # EU numbers use comma as decimal separator
    else:
        raise ValueError("Unsupported country. Please select 'US' or 'EU'.")
        
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = [header.strip() for header in next(reader)]  # Read and clean the headers
        second_row = [header.strip() for header in next(reader)]  # Read and clean the second row of headers
        next(reader)  # Skip the third empty row
        for header, sec_header in zip(headers[1:], second_row[1:]):
            #if header !='':
            data[header] = {sec_header: []}
            #else:
            #    data['wavl'] = {sec_header: []}
        for row in reader:
            for idx, value in enumerate(row[1:]):  # Skip the first element of the row (empty)
                if value != '':  # Check if the value is not empty
                    header = headers[idx + 1]  # Adjust index to match the correct header
                    sec_header = second_row[idx + 1]  # Adjust index to match the correct second header
                    data[header][sec_header].append(float(value))
    # Convert lists to numpy arrays
    for header, sub_dict in data.items():
        for sec_header, values in sub_dict.items():
            data[header][sec_header] = np.array(values)
    keys_list = list(data.keys())
    for key in keys_list:
        if key == '':
            data['wavl'] = data[key]
            del data[key]
    return data

# Read the CSV file
data = read_csv_file(file_path)

# Display the data
keys_list = list(data.keys())
print(keys_list)

for item in keys_list:
    print (item)
    try:
        float_values = [float(value) for value in item.split()[-2:]]
        print (float_values[0])
    except ValueError as e:
        print(f"Error processing key '{item}': {e}")
        continue
    

#for key in data.keys():
#    print(key)
#    print(data[key])


# # Display the data
# for header, values in data.items():
#     print(header)
#     #print(values['values'])
#     print()