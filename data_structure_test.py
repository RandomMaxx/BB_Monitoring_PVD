Created on Sat Jan 13 20:20:00 2024

@author: RandomMaxx
"""

import numpy as np

class DataStructure():
    def __init__(self):
        self.data_dict = {}

    def add_array(self, key_str, key_float, array):
        key = (key_str, key_float)
        self.data_dict[key] = array

    def get_array(self, key_str, key_float):
        key = (key_str, key_float)
        return self.data_dict.get(key, None)

    def keys(self):
        return list(self.data_dict.keys())
        
    def values(self):
        #return np.array(list(self.data_dict.values()))
        return np.stack(list(self.data_dict.values()), axis=0)

# Example usage:
# Create an instance of the DataStructure class
data_structure = DataStructure()

# Create a NumPy array
example_array = np.arange(2,11,1)
example_array2 = np.arange(1,10,1)

# Add the array to the data structure with a specific key
data_structure.add_array("example_key", 1.5, example_array)
data_structure.add_array("test_key", 5.5, example_array2)

# Retrieve the array using the key
result_array = data_structure.get_array("example_key", 1.5)

# Print the result
print(result_array)

a= data_structure.keys()
print (a)

b= data_structure.values()
print (b)
