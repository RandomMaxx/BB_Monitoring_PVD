# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 00:32:24 2024

@author: Frank
"""

class dummy_opticalclass():
    def __init__(self, nk, wavl_range):
        self.nk = nk
        self.wavl_range = wavl_range
        self.len_wavl_range = len(wavl_range)

class CoatingMaterial:
    def __init__(self, name, Opticalclass, thickness_var=(False, 0,0,0,0), n_var=(False, 0,0,0,0), k_var=(False, 0,0,0,0), description = None, tf_stress = 100):
        self.name = name
        self.opticalclass = Opticalclass
        
        self.thick_var = thickness_var[0]
        self.thick_rand_factor = thickness_var[1]
        self.thick_rand_summand = thickness_var[2]
        self.thick_sys_factor = thickness_var[3]
        self.thick_sys_summand = thickness_var[4]
        
        self.n_var = n_var[0]
        self.n_rand_factor = n_var[1] #e.g. 0.03 = +/-3%
        self.n_rand_summand = n_var[2] 
        self.n_sys_factor = n_var[3] #e.g. factir 0.03 = +3%
        self.n_sys_summand = n_var[4]
        
        self.k_var = k_var[0]
        self.k_rand_factor = k_var[1]
        self.k_rand_summand = k_var[2]
        self.k_sys_factor = k_var[3]
        self.k_sys_summand = k_var[4]
        
        self.description = description
        self.tf_stress = tf_stress # in MPa
        
    @property
    def nk(self):
        return self.opticalclass.nk
    
    @property
    def wavl_range(self):
        return self.opticalclass.wavl_range
    
    @property
    def len_wavl_range(self):
        return self.opticalclass.wavl_range.len_wavl_range
    
    def set_wavelength(self, wavl_range):
        self.opticalclass.set_wavelength(wavl_range)
        
    def set_thick_var(self, variations):
        self.thick_var, self.thick_rand_factor, self.thick_rand_summand, self.thick_sys_factor, self.thick_sys_summand = variations
    
    def get_thick_var(self):
        return (self.thick_var, self.thick_rand_factor, self.thick_rand_summand, self.thick_sys_factor, self.thick_sys_summand) 
    
    def set_n_var(self, variations):
        self.n_var, self.n_rand_factor, self.n_rand_summand, self.n_sys_factor, self.n_sys_summand = variations
        
    def get_n_var(self):
        return (self.n_var, self.n_rand_factor, self.n_rand_summand, self.n_sys_factor, self.n_sys_summand)
    
    def set_k_var(self, variations):
        self.k_var, self.k_rand_factor, self.k_rand_summand, self.k_sys_factor, self.k_sys_summand = variations
    
    def get_k_var(self):
        return (self.k_var, self.k_rand_factor, self.k_rand_summand, self.k_sys_factor, self.k_sys_summand) 
    
    def _get_value_variation(self, value, rand_factor, rand_summand, sys_factor, sys_summand, var_type='normal'):
        if var_type == 'normal':
            # Generate random numbers using numpy.random.normal
            var_array_factor = np.random.normal(loc=0, scale=self.rand_factor, size=self.opticalclass.len_wavl_range)
            var_array_summand = np.random.normal(loc=0, scale=self.rand_summand, size=self.opticalclass.len_wavl_range)
        
        elif var_type == 'uniform':
            # Generate random numbers using numpy.random.uniform
            var_array_factor = np.random.uniform(-1*self.rand_factor, self.rand_factor, size=self.opticalclass.len_wavl_range)
            var_array_summand = np.random.uniform(-1*self.rand_summand, self.rand_summand, size=self.opticalclass.len_wavl_range)
            
        else:
            # Generate random numbers using numpy.random.normal 2-Sigma
            var_array_factor = np.random.normal(loc=0, scale=0.66*self.rand_factor, size=self.opticalclass.len_wavl_range)
            var_array_summand = np.random.normal(loc=0, scale=0.66*self.rand_summand, size=self.opticalclass.len_wavl_range)
        
        value = (value * (1 + var_array_factor) + var_array_summand) * (1 + self.sys_factor) + self.sys_summand
        return value
    
    def get_thickness_variation(self, thickness, var_type='normal'):
        return self._get_value_variation(thickness, self.thick_rand_factor, self.thick_rand_summand, self.thick_sys_factor, self.thick_sys_summand, var_type)
    
    def get_n_variation(self, nness, var_type='normal'):
        return self._get_value_variation(n, self.n_rand_factor, self.n_rand_summand, self.n_sys_factor, self.n_sys_summand, var_type)
    
    def get_k_variation(self, k, var_type='normal'):
        return self._get_value_variation(k, self.k_rand_factor, self.k_rand_summand, self.k_sys_factor, self.k_sys_summand, var_type)