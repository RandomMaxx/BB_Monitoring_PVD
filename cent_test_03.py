# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:53:58 2023

@author: Frank
"""

from centurionioptical import ScatteringMatrix
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import least_squares
import time

start_time = time.time()

spectral_dict ={'R':0, 'Rs':1, 'Rp':2, 'T':3, 'Ts':4, 'Tp':5}
spectral_type = 'R'


step_size=5
start_wavl=400
stop_wavl=850
wavl = np.arange(start_wavl,stop_wavl+step_size,step_size)
wavl_len = len(wavl)
print (wavl.size)

angle = 10
AOI = np.deg2rad(angle)

N_air =  np.linspace(1.00, 1.00, wavl.size)
N_substrate = np.linspace(1.58, 1.55, wavl.size)

N_low = np.linspace(1.48, 1.45, wavl.size)
k_low = np.linspace(1E-4, 1E-8, wavl.size)
N_high = np.linspace(2.25, 2.10, wavl.size)
k_high = np.linspace(1E-3, 1E-6, wavl.size)

N_c_low = np.vectorize(complex)(N_low, k_low)
N_c_high = np.vectorize(complex)(N_high, k_high)

qwot1 = 550/4/1.55
qwot2 = 550/4/2.25


shutter_delay = 0.5 #seconds
shutter_variation = 0.15 #seconds

#shutter_delay = 0.05 #seconds
#shutter_variation = 0.01 #seconds

scan_rate = 1 #every x seconds


#print (N_c_high[::25])

medium_front = [100, N_air, False, 0, 0, 0, 'Medium_Front']
medium_back = [100, N_air, False, 0, 0, 0, 'Medium_Back']
substrate = [1E9, N_substrate, True, 0, 0, 0, 'Substrate']
layer_high = [qwot2, N_c_high, False, 0, 0.3, 0.06, 'Layer_high']
layer_low = [qwot1, N_c_low, False, 0, 0.5, 0.1, 'Layer_low']

layer_list1 = [medium_front, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, substrate, medium_back]
#compute order is from incidence medium to substrate

RT1 = ScatteringMatrix.ComputeRT(layer_list1,wavl,AOI)
#plt.plot(wavl, RT1[0], label='system')
#plt.show()

#layer_list = [layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high]
layer_list = [layer_high, layer_low, layer_high, layer_low, layer_high]
#for layer growth from 1st layer substrate to last layer medium


#create noise list of rate
rate_growth_list = []
rate_noise_list = []
for idx, element in enumerate(layer_list):
    thickness = element[0]*3
    rate =  element[4]
    rate_noise = element[5]
    growth = np.arange(0,thickness*3, rate * scan_rate)
    rate = np.ones(len(growth))*rate*scan_rate*np.random.normal(1, rate_noise*scan_rate, len(growth))
    #rate = np.random.normal(element[4], element[5], len(growth))
    #print(noise)
    #print(rate)
    #rate = growth*noise
    rate_growth_list.append(growth)
    rate_noise_list.append(rate)
    
noise_percentage = 0.01
#noise_percentage = 0.001
noise_characteristic = [5,3,1,1]
noise_char_wavl =[400,430,450,850]
noise_level = np.interp(wavl, noise_char_wavl, noise_characteristic)*noise_percentage
spectral_noise = noise_level*np.random.normal(0, 1, wavl_len)
#print (spectral_noise)

#def fit_model():
#    return
    
#def cost_function(parameter, wavl, spectral_data): 
#    return fit_model(parameter,)

environment_1 = medium_front
environment_2 = [substrate, medium_back]

start_system = [medium_front, substrate, medium_back]

RT_start = ScatteringMatrix.ComputeRT(start_system,wavl,AOI)

spc = spectral_dict[spectral_type]

#'''
plt.ion()
fig, ax = plt.subplots()

line1, = ax.plot(wavl, RT_start[spc], label='Monitoring Singal', color='mediumspringgreen')
line2, = ax.plot(wavl, RT_start[spc], label='Preview', linestyle='--', color='darkgoldenrod')
line4, = ax.plot(wavl, RT_start[spc], label='Reference', linestyle='--', color='slategray')
line3, = ax.plot(wavl, RT_start[spc], label='Opt', linestyle='-.', color='darkcyan')
ax.set_title('BBM Monitoring Curve')
ax.set_xlabel('wavelength')
ax.set_ylabel('T&R')
ax.legend(loc='upper right')
ax.grid(True)
#'''
#building from substrate to medium
#layer_system = [medium_back, substrate]
layer_system = environment_2[::-1]
fit_system = environment_2[::-1]
reference_system = environment_2[::-1]

actual_delay = np.random.uniform(-1.0, 1.0, len(layer_list))*shutter_variation+shutter_delay

print (actual_delay)

for idx, element in enumerate(layer_list):
    new_layer = copy.copy(element)
    fit_layer = copy.copy(element)
    fit_thickness =  copy.copy(fit_layer[4])
    fit_layer[0] = fit_thickness
    layer_system.append(new_layer)
    fit_system.append(fit_layer)
    reference_system.append(element)
    layer_system.append(environment_1)
    fit_system.append(environment_1)
    reference_system.append(environment_1)
    RT_preview = ScatteringMatrix.ComputeRT(layer_system[::-1],wavl,AOI)
    RT_reference = ScatteringMatrix.ComputeRT(reference_system[::-1],wavl,AOI)
    line2.set_ydata(RT_preview[spc])
    line4.set_ydata(RT_reference[spc])
    target_thickness = element[0]
    upper_bound = new_layer[0]*3
    new_thickness = 0
    def fit_model_current_layer(params, x):
        layer_opt_thick = params
        #print (params)
        fit_system[-2][0] = layer_opt_thick
        RT_temp = ScatteringMatrix.ComputeRT(fit_system[::-1],wavl,AOI)
        return RT_temp[spc]
    
    def cost_function_current_layer(params, x, spectral_monitor):
        delta = fit_model_current_layer(params, x) - spectral_monitor
        return delta
    
    #print (element[6])
   # print (target_thickness)
    thick = 0
    for ix, rate in enumerate(rate_noise_list[idx]):
        thick = thick + rate
        if new_thickness > 2*target_thickness:
            print (new_thickness)
            print (thick)
            #print (thick)
            print ('error_break')
            break
        #print (element[4])
        layer_system[-2][0] = thick
        RT = ScatteringMatrix.ComputeRT(layer_system[::-1],wavl,AOI)
        spectral_data = RT[spc]+(noise_level*np.random.normal(0, 0.5, wavl_len))        
        result = least_squares(cost_function_current_layer, fit_thickness, bounds=(0, upper_bound), args=(wavl, spectral_data))
        new_thickness = result.x
        new_rate = new_thickness - fit_thickness
         
        #print (rate)

        #fit_system[-2][0] = new_thickness
        fit_thickness = new_thickness
        fit_system[-2][0] = new_thickness
        opt_spectral_data = ScatteringMatrix.ComputeRT(fit_system[::-1],wavl,AOI)
        
        if new_thickness + new_rate / scan_rate * shutter_delay >= target_thickness:
            #gives signal to close shutter
            #number of cycles after shutter signal
            no_last_cycles = actual_delay[idx] * scan_rate
            #print ('no of last cycles: ', no_last_cycles)
            addon_thickness = 0
            final_rate_multiplier = no_last_cycles
            for i in range(int(no_last_cycles)+1)[::-1]:
                #print ('round: ', i)
                if i == 0:
                    #print ('rate noise value last' , rate_noise_list[idx][ix+i+1])
                    current_rate = rate_noise_list[idx][ix+i+1]
                    addon_thickness = addon_thickness + current_rate * final_rate_multiplier
                    #print ('addon thicknes last' , addon_thickness)
                else:
                    final_rate_multiplier = final_rate_multiplier - 1
                    current_rate = rate_noise_list[idx][ix+i+1]
                    addon_thickness = addon_thickness + current_rate
                    #print ('rate noise value' , rate_noise_list[idx][ix+i+1])
                    #print ('addon thicknes' , addon_thickness)
                    
                #print ('final rate multiplier', final_rate_multiplier)
            thick = thick + addon_thickness
            layer_system[-2][0] = thick
            RT = ScatteringMatrix.ComputeRT(layer_system[::-1],wavl,AOI)
            spectral_data = RT[spc]+(noise_level*np.random.normal(0, 0.5, wavl_len))        
            result = least_squares(cost_function_current_layer, fit_thickness, bounds=(0, upper_bound), args=(wavl, spectral_data))
            new_thickness = result.x
            fit_system[-2][0] = new_thickness
            new_rate = new_thickness - fit_thickness
            #print('shutter break')
            #print ('final thickness', layer_system[-2][0])
            #print ('opt thickness', new_thickness)
            #print ('final fit thickness', fit_system[-2][0])
            opt_spectral_data = ScatteringMatrix.ComputeRT(fit_system[::-1],wavl,AOI)
            break

        
        line1.set_ydata(spectral_data)
        
        line3.set_ydata(opt_spectral_data[spc])
        ax.set_ylim(RT[spc].min()-0.05,RT[spc].max()+0.05)
        #layer_system.append(environment_1)
        plt.pause(0.01)    
    line2.set_ydata(RT[spc])
    layer_system = layer_system[:-1]
    fit_system = fit_system[:-1]
    reference_system = reference_system[:-1]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


plt.ioff()
plt.show()


layer_system.append(environment_1)
fit_system.append(environment_1)
reference_system.append(environment_1)

for element in fit_system[2:-1:]:
    print (element[0])
    

'''


plt.figure()
RT2 = ScatteringMatrix.ComputeRT(layer_system[::-1],wavl,AOI)
plt.plot(wavl, RT2[spc], label='Test2')
RT2 = ScatteringMatrix.ComputeRT(fit_system[::-1],wavl,AOI)
plt.plot(wavl, RT2[spc], label='Test2')
RT2 = ScatteringMatrix.ComputeRT(reference_system[::-1],wavl,AOI)
plt.plot(wavl, RT2[spc], label='Test2')

plt.show()
print (RT[3])
'''