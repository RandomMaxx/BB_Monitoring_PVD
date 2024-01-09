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
#print (wavl.size)

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

qwot1 = 550/4/1.55/1
qwot2 = 550/4/2.25/1
print(qwot1)
print(qwot2)

shutter_delay = 0.5 #seconds
shutter_variation = 0.15 #seconds

#shutter_delay = 0.05 #seconds
#shutter_variation = 0.01 #seconds

scan_rate = 1 #every x seconds


#print (N_c_high[::25])

#Layer = [0=thickness in nm, 1=nk, 2=incoherent, 3=roughnes in Angstrom, 4=deposition rate in nm/s, 5=rate variation in nm/s, 6=Ihomogeneity]
#6 = [0=Ihomogeneity (True/False), 1=refractive n-factor, 2=direction (0=up, 1 =down), 3=initial thickness in nm, 4=gradient thickness in nm, ]
medium_front = [100, N_air, False, 0, 0, 0, 'Medium_Front']
medium_back = [100, N_air, False, 0, 0, 0, 'Medium_Back']
substrate = [1E9, N_substrate, True, 0, 0, 0, 'Substrate']
layer_high = [qwot2, N_c_high, False, 0, 0.6, 0.06, [True, 0.97, 0, 0.0, 3.0], 'Layer_high']
layer_low = [qwot1, N_c_low, False, 0, 1.0, 0.1, [True, 0.97, 0, 4.0, 3.0],'Layer_low']

layer_list1 = [medium_front, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, substrate, medium_back]
#compute order is from incidence medium to substrate

RT1 = ScatteringMatrix.ComputeRT(layer_list1,wavl,AOI)
#plt.plot(wavl, RT1[0], label='system')
#plt.show()

#layer_list = [layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high, layer_low, layer_high]
layer_list_reference = [layer_high, layer_low, layer_high, layer_low, layer_high]
#for layer growth from 1st layer substrate to last layer medium

layer_list_physical = []
layer_list_ref_physical = []

#create "physical stack"
for element in layer_list_reference:
    if element[6][0] == True:
        no_of_sub_layers = int(np.ceil(element[6][4]**0.4)) + 3
        if element[6][2] == 0:
            n_steps = np.linspace(element[6][1], 1, no_of_sub_layers)
        else:
            n_steps = np.linspace(1, element[6][1], no_of_sub_layers)
        
        if element[6][3] > 0:
            print ('initial1', element[6][3])
            n_steps_gradient = n_steps[1:-1]
            no_of_sub_layers = len(n_steps_gradient)
            layer_list_physical.append(list(range(no_of_sub_layers + 2)))
            new_layer = copy.copy(element)
            new_layer[0] = element[6][3]
            new_layer[1] = element[1]*n_steps[0]
            layer_list_physical[-1][0] = new_layer
            idx_addon = 1
        else:
            print ('initial2', element[6][3])
            n_steps_gradient = n_steps[0:-1]
            no_of_sub_layers = len(n_steps_gradient)
            layer_list_physical.append(list(range(no_of_sub_layers + 1)))
            idx_addon = 0

        for idx, factor in enumerate(n_steps_gradient):
            new_layer = copy.copy(element)
            new_layer[0] = element[6][4] / no_of_sub_layers
            new_layer[1] = element[1] * factor
            layer_list_physical[-1][idx+idx_addon] = new_layer
            print (new_layer[0], factor)
            
        new_layer = copy.copy(element)
        #last_thick = element[0]-(element[6][3]+element[6][4])
        #if last_thick <= 0:
        #    last_thick = 50
        last_thick = element[0]
        new_layer[0] = last_thick*2
        new_layer[1] = element[1] * n_steps[-1]
        layer_list_physical[-1][-1] = new_layer
    else:
        layer_list_physical.append([element])

'''
#create "physical reference stack"
for element in layer_list_reference:
    if element[6][0] == True:
        no_of_sub_layers = int(np.ceil(element[6][4]**0.4)) + 3
        if element[6][2] == 0:
            n_steps = np.linspace(element[6][1], 1, no_of_sub_layers)
        else:
            n_steps = np.linspace(1, element[6][1], no_of_sub_layers)
        #sub_layer_list = []
        if element[6][3] > 0:
            print ('initial3', element[6][3])
            n_steps_gradient = n_steps[1:-1]
            no_of_sub_layers = len(n_steps_gradient)
            sub_layer_list = (list(range(no_of_sub_layers + 2)))
            new_layer = copy.copy(element)
            new_layer[0] = element[6][3]
            new_layer[1] = element[1]*n_steps[0]
            sub_layer_list[-1] = new_layer
            idx_addon = 1
        else:
            print ('initial4', element[6][3])
            n_steps_gradient = n_steps[0:-1]
            no_of_sub_layers = len(n_steps_gradient)
            sub_layer_list = (list(range(no_of_sub_layers + 1)))
            idx_addon = 0
        
        print ('length sublayer', len(sub_layer_list))
        for idx, factor in enumerate(n_steps_gradient):
            print(idx+idx_addon)
            new_layer = copy.copy(element)
            new_layer[0] = element[6][4] / no_of_sub_layers
            new_layer[1] = element[1] * factor
            sub_layer_list[idx+idx_addon] = new_layer
            print (new_layer[0], factor)
            
        new_layer = copy.copy(element)
        last_thick = element[0]
        new_layer[0] = last_thick*2
        new_layer[1] = element[1] * n_steps[-1]
        sub_layer_list[-1] = new_layer
        
        temp_list = []
        thick = 0
        for idx, sub_layer in sub_layer_list:
            if thick <= element[0]:
                if <= element[0]:
                    temp_list.append[sub_layer]
                    thick = thick + element[0]:
                else:
                    
            else:
                last_thick = element[0]-(element[6][3]+element[6][4])
                if last_thick <= 0:   
            thick = thick + sub_layer[idx][0]    
                
        layer_list_ref_physical.append([sub_layer_list])
    else:
        layer_list_ref_physical.append([element])
'''

for idx,element in enumerate(layer_list_physical):
    print ('Length layer_System', len(layer_list_ref_physical))
    print ('Length Sublayer',len(element))

#for element in layer_list_physical:
#    for item in element:
#        print('lenght of Layer-List_Physical ', item[0])
        
#print (layer_list_sim)

#create noise list of rate
#rate_growth_list = []
rate_noise_list = []
rate_stability = 5
for idx, element in enumerate(layer_list_reference):
    thickness = element[0]*3
    rate =  element[4]
    rate_noise = element[5]
    growth = np.arange(0,thickness*3, rate * scan_rate)

    growth_stab = np.arange(0,thickness*3, rate * scan_rate * rate_stability)

    rate = np.ones(len(growth_stab))*rate*scan_rate*np.random.normal(1, rate_noise*scan_rate, len(growth_stab))
    rate = np.interp(growth, growth_stab, rate)
    #rate = np.random.normal(element[4], element[5], len(growth))
    #print(noise)
    #print(rate)
    #rate = growth*noise
    #rate_growth_list.append(growth)
    rate_noise_list.append(rate)
    
#noise_percentage = 0.01 # Noise level in percent
noise_percentage = 0.001
noise_characteristic = [5,3,1,1] #Factors for noise level across the wavlength range
noise_char_wavl =[400,430,450,850] #wavl for characteristic factors for noise level across the wavlength range
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

actual_delay = np.random.uniform(-1.0, 1.0, len(layer_list_reference))*shutter_variation+shutter_delay

print (actual_delay)

rate_averaging = 5 #no of cycles


for idx, element in enumerate(layer_list_reference):
    #print ('top_level ', idx)
    
    reference_system.append(element)
    reference_system.append(environment_1)
    
    fit_layer = copy.copy(element)
    fit_thickness =  copy.copy(fit_layer[4]) #initial thickness for fitting system using the initial rate
    fit_layer[0] = fit_thickness
    fit_system.append(fit_layer)
    fit_system.append(environment_1)
    
    rate_list = [fit_thickness] * rate_averaging
    #print (rate_list)
    #new_layer = copy.copy(element)
    sub_idx = 0
    new_layer = copy.copy(layer_list_physical[idx][sub_idx]) #copy of the physical layer stack
    layer_system.append(new_layer)
    layer_system.append(environment_1)
     
    RT_preview = ScatteringMatrix.ComputeRT(layer_system[::-1],wavl,AOI)
    RT_reference = ScatteringMatrix.ComputeRT(reference_system[::-1],wavl,AOI)
    line2.set_ydata(RT_preview[spc])
    line4.set_ydata(RT_reference[spc])

    target_thickness = layer_list_reference[idx][0]
    #print ('target thickness', target_thickness)
    upper_bound = layer_list_reference[idx][0]*3
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
        #print ("subidx ", sub_idx)
        thick = thick + rate
        if new_thickness > 2*target_thickness:
            #print (new_thickness)
            #print (thick)
            #print (thick)
            print ('error_break')
            break
        #print (element[4])
        
        if thick >  layer_list_physical[idx][sub_idx][0] and (sub_idx+1) < len(layer_list_physical[idx]):
            #print ('length sublayer', len(layer_list_physical[idx]))
            #print ('1', thick)
            layer_system[-2][0] = layer_list_physical[idx][sub_idx][0]
            thick = thick - layer_list_physical[idx][sub_idx][0]
            #print ('2', thick)
            if (sub_idx+1) < len(layer_list_physical[idx]):
                #print ('3', thick)
                sub_idx = sub_idx + 1
                new_layer = copy.copy(layer_list_physical[idx][sub_idx])
                new_layer[0] = thick
                layer_system.insert(-1, new_layer)
                #layer_system = layer_system[:-1]
                #layer_system.append(new_layer)
                #layer_system.append(environment_1)
                #layer_system[-2][0] = thick
            #print ('Sub_IDX', sub_idx)
            #print ("thickness ", thick)
        else:
            #print ('else', thick)
            #print ('Sub_IDX', sub_idx)
            layer_system[-2][0] = thick
        RT = ScatteringMatrix.ComputeRT(layer_system[::-1],wavl,AOI)
        spectral_data = RT[spc]+(noise_level*np.random.normal(0, 0.5, wavl_len))        
        result = least_squares(cost_function_current_layer, fit_thickness, bounds=(0, upper_bound), args=(wavl, spectral_data))
        new_thickness = result.x[0]
        new_rate = new_thickness - fit_thickness
        rate_list.pop(0) 
        rate_list.append(new_rate)
        new_rate_average = np.mean(rate_list)

        #fit_system[-2][0] = new_thickness
        fit_thickness = new_thickness
        #print(fit_thickness)
        #print (new_rate_average)
        fit_system[-2][0] = new_thickness
        opt_spectral_data = ScatteringMatrix.ComputeRT(fit_system[::-1],wavl,AOI)
        
        if new_thickness + new_rate_average / scan_rate * shutter_delay >= target_thickness:
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
            new_thickness = result.x[0]
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
        plt.pause(0.001)    
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