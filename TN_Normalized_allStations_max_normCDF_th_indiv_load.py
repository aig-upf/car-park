#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import integrate
from car_park_functions import *


# In[2]:


available_parkings = ['Vilanova', 'SantSadurni', 'SantBoi', 'QuatreCamins',
                      'Cerdanyola','Granollers','Martorell','Mollet',
                      'SantQuirze','PratDelLlobregat']
df_column_name=['Parking Vilanova Renfe','Parking Sant Sadurní Renfe','Parking Sant Boi de Llobregat',
              'Parking Quatre Camins','Cerdanyola Universitat Renfe','Parking Granollers Renfe',
                'Parking Martorell FGC','Parking Mollet Renfe','Parking Sant Quirze FGC',
               'Parking Prat del Ll.']
current_parking_ix=1
# parkings which fill up: 3 QuatreCamins, 7 Mollet, 1 SantSadurni (sometimes),
# problems on Weekend with 2 SantBoi, 4 Cerdanyola, 
# bad data: 6 Martorell, 8 SantQuirze DO NOT USE
#good 0 Vilanova, 1 SantSadurni, 3 QuatreCamins, 5 Granollers, 7 Mollet, 9 PratDelLlobregat 
current_parking = available_parkings[current_parking_ix]
current_column_name=df_column_name[current_parking_ix]


# loadind data:
    # Getting back the objects:
with open('data/'+current_parking+'_normalized.pkl','rb') as f:  # Python 3: open(..., 'rb')
    df_normalized, weekday_offset, friday_offset,  weekend_offset, max_value= pickle.load(f)
    f.close()
axis_ylim = max_value+20

df_training, df_testing = split_data(df_normalized, 3)



df_mean_areas = df_training.groupby(['Profile_3'], as_index=False).mean() 
df_mean_areas[['Profile_3', 'Area']]

friday_area = df_mean_areas.iloc[0]['Area']
weekday_area = df_mean_areas.iloc[1]['Area']
weekend_area = df_mean_areas.iloc[2]['Area']

friday_max = df_mean_areas.iloc[0]['MaxV']
weekday_max = df_mean_areas.iloc[1]['MaxV']
weekend_max = df_mean_areas.iloc[2]['MaxV']

print('Weekday area: ' , weekday_area)
print('Friday area: ' , friday_area)
print('Weekend area: ' , weekend_area)

print('Weekday maximum: ' , weekday_max)
print('Friday maximum: ' , friday_max)
print('Weekend maximum: ' , weekend_max)



# ## MATHEMATICAL MODEL - CDF Subtraction

# In[132]:


# ********************************************** WEEKDAY *************************************************************
#from scipy.special import tna, factorial
from scipy.stats import truncnorm 
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
time = np.linspace(0,23.5,48)
time_tn=time/24

def generate_cdf(data):
    cumulative=[0]
    for ii in range(1,len(data)):
        cumulative.append(sum(data[:ii]))
    return cumulative/sum(data)

 
def tn(x, loc, scale):
    a = -loc/scale
    b = (1-loc)/scale
    return truncnorm.pdf(x, a, b, loc, scale)


def tn_cdf(x, loc, scale):
    a = -loc/scale
    b = (1-loc)/scale
    return truncnorm.cdf(x, a, b, loc, scale)


training_weekdays_norm  = get_days_of_protos_normalized("Weekday", df_training)
training_weekdays_isfull  = get_parkingfull_of_protos("Weekday", df_training)
training_fridays_norm  = get_days_of_protos_normalized("Friday", df_training)
training_fridays_isfull  = get_parkingfull_of_protos("Friday", df_training)
training_weekends_norm  = get_days_of_protos_normalized("Weekend", df_training)
training_weekends_norm = training_weekends_norm[:-1]
# t = []
# for i in range(0,len(training_weekends_norm)):
#     if training_weekends_norm[i].mean() != 0:
#         t.append(training_weekends_norm[i])
        
# training_weekends_norm = t
wd_length = len(training_weekdays_norm)
f_length = len(training_fridays_norm)
we_length = len(training_weekends_norm)


def model_weekdays_tn(params): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]

     # arrivals
    #a_ar = -loc_ar/scale_ar
    #b_ar = (1-loc_ar)/scale_ar

    # departures
    #a_de = -loc_de/scale_de
    #b_de = (1-loc_de)/scale_de
    
    # make tn for arrivals
    # arrival_pdf = tn(time_tn, loc_ar, scale_ar)
    # make tn for departures
    # departure_pdf = tn(time_tn, loc_de, scale_de)
    # compute CDF for arrivals
    # arrival_cdf = generate_cdf(arrival_pdf)
    
    # compute CDF for departures
    # departure_cdf = generate_cdf(departure_pdf)
    #cdf_ar = truncnorm.cdf(time_tn, a_ar, b_ar, loc=loc_ar, scale=scale_ar)
    #cdf_de = truncnorm.cdf(time_tn, a_de, b_de, loc=loc_de, scale=scale_de)
    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)

    #res = np.array(arrival_cdf) - np.array(departure_cdf)
    #res_n = res/sum(res)
    
    #res = np.array(cdf_ar) - np.array(cdf_de)
    res = cdf_ar - cdf_de
    res_n = res/sum(res)
  
    #res_n[res_n>thresh] = thresh
    #plt.plot(res_n)
    #plt.show()
    error = 0  
    for ii in range(0,wd_length):
        day = training_weekdays_norm[ii]
        error += mean_squared_error(res_n, day)

    #plot_model_tn_pres(loc_ar, scale_ar, loc_de, scale_de) 
    #print("mua = " + str(loc_ar) + "\tstda  = " + str(scale_ar))
    #print("mus = " + str(loc_de) + "\tstds = " + str(scale_de))
    #print("Err = " + str(error))
    return error

def model_weekdays_tn_th_max(params): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
    thresh=params[4]

    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)

    cdf_ar[cdf_ar>thresh] = thresh
    cdf_ar = cdf_ar/thresh

    res = cdf_ar - cdf_de
    res_n = res #/sum(res)
  
    error = 0  
    for ii in range(0,wd_length):
        day = training_weekdays_norm[ii]
        error += np.sum(np.power(res_n - day, 2))
        #error += mean_squared_error(res_n, day)
    return error


def model_weekdays_tn_th_ind_max(params): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
 

    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)

    #cdf_ar_th=cdf_ar
    #cdf_ar_th[cdf_ar>thresh] = thresh
    #cdf_ar_th = cdf_ar_th/thresh

    res = cdf_ar - cdf_de
    res_n = res#/sum(res)
    
    #res_th = cdf_ar_th - cdf_de
    #res_th_n = res_th/sum(res_th)
  
    error = 0  

    for ii in range(0,wd_length):
        day = training_weekdays_norm[ii]
        dayisFull=training_weekdays_isfull[ii]
        
        if dayisFull:
            thresh=params[4+ii]
            cdf_ar_th=cdf_ar
            cdf_ar_th[cdf_ar>thresh] = thresh
            cdf_ar_th = cdf_ar_th/thresh
    
            res_th = cdf_ar_th - cdf_de
            res_th_n = res_th#/sum(res_th)
            
            error += np.sum(np.power(res_th_n - day, 2))
            # error += mean_squared_error(res_th_n, day)
        else:
            error += np.sum(np.power(res_n - day,2))
            #error += mean_squared_error(res_n, day)           
    return error

# params order = a1, b1, a2, b2, rescale
# params order: loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1, rescale
#parameters = np.array([ 2 , 20, 5, 80, 0.02])
#parameters_tn = np.array([.3 ,.05,.8,.1])
#parameters_tn = np.array([.2 ,.05,.7,.1])
parameters_tn_th = np.array([.2 ,.05,.7,.1,1])
#optimal_params_weekday = minimize(model_weekdays, parameters, method='Nelder-Mead', tol=0.01)
#optimal_params_weekdaytn = minimize(model_weekdays_tn, parameters_tn, method='Nelder-Mead', tol=0.01)
#optimal_params_weekdaytn = minimize(model_weekdays_tn, parameters_tn, method='Nelder-Mead',
#                                    tol=1e-6, options={'disp': T3rue})
optimal_params_weekdaytn_glo = minimize(model_weekdays_tn_th_max, parameters_tn_th, method='Nelder-Mead',
                                    tol=1e-6, options={'disp': True})
var_glo = optimal_params_weekdaytn_glo.fun/np.size(training_weekdays_norm)

parameters_tn_th_ind = np.array([.2 ,.05,.7,.1] + [.8]*wd_length)

optimal_params_weekdaytn = minimize(model_weekdays_tn_th_ind_max, parameters_tn_th_ind, method='Nelder-Mead',
                                    tol=1e-6, options={'disp': True, 'maxfev': 100000})
var = optimal_params_weekdaytn.fun/np.size(training_weekdays_norm)



# In[133]:


optimal_params_weekdaytn_glo


# In[134]:


optimal_params_weekdaytn


# In[136]:


th_vec=optimal_params_weekdaytn.x[4:]
th_vec[training_weekdays_isfull]


# In[84]:


#plt.hist(th_vec[training_weekdays_isfull],8)


# In[137]:
def plot_model_tn(loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1):
    # arrivals
    a_ar = -loc_ar/scale_ar
    b_ar = (1-loc_ar)/scale_ar

    # departures
    a_de = -loc_de/scale_de
    b_de = (1-loc_de)/scale_de


    pdf_ar = truncnorm.pdf(time_tn, a_ar, b_ar, loc=loc_ar, scale=scale_ar)
    pdf_de = truncnorm.pdf(time_tn, a_de, b_de, loc=loc_de, scale=scale_de)
    cdf_ar = truncnorm.cdf(time_tn, a_ar, b_ar, loc=loc_ar, scale=scale_ar)
    cdf_de = truncnorm.cdf(time_tn, a_de, b_de, loc=loc_de, scale=scale_de)

    fig, ax = plt.subplots(2)
    ax[0].plot(time, pdf_ar , '-b')
    ax[0].plot(time, pdf_de, '-r')
    ax[0].set_title('pdfs')

    ax[1].plot(time, cdf_ar , '--b')
    ax[1].plot(time, cdf_de, '--r')
    ax[1].plot(time, cdf_ar-cdf_de, 'r')
    ax[1].set_title('cdfs')


# In[138]:


plot_model_tn(optimal_params_weekdaytn.x[0],optimal_params_weekdaytn.x[1],optimal_params_weekdaytn.x[2],optimal_params_weekdaytn.x[3])


# In[139]:


def plot_model_tn_th(loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1,thresh=.8):
    # arrivals
    a_ar = -loc_ar/scale_ar
    b_ar = (1-loc_ar)/scale_ar

    # departures
    a_de = -loc_de/scale_de
    b_de = (1-loc_de)/scale_de


    pdf_ar = truncnorm.pdf(time_tn, a_ar, b_ar, loc=loc_ar, scale=scale_ar)
    pdf_de = truncnorm.pdf(time_tn, a_de, b_de, loc=loc_de, scale=scale_de)
    cdf_ar = truncnorm.cdf(time_tn, a_ar, b_ar, loc=loc_ar, scale=scale_ar)
    
    ix_parking_full= np.argmax(cdf_ar>thresh)
    pdf_ar[cdf_ar>thresh] =0
    cdf_ar[cdf_ar>thresh] = thresh
    cdf_ar = cdf_ar/thresh
    
    cdf_de = truncnorm.cdf(time_tn, a_de, b_de, loc=loc_de, scale=scale_de)

    fig, ax = plt.subplots(2)
    ax[0].plot(time, pdf_ar , '-b')
    ax[0].plot(time, pdf_de, '-r')
    ax[0].plot(0.5*ix_parking_full*np.array([1, 1]),[0, max(pdf_ar)],'--')
    ax[0].set_title('pdfs')

    ax[1].plot(time, cdf_ar , '--b')
    ax[1].plot(time, cdf_de, '--r')
    ax[1].plot(time, cdf_ar-cdf_de, 'r')
    ax[1].plot(0.5*ix_parking_full*np.array([1, 1]),[0,1],'--')
    ax[1].set_title('cdfs')


# In[140]:


plot_model_tn_th(optimal_params_weekdaytn.x[0],optimal_params_weekdaytn.x[1],optimal_params_weekdaytn.x[2],
              optimal_params_weekdaytn.x[3],optimal_params_weekdaytn.x[4])


# In[141]:


weekday_math_params = optimal_params_weekdaytn.x
weekday_math_params


# In[221]:


time2 = np.linspace(0,23.5,48)
tn1_wd=tn(time_tn, optimal_params_weekdaytn.x[0], optimal_params_weekdaytn.x[1])
tn2_wd=tn(time_tn, optimal_params_weekdaytn.x[2], optimal_params_weekdaytn.x[3])
#tn1_wd = gam(time2, optimal_params_weekday.x[0], optimal_params_weekday.x[1])
#tn2_wd = gam(time2, optimal_params_weekday.x[2], optimal_params_weekday.x[3])

#cdf1_wd_ap = generate_cdf(tn1_wd)
#cdf2_wd_ap = generate_cdf(tn2_wd)
cdf1_wd = tn_cdf(time_tn, optimal_params_weekdaytn.x[0], optimal_params_weekdaytn.x[1])
prototype_math_arr_weekday=cdf1_wd
cdf2_wd = tn_cdf(time_tn, optimal_params_weekdaytn.x[2], optimal_params_weekdaytn.x[3])
prototype_math_dep_weekday=cdf2_wd
#cdf1_wd = generate_cdf(tn1_wd)
#cdf2_wd = generate_cdf(tn2_wd)

resta_wd = np.array(cdf1_wd) - np.array(cdf2_wd)
prototype_math_weekday = resta_wd#/sum(resta_wd)

#resta_wd_ap= np.array(cdf1_wd_ap) - np.array(cdf2_wd_ap)
#prototype_math_weekday_ap = resta_wd_ap/sum(resta_wd_ap)

fig = plt.figure(figsize=(18,10))
fig.suptitle("PDF and CDF for arrival and deartures - WEEKDAYS ("+current_parking+")", fontsize=20)
plt.plot(time2, tn1_wd/sum(tn1_wd), label="Probability that a slot gets occupied")
plt.plot(time2, tn2_wd/sum(tn2_wd),  label="Probability that a slot gets free")
plt.plot(time2, cdf1_wd, label="Cummulative probability arrival")
plt.plot(time2, cdf2_wd, label="Cummulative probability departure")
#plt.plot(time2, cdf1_wd_ap, label="Cummulative probability arrival approx")
#plt.plot(time2, cdf2_wd_ap, label="Cummulative probability departure approx")
plt.grid(linestyle='dotted')
plt.xlabel("Time [h]", fontsize=18)
plt.ylabel("PDF & CDF", fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16, loc="upper left");


# In[223]:


time2 = np.linspace(0,23.5,48)
tn1_wd=tn(time_tn, optimal_params_weekdaytn.x[0], optimal_params_weekdaytn.x[1])
tn2_wd=tn(time_tn, optimal_params_weekdaytn.x[2], optimal_params_weekdaytn.x[3])

cdf1_wd = tn_cdf(time_tn, optimal_params_weekdaytn.x[0], optimal_params_weekdaytn.x[1])

ix_parking_full= np.argmax(cdf1_wd>optimal_params_weekdaytn.x[4])
tn1_wd[cdf1_wd>optimal_params_weekdaytn.x[4]] =0
cdf1_wd[cdf1_wd>optimal_params_weekdaytn.x[4]] = optimal_params_weekdaytn.x[4]
cdf1_wd = cdf1_wd/optimal_params_weekdaytn.x[4]

cdf2_wd = tn_cdf(time_tn, optimal_params_weekdaytn.x[2], optimal_params_weekdaytn.x[3])
#cdf1_wd = generate_cdf(tn1_wd)
#cdf2_wd = generate_cdf(tn2_wd)

resta_wd = np.array(cdf1_wd) - np.array(cdf2_wd)
prototype_math_weekday = resta_wd#/sum(resta_wd)

#resta_wd_ap= np.array(cdf1_wd_ap) - np.array(cdf2_wd_ap)
#prototype_math_weekday_ap = resta_wd_ap/sum(resta_wd_ap)

fig = plt.figure(figsize=(18,10))
fig.suptitle("PDF and CDF for arrival and deartures - WEEKDAYS ("+current_parking+")", fontsize=20)
plt.plot(time2, tn1_wd/sum(tn1_wd), label="Probability that a slot gets occupied")
plt.plot(time2, tn2_wd/sum(tn2_wd),  label="Probability that a slot gets free")
plt.plot(time2, cdf1_wd, label="Cummulative probability arrival")
plt.plot(time2, cdf2_wd, label="Cummulative probability departure")
plt.plot(0.5*ix_parking_full*np.array([1, 1]),[0,1],'--',label="Parking full")
#plt.plot(time2, cdf1_wd_ap, label="Cummulative probability arrival approx")
#plt.plot(time2, cdf2_wd_ap, label="Cummulative probability departure approx")
plt.grid(linestyle='dotted')
plt.xlabel("Time [h]", fontsize=18)
plt.ylabel("PDF & CDF", fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16, loc="upper left");


# In[225]:


fig = plt.figure(figsize=(18,10))
fig.suptitle("Normalized mathematical prototope from CDF subtraction - WEEKDAYS ("+current_parking+")", fontsize=20)
plt.grid(linestyle='dotted')
plt.xlabel("Time [h]", fontsize=18)
plt.ylabel("PDF & CDF", fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.plot(time2, prototype_math_arr_weekday/optimal_params_weekdaytn.x[4], linewidth=3, color='blue', label="excess")
plt.plot(time2, prototype_math_weekday, linewidth=3, color='red', label="(CDF1 - CDF2)")
#plt.plot(time2, prototype_math_weekday_ap, linewidth=3, color='red', label="approx")
for i in range(0,len(training_weekdays_norm)):
    plt.plot(time, training_weekdays_norm[i], linewidth=0.45)

plt.legend(fontsize=16, loc="upper left");


# In[146]:


tn1_wd=tn(time_tn, optimal_params_weekdaytn.x[0], optimal_params_weekdaytn.x[1])
tn2_wd=tn(time_tn, optimal_params_weekdaytn.x[2], optimal_params_weekdaytn.x[3])

cdf2_wd = tn_cdf(time_tn, optimal_params_weekdaytn.x[2], optimal_params_weekdaytn.x[3])

for ii in range(0,len(training_weekdays_norm)):
    idx_th=ii+4

    cdf1_wd = tn_cdf(time_tn, optimal_params_weekdaytn.x[0], optimal_params_weekdaytn.x[1])


    dayisFull=training_weekdays_isfull[ii]
    if dayisFull:
        print(optimal_params_weekdaytn.x[idx_th])
    
        ix_parking_full= np.argmax(cdf1_wd>optimal_params_weekdaytn.x[idx_th])
        time_parking_full= 0.5*ix_parking_full
        str_parking_full= f'{int(time_parking_full):02.0f}:{int((time_parking_full-int(time_parking_full))*60):02.0f}h'

        print('Parking full        = '+str_parking_full)
        tn1_wd[cdf1_wd>optimal_params_weekdaytn.x[idx_th]] =0
        cdf1_wd[cdf1_wd>optimal_params_weekdaytn.x[idx_th]] = optimal_params_weekdaytn.x[idx_th]
        cdf1_wd = cdf1_wd/optimal_params_weekdaytn.x[idx_th]
    else:
        ix_parking_full=0
    resta_wd = np.array(cdf1_wd) - np.array(cdf2_wd)
    prototype_math_weekday = resta_wd#/sum(resta_wd)
    fig = plt.figure(figsize=(18,10))
    fig.suptitle("Normalized mathematical prototope from CDF subtraction - weekdayS ("+current_parking+")", fontsize=20)
    if dayisFull:
        plt.plot(0.5*ix_parking_full*np.array([1, 1]),[0,1],'--',label="Parking full "+str_parking_full)
    plt.grid(linestyle='dotted')
    plt.xlabel("Time [h]", fontsize=18)
    plt.ylabel("PDF & CDF", fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.plot(time, prototype_math_weekday, linewidth=3, color='red', label="(CDF1 - CDF2)*Rescale")
    #for i in range(0,len(training_weekdays_norm)):
    plt.plot(time, training_weekdays_norm[ii], linewidth=0.45)
    plt.legend(fontsize=16);


# ### FRIDAYS

# In[147]:


def model_fridays_tn(params): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
    error = 0
    # make tn for arrivals
    arrival_cdf = tn_cdf(time_tn, loc_ar, scale_ar)
    # make tn for departures
    departure_cdf = tn_cdf(time_tn, loc_de, scale_de)
    # compute CDF for arrivals
    #arrival_cdf = generate_cdf(arrival_pdf)
    # compute CDF for departures
    #departure_cdf = generate_cdf(departure_pdf)
    #res = np.array(arrival_cdf) - np.array(departure_cdf)
    res =arrival_cdf - departure_cdf
    res_n = res/sum(res)
    
    for ii in range(0,f_length):
        day = training_fridays_norm[ii]
        error += mean_squared_error(res_n, day)
    return error

def model_fridays_tn_th_max(params): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
    thresh=params[4]

    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)

    cdf_ar[cdf_ar>thresh] = thresh
    cdf_ar = cdf_ar/thresh

    res = cdf_ar - cdf_de
    res_n = res#/sum(res)
  
    error = 0  
    for ii in range(0,f_length):
        day = training_fridays_norm[ii]
        error += mean_squared_error(res_n, day)
    return error

def model_fridays_tn_th_opt_max(params): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
    thresh=params[4]

    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)

    cdf_ar_th=cdf_ar
    cdf_ar_th[cdf_ar>thresh] = thresh
    cdf_ar_th = cdf_ar_th/thresh

    res = cdf_ar - cdf_de
    res_n = res/sum(res)
    
    res_th = cdf_ar_th - cdf_de
    res_th_n = res_th#/sum(res_th)
  
    error = 0  
    for ii in range(0,f_length):
        day = training_fridays_norm[ii]
        dayisFull=training_fridays_isfull[ii]
        #print(dayisFull)
        if dayisFull:
            error += mean_squared_error(res_th_n, day)
        else:
            error += mean_squared_error(res_n, day)           
    return error

def model_fridays_tn_th_ind_max(params): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
 

    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)

    #cdf_ar_th=cdf_ar
    #cdf_ar_th[cdf_ar>thresh] = thresh
    #cdf_ar_th = cdf_ar_th/thresh

    res = cdf_ar - cdf_de
    res_n = res#/sum(res)
    
    #res_th = cdf_ar_th - cdf_de
    #res_th_n = res_th/sum(res_th)
  
    error = 0  

    for ii in range(0,f_length):
        day = training_fridays_norm[ii]
        dayisFull=training_fridays_isfull[ii]
        
        if dayisFull:
            thresh=params[4+ii]
            cdf_ar_th=cdf_ar
            cdf_ar_th[cdf_ar>thresh] = thresh
            cdf_ar_th = cdf_ar_th/thresh
    
            res_th = cdf_ar_th - cdf_de
            res_th_n = res_th#/sum(res_th)
            
            error += mean_squared_error(res_th_n, day)
        else:
            error += mean_squared_error(res_n, day)           
    return error

# params order = a1, b1, a2, b2
#parameters = np.array([ 2 , 20, 5, 80, 0.02])
#optimal_params_friday = minimize(model_fridays, parameters, method='Nelder-Mead', tol=0.01)

# params order: loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1, rescale
#parameters_tn = np.array([.3 ,.05,.8,.1])
#optimal_params_fridaytn = minimize(model_fridays_tn, parameters_tn, method='Nelder-Mead', tol=0.01)

#parameters_tn = np.array([.2 ,.05,.7,.1])
#optimal_params_fridaytn = minimize(model_fridays_tn, parameters_tn, method='Nelder-Mead',
#                                    tol=1e-6, options={'disp': True})
parameters_tn_th = np.array([.2 ,.05,.7,.1,.8])
optimal_params_fridaytn_glo = minimize(model_fridays_tn_th_opt_max, parameters_tn_th, method='Nelder-Mead',
                                    tol=1e-6, options={'disp': True})
parameters_tn_th_ind = np.array([.2 ,.05,.7,.1] + [.8]*f_length)

optimal_params_fridaytn = minimize(model_fridays_tn_th_ind_max, parameters_tn_th_ind, method='Nelder-Mead',
                                    tol=1e-6, options={'disp': True, 'maxfev': 100000})


# In[148]:


optimal_params_fridaytn


# In[149]:


optimal_params_fridaytn_glo


# In[150]:


#optimal_params_friday.x
#friday_math_params = optimal_params_friday.x
friday_math_params = optimal_params_fridaytn.x


# In[151]:



#tn1_fri = gam(time, optimal_params_friday.x[0], optimal_params_friday.x[1])
#tn2_fri = gam(time, optimal_params_friday.x[2], optimal_params_friday.x[3])
tn1_fri=tn(time_tn, optimal_params_fridaytn.x[0], optimal_params_fridaytn.x[1])
tn2_fri=tn(time_tn, optimal_params_fridaytn.x[2], optimal_params_fridaytn.x[3])

#cdf1_fri = generate_cdf(tn1_fri)
#cdf2_fri = generate_cdf(tn2_fri)
#cdf1_fri = generate_cdf(tn1_fri)
#cdf2_fri = generate_cdf(tn2_fri)
cdf1_fri=tn_cdf(time_tn, optimal_params_fridaytn.x[0], optimal_params_fridaytn.x[1])
cdf2_fri=tn_cdf(time_tn, optimal_params_fridaytn.x[2], optimal_params_fridaytn.x[3])

resta = np.array(cdf1_fri) - np.array(cdf2_fri)
prototype_math_friday = resta/sum(resta)
fig = plt.figure(figsize=(18,10))
fig.suptitle("PDF and CDF for occupying and freeing a slot - FRIDAYS ("+current_parking+")", fontsize=20)
plt.plot(time, cdf1_fri, label="CDF Slot occupied")
plt.plot(time, cdf2_fri, label="CDF Slot free")
plt.plot(time, tn1_fri/sum(tn1_fri), label="Probability that a slot gets occupied")
plt.plot(time, tn2_fri/sum(tn2_fri),  label="Probability that a slot gets free")
plt.grid(linestyle='dotted')
plt.xlabel("Time [h]", fontsize=18)
plt.ylabel("PDF & CDF", fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16, loc="upper left")

plt.legend(fontsize=16)


# In[241]:


idx_th=4
time2 = np.linspace(0,23.5,48)
tn1_fri=tn(time_tn, optimal_params_fridaytn.x[0], optimal_params_fridaytn.x[1])
tn2_fri=tn(time_tn, optimal_params_fridaytn.x[2], optimal_params_fridaytn.x[3])

cdf1_fri = tn_cdf(time_tn, optimal_params_fridaytn.x[0], optimal_params_fridaytn.x[1])
prototype_math_arr_friday=cdf1_fri.copy()


ix_parking_full= np.argmax(cdf1_fri>optimal_params_fridaytn.x[idx_th])
tn1_fri[cdf1_fri>optimal_params_fridaytn.x[idx_th]] =0
cdf1_fri[cdf1_fri>optimal_params_fridaytn.x[idx_th]] = optimal_params_fridaytn.x[idx_th]

cdf1_fri = cdf1_fri/optimal_params_fridaytn.x[idx_th]


cdf2_fri = tn_cdf(time_tn, optimal_params_fridaytn.x[2], optimal_params_fridaytn.x[3])
prototype_math_dep_friday=cdf2_fri

#cdf1_wd = generate_cdf(tn1_wd)
#cdf2_wd = generate_cdf(tn2_wd)

resta_fri = np.array(cdf1_fri) - np.array(cdf2_fri)
prototype_math_friday = resta_fri#/sum(resta_fri)

fig = plt.figure(figsize=(18,10))
fig.suptitle("PDF and CDF for occupying and freeing a slot - FRIDAYS ("+current_parking+")", fontsize=20)
plt.plot(time2, cdf1_fri, label="CDF Slot occupied")
plt.plot(time2, cdf2_fri, label="CDF Slot free")
plt.plot(time2, tn1_fri/sum(tn1_fri), label="Probability that a slot gets occupied")
plt.plot(time2, tn2_fri/sum(tn2_fri),  label="Probability that a slot gets free")
plt.plot(0.5*ix_parking_full*np.array([1, 1]),[0,1],'--',label="Parking full")
plt.grid(linestyle='dotted')
plt.xlabel("Time [h]", fontsize=18)
plt.ylabel("PDF & CDF", fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16, loc="upper left");


# In[242]:


fig = plt.figure(figsize=(18,10))
fig.suptitle("Normalized mathematical prototope from CDF subtraction - FRIDAYS ("+current_parking+")", fontsize=20)
plt.grid(linestyle='dotted')
plt.xlabel("Time [h]", fontsize=18)
plt.ylabel("PDF & CDF", fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.plot(time, prototype_math_arr_friday/optimal_params_fridaytn.x[idx_th], linewidth=3, color='blue', label="excess")
plt.plot(time, prototype_math_friday, linewidth=3, color='red', label="(CDF1 - CDF2)*Rescale")
for i in range(0,len(training_fridays_norm)):
    plt.plot(time, training_fridays_norm[i], linewidth=0.45)
plt.legend(fontsize=16);


# In[155]:


tn1_fri=tn(time_tn, optimal_params_fridaytn.x[0], optimal_params_fridaytn.x[1])
tn2_fri=tn(time_tn, optimal_params_fridaytn.x[2], optimal_params_fridaytn.x[3])

cdf2_fri = tn_cdf(time_tn, optimal_params_fridaytn.x[2], optimal_params_fridaytn.x[3])

for ii in range(0,len(training_fridays_norm)):
    idx_th=ii+4

    cdf1_fri = tn_cdf(time_tn, optimal_params_fridaytn.x[0], optimal_params_fridaytn.x[1])


    dayisFull=training_fridays_isfull[ii]
    if dayisFull:
        print(optimal_params_fridaytn.x[idx_th])
    
        ix_parking_full= np.argmax(cdf1_fri>optimal_params_fridaytn.x[idx_th])
        time_parking_full= 0.5*ix_parking_full
        str_parking_full= f'{int(time_parking_full):02.0f}:{int((time_parking_full-int(time_parking_full))*60):02.0f}h'

        print('Parking full        = '+str_parking_full)
        tn1_fri[cdf1_fri>optimal_params_fridaytn.x[idx_th]] =0
        cdf1_fri[cdf1_fri>optimal_params_fridaytn.x[idx_th]] = optimal_params_fridaytn.x[idx_th]
        cdf1_fri = cdf1_fri/optimal_params_fridaytn.x[idx_th]
    else:
        ix_parking_full=0
    resta_fri = np.array(cdf1_fri) - np.array(cdf2_fri)
    prototype_math_friday = resta_fri#/sum(resta_fri)
    fig = plt.figure(figsize=(18,10))
    fig.suptitle("Normalized mathematical prototope from CDF subtraction - FRIDAYS ("+current_parking+")", fontsize=20)
    if dayisFull:
        plt.plot(0.5*ix_parking_full*np.array([1, 1]),[0,.06],'--',label="Parking full "+str_parking_full)
    plt.grid(linestyle='dotted')
    plt.xlabel("Time [h]", fontsize=18)
    plt.ylabel("PDF & CDF", fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.plot(time, prototype_math_friday, linewidth=3, color='red', label="(CDF1 - CDF2)*Rescale")
    #for i in range(0,len(training_fridays_norm)):
    plt.plot(time, training_fridays_norm[ii], linewidth=0.45)
    plt.legend(fontsize=16);


# ### WEEKENDS

# In[197]:


training_weekends_norm  = get_days_of_protos_normalized("Weekend", df_training)
training_weekends_norm = training_weekends_norm[:-1]
t = []
for i in range(0,len(training_weekends_norm)):
    if training_weekends_norm[i].mean() != 0:
        t.append(training_weekends_norm[i])
        
training_weekends_norm = t
we_length = len(t)

def model_weekends_tn(params): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
    error = 0
    # make tn for arribals
    #arrival_pdf = tn(time_tn, loc_ar, scale_ar)
    # make tn for departures
    #departure_pdf = tn(time_tn, loc_de, scale_de)
    # compute CDF for arribals
    #arrival_cdf = generate_cdf(arrival_pdf)
    # compute CDF for departures
    #departure_cdf = generate_cdf(departure_pdf)
    
    arrival_cdf = tn_cdf(time_tn, loc_ar, scale_ar)
    departure_cdf = tn_cdf(time_tn, loc_de, scale_de)
    departure_cdf=departure_cdf/max(departure_cdf)
    res =arrival_cdf - departure_cdf
    
    #res = np.array(arrival_cdf) - np.array(departure_cdf)
    res_n = res#/sum(res)
    
    #print(loc_de)
    #print(scale_de)
    #print(departure_cdf)
    
    for ii in range(0,we_length):
        day = training_weekends_norm[ii]
        error += mean_squared_error(res_n, day)
        
        
        
    #plot_model_tn_pres(loc_ar, scale_ar, loc_de, scale_de) 
    #print("mua = " + str(loc_ar) + "\tstda  = " + str(scale_ar))
    #print("mus = " + str(loc_de) + "\tstds = " + str(scale_de))
    #print("Err = " + str(error))     

    return error


# params order = a1, b1, a2, b2
#parameters = np.array([ 2 , 20, 5, 80, 2])
#optimal_params_weekend = minimize(model_weekends, parameters, method='Nelder-Mead', tol=0.01)

# params order: loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1, rescale
parameters_tn = np.array([.3 ,.1,.8,0.5])
#optimal_params_weekendtn = minimize(model_weekends_tn, parameters_tn, method='Nelder-Mead', tol=0.01)
#optimal_params_weekendtn = minimize(model_weekends_tn, parameters_tn, method='Nelder-Mead',
#                                    tol=1e-6, options={'disp': True})
if ((current_parking == "SantBoi") or (current_parking == "Cerdanyola")): 
    optimal_params_weekendtn = minimize(model_weekends_tn, parameters_tn, method='SLSQP',
                                        bounds=((0, None), (0, None),(0, None),(0, None)),
                                        tol=1e-6, options={'disp': True})
else:
    optimal_params_weekendtn = minimize(model_weekends_tn, parameters_tn, method='Nelder-Mead',
                                        tol=1e-6, options={'disp': True, 'maxfev': 100000})  


# In[198]:


#weekend_math_params = optimal_params_weekend.x
#optimal_params_weekend.x
weekend_math_params = optimal_params_weekendtn.x
optimal_params_weekendtn.x


# In[201]:


#tn1_we = gam(time, optimal_params_weekend.x[0], optimal_params_weekend.x[1])
#tn2_we = gam(time, optimal_params_weekend.x[2], optimal_params_weekend.x[3])
tn1_we=tn(time_tn, optimal_params_weekendtn.x[0], optimal_params_weekendtn.x[1])
tn2_we=tn(time_tn, optimal_params_weekendtn.x[2], optimal_params_weekendtn.x[3])



#cdf1_we = generate_cdf(tn1_we)
#cdf2_we = generate_cdf(tn2_we)
cdf1_we = tn_cdf(time_tn, optimal_params_weekendtn.x[0], optimal_params_weekendtn.x[1])
cdf2_we = tn_cdf(time_tn, optimal_params_weekendtn.x[2], optimal_params_weekendtn.x[3])
cdf2_we=cdf2_we/max(cdf2_we)
print(max(cdf1_we))

print(max(cdf2_we))

resta_we = np.array(cdf1_we) - np.array(cdf2_we)
prototype_math_weekend = resta_we#/sum(resta_we)
fig = plt.figure(figsize=(18,10))
fig.suptitle("PDF and CDF for occupying and freeing a slot - WEEKENDS ("+current_parking+")", fontsize=20)
plt.plot(time, cdf1_we, label="CDF Slot occupied")
plt.plot(time, cdf2_we, label="CDF Slot free")
plt.plot(time, tn1_we/sum(tn1_we), label="Probability that a slot is occupied")
plt.plot(time, tn2_we/sum(tn2_we),  label="Probability  a slot gets free")
plt.grid(linestyle='dotted')
plt.xlabel("Time [h]", fontsize=18)
plt.ylabel("PDF & CDF", fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16, loc="upper left")

plt.legend(fontsize=16);


# In[202]:


fig = plt.figure(figsize=(18,10))
fig.suptitle("Normalized mathematical prototope from CDF subtraction - WEEKENDS ("+current_parking+")", fontsize=20)
plt.grid(linestyle='dotted')
plt.xlabel("Time [h]", fontsize=18)
plt.ylabel("PDF & CDF", fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.plot(time, prototype_math_weekend, linewidth=3, color='red', label="(CDF1 - CDF2)*Rescale")
for i in range(0,len(training_weekends_norm)):
    plt.plot(time, training_weekends_norm[i], linewidth=0.45)
plt.legend(fontsize=16)


# In[244]:


tn_weekday_n = prototype_math_weekday
tn_friday_n = prototype_math_friday
tn1_weekend_n = prototype_math_weekend

tn_arr_weekday_n = prototype_math_arr_weekday
tn_dep_weekday_n = prototype_math_dep_weekday
tn_arr_friday_n = prototype_math_arr_friday
tn_dep_friday_n = prototype_math_dep_friday


# In[204]:


ymax=1.05
fig, ax = plt.subplots(1,3)
fig.set_figwidth(20)
fig.set_figheight(4)
fig.suptitle('Decomoposition for each prototype [TN Probability Normalized] - ' + current_parking, fontsize=20)

# for ii in range(len(training_weekdays_norm)):
#     ax[0].plot(time,training_weekdays_norm[ii], linewidth='0.5')
ax[0].plot(time2, tn_weekday_n, linewidth=2, color= 'green', label='Mathematical Weekday fit')
ax[0].set_ylim([0,ymax])
ax[0].grid(linestyle='dotted')
ax[0].legend(fontsize=8)
ax[0].set_xlabel('Time (hour)')
ax[0].set_ylabel('Probability')

# for ii in range(len(training_fridays_norm)):
#     ax[1].plot(time,training_fridays_norm[ii], linewidth='0.5')
ax[1].plot(time2, tn_friday_n, linewidth=2, color= 'green', label='Mathematical Friday fit')
ax[1].set_ylim([0,ymax])
ax[1].grid(linestyle='dotted')
ax[1].legend(fontsize=9)
ax[1].set_xlabel('Time (hour)')
ax[1].set_ylabel('Probability')

# for ii in range(len(training_weekends_norm)):
#      ax[2].plot(time,training_weekends_norm[ii], linewidth='0.5')
ax[2].plot(time2, tn1_weekend_n, linewidth=2, color='green', label='Mathematical Weekend wave')
ax[2].set_ylim([0,ymax])
ax[2].grid(linestyle='dotted')
ax[2].legend()
ax[2].set_xlabel('Time (hour)')
ax[2].set_ylabel('Probability');


# In[205]:


fig, ax = plt.subplots(1,3)
fig.set_figwidth(20)
fig.set_figheight(4)
fig.suptitle('Decomoposition for each prototype [TN Probability Normalized] - ' + current_parking, fontsize=20)

# for ii in range(len(training_weekdays_norm)):
#     ax[0].plot(time,training_weekdays_norm[ii], linewidth='0.5')
ax[0].plot(time, tn_weekday_n, linewidth=2, color= 'green', label='Mathematical Weekday fit')
ax[0].set_ylim([0,ymax])
ax[0].grid(linestyle='dotted')
ax[0].legend(fontsize=8)
ax[0].set_xlabel('Time (hour)')
ax[0].set_ylabel('Probability')

# for ii in range(len(training_fridays_norm)):
#     ax[1].plot(time,training_fridays_norm[ii], linewidth='0.5')
ax[1].plot(time, tn_friday_n, linewidth=2, color= 'green', label='Mathematical Friday fit')
ax[1].set_ylim([0,ymax])
ax[1].grid(linestyle='dotted')
ax[1].legend(fontsize=9)
ax[1].set_xlabel('Time (hour)')
ax[1].set_ylabel('Probability')

# CHECK THIS
Area = integrate.simps(tn1_weekend_n) 
tn1_weekend_n = tn1_weekend_n/Area

# for ii in range(len(training_weekends_norm)):
#      ax[2].plot(time,training_weekends_norm[ii], linewidth='0.5')
ax[2].plot(time, tn1_weekend_n, linewidth=2, color='green', label='Mathematical Weekend wave')
ax[2].set_ylim([0,ymax])
ax[2].grid(linestyle='dotted')
ax[2].legend()
ax[2].set_xlabel('Time (hour)')
ax[2].set_ylabel('Probability');


# In[271]:


#**************************************WEEKDAY************************************
tn_weekday = tn_weekday_n*weekday_max + weekday_offset
tn_arr_weekday = tn_arr_weekday_n*weekday_max + weekday_offset
tn_dep_weekday = tn_dep_weekday_n*weekday_max #+ weekday_offset

#**************************************FRIDAY************************************
tn_friday = tn_friday_n*friday_max + friday_offset
tn_arr_friday = tn_arr_friday_n*friday_max + friday_offset
tn_dep_friday = tn_dep_friday_n*friday_max #+ friday_offset

#**************************************WEEKEND************************************
tn1_weekend = tn1_weekend_n*weekend_max + weekend_offset 


# In[267]:


fig, ax = plt.subplots(1,3)
fig.set_figwidth(20)
fig.set_figheight(4)
fig.suptitle('Decomoposition for each prototype [TN Probability Denormalized] - ' + current_parking , fontsize=20)

# for ii in range(len(training_weekdays)):
#     ax[0].plot(time,training_weekdays[ii], linewidth='0.5')
ax[0].plot(time, tn_weekday, linewidth=2, color= 'green', label='CDF subtraction Weekday fit')
ax[0].set_ylim([0,axis_ylim])
ax[0].grid(linestyle='dotted')
ax[0].legend(fontsize=8)
ax[0].set_xlabel('Time (hour)')
ax[0].set_ylabel('Probability')

# for ii in range(len(training_fridays)):
#     ax[1].plot(time,training_fridays[ii], linewidth='0.5')
ax[1].plot(time, tn_friday, linewidth=2, color= 'green', label='CDF subtraction Friday fit')
ax[1].set_ylim([0,axis_ylim])
ax[1].grid(linestyle='dotted')
ax[1].legend(fontsize=9)
ax[1].set_xlabel('Time (hour)')
ax[1].set_ylabel('Probability')

# for ii in range(len(training_weekends)):
#     ax[2].plot(time,training_weekends[ii], linewidth='0.5')
ax[2].plot(time, tn1_weekend, linewidth=2, color='green', label='TN Weekend wave')
axis_ylim_we=max(tn1_weekend)+20
ax[2].set_ylim([0,axis_ylim_we])
ax[2].grid(linestyle='dotted')
ax[2].legend()
ax[2].set_xlabel('Time (hour)')
ax[2].set_ylabel('Probability');
# ax[2].set_yticks([0,0.005,0.01])
# ax[2].set_yticks(["0","0.05","0.1"])


# ### Comparing normalized mathematical fitted prototype with testing data

# In[212]:




# ### Denormalization / Rescaling

# In[213]:


fig, ax = plt.subplots(3, 3)
fig.set_figwidth(20)
fig.set_figheight(14)
fig.suptitle('Comparison of each day real data against CDF subtraction (Denormalized) - ' + current_parking, fontsize=20)

tn_weekday = tn_weekday_n*weekday_max + weekday_offset
tn_friday = tn_friday_n*friday_max + friday_offset
tn1_weekend = tn1_weekend_n*weekend_max + weekend_offset 

    
# loadind testind data:
with open('data/'+current_parking+'_testing.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [testing_mondays, testing_mondays_dates, testing_tuesdays, testing_tuesdays_dates, 
    testing_wednesdays, testing_wednesdays_dates, testing_thursdays, testing_thursdays_dates, 
    testing_fridays, testing_fridays_dates, testing_saturdays, testing_saturdays_dates,
    testing_sundays, testing_sundays_dates]= pickle.load(f)
    f.close()  


subplot_training(fig, ax, 0, 0, tn_weekday, testing_mondays, 'Monday', 'Weekday',axis_ylim)
subplot_training(fig, ax, 0, 1, tn_weekday, testing_tuesdays, 'Tuesday', 'Weekday',axis_ylim)
subplot_training(fig, ax, 0, 2, tn_weekday, testing_wednesdays, 'Wednesday', 'Weekday',axis_ylim)
subplot_training(fig, ax, 1, 0, tn_weekday, testing_thursdays, 'Thursday', 'Weekday',axis_ylim)
subplot_training(fig, ax, 1, 1, tn_friday, testing_fridays, 'Friday', 'Friday',axis_ylim)
subplot_training(fig, ax, 1, 2, tn1_weekend, testing_saturdays, 'Saturday', 'Weekend',axis_ylim)
subplot_training(fig, ax, 2, 1, tn1_weekend, testing_sundays, 'Sunday', 'Weekend',axis_ylim)

ax[2,0].set_visible(False)
ax[2,2].set_visible(False)
ax[2,1].set_ylim(0,axis_ylim_we)
ax[1,2].set_ylim(0,axis_ylim_we)
fig.tight_layout(pad=5.0)


# In[214]:


fig = plt.figure(figsize=(17,11))
plt.title('Wednesday entire day prediction compared with testing data - '+current_parking, fontsize = 22)
plt.plot(time, tn_weekday, linestyle='dashed', linewidth=3, color='black', label='Weekday TN Prototype')
plt.plot(time, testing_wednesdays[0], linewidth=2, label="Testing Wednesday")
plt.plot(time, testing_wednesdays[1], linewidth=2, label="Testing Wednesday")
# plt.plot(time, testing_wednesdays[3], linewidth=2, label="Testing Wednesday")
plt.grid(linestyle='dotted')
plt.legend(fontsize=18)
plt.xlabel('Time [h], step 0.5', fontsize=20)
plt.ylabel('Occupancy', fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16);


# In[215]:


def compute_testing_prop_error(testing_days, proto_data):
    errors = np.zeros(48)
    n_test_days = len(testing_days)
    proto = np.array(proto_data)
    
    for i in range(0, n_test_days):
        day = np.array(testing_days[i])
        er = np.array((np.absolute(proto - day)/max_value)*100)
        errors += er
    return errors/n_test_days

error_monday_tn = compute_testing_prop_error(testing_mondays, tn_weekday)
error_tuesday_tn = compute_testing_prop_error(testing_tuesdays, tn_weekday)
error_wednesday_tn = compute_testing_prop_error(testing_wednesdays, tn_weekday)
error_thursday_tn = compute_testing_prop_error(testing_thursdays, tn_weekday)
error_friday_tn = compute_testing_prop_error(testing_fridays, tn_friday)
error_saturday_tn = compute_testing_prop_error(testing_saturdays, tn1_weekend)
error_sunday_tn = compute_testing_prop_error(testing_sundays, tn1_weekend)

def subplotCDFsubtractionErr(fig, ax, axx, axy, x, error, mean, title, day ):
    ax[axx,axy].plot(x, error, color="tomato", linewidth=2, zorder=10, label='Proportional error')
    ax[axx,axy].plot(x, mean, linewidth=1, linestyle='--' ,color='black', label='Mean error')
    ax[axx,axy].grid(linestyle='dotted')
    ax[axx,axy].set_ylabel('Proportional error %', fontsize=20)
    ax[axx,axy].set_xlabel('Hours', fontsize=20)
    ax[axx,axy].set_title(title , fontsize=20, pad=10)
    ax[axx,axy].set_ylim((0,1.1*max(error)))
    ax[axx,axy].legend(fontsize=16)
    ax[axx,axy].tick_params( labelsize=15)

    
fig, ax = plt.subplots(3, 3)
fig.set_figwidth(20)
fig.set_figheight(12)
fig.suptitle('Offline prediction through mathematical modeling ERROR - '+current_parking+' (Denormalized)', fontsize=24)
time = np.linspace(0,23.5,48)

mean_Monday =  [np.mean(error_monday_tn)]*len(time)
subplotCDFsubtractionErr(fig, ax, 0, 0, time, error_monday_tn, mean_Monday, 
                 'Proportional error Monday ', 'Monday')

mean_Tuesday =  [np.mean(error_tuesday_tn)]*len(time)
subplotCDFsubtractionErr(fig, ax, 0, 1, time, error_tuesday_tn, mean_Tuesday, 
                 'Proportional error Tuesday ', 'Tuesday')

mean_Wednesday =  [np.mean(error_wednesday_tn)]*len(time)
subplotCDFsubtractionErr(fig, ax, 0, 2, time, error_wednesday_tn, mean_Wednesday, 
                 'Proportional error Wednesday ', 'Wednesday')

mean_Thursday =  [np.mean(error_thursday_tn)]*len(time)
subplotCDFsubtractionErr(fig, ax, 1, 0, time, error_thursday_tn, mean_Thursday, 
                 'Proportional error Thursday ', 'Thursday')

mean_Friday =  [np.mean(error_friday_tn)]*len(time)
subplotCDFsubtractionErr(fig, ax, 1, 1, time, error_friday_tn, mean_Friday, 
                 'Proportional error Friday ', 'Friday')

mean_Saturday =  [np.mean(error_saturday_tn)]*len(time)
subplotCDFsubtractionErr(fig, ax, 1, 2, time, error_saturday_tn, mean_Saturday, 
                 'Proportional error Saturday ', 'Saturday')

mean_Sunday =  [np.mean(error_sunday_tn)]*len(time)
subplotCDFsubtractionErr(fig, ax, 2, 1, time, error_sunday_tn, mean_Sunday, 
                 'Proportional error Sunday ', 'Sunday')

ax[2,0].set_visible(False)
ax[2,2].set_visible(False)
fig.tight_layout(pad=5.0)
for ax in ax.flat:
    ax.set_ylabel('PROPORTIONAL ERROR %', fontsize=10)
    ax.set_xlabel('HOUR', fontsize=11)


print('______MEAN________')    

print(mean_Monday[0])
print(mean_Tuesday[0])
print(mean_Wednesday[0])
print(mean_Thursday[0])
print(mean_Friday[0])
print(mean_Saturday[0])
print(mean_Sunday[0])

print('______STDV________')

print(np.std(error_monday_tn))
print(np.std(error_tuesday_tn))
print(np.std(error_wednesday_tn))
print(np.std(error_thursday_tn))
print(np.std(error_friday_tn))
print(np.std(error_saturday_tn))
print(np.std(error_sunday_tn))


# ### MATHEMATICAL PROTOTYE: Real time prediction by SCALING

# In[216]:


# Plotting methods to reduce cell dimension

def real_timing_predition(fig, ax, axx, day, tn_proto, real_day, scaled_proto, Prototype, limit_hour, t_date):
    fig.suptitle('Real time prediction Scaling mathematical and mean proto for Testing '
                 + day + ' ' +t_date + ' ('+ current_parking+')', fontsize='18')
    ax[axx].plot(time, real_day.values, linestyle='dashdot', linewidth=2, label='Real ' + day)
    ax[axx].plot(time, tn_proto,'--',color='grey', label='TN prototype (not scaled)')
    ax[axx].plot(time, scaled_proto, color='green', linewidth=2, label='TN proto (scaled)')
    ax[axx].plot(time, Prototype, color='orange', linewidth=2, label='Stat. scaled proto')
    ax[axx].axvline(x=limit_hour, linestyle='--', color='grey', linewidth=2, label='Moment of prediction')
    ax[axx].axvspan(0, limit_hour, facecolor='grey', alpha=0.2, label='Known Activity')
    ax[axx].grid(linestyle='dotted', linewidth='0.5', color='grey')
    ax[axx].legend(fontsize=9)
    ax[axx].set_ylim([0,1.1*max([max(real_day.values),max(tn_proto),max(scaled_proto),max(Prototype)])])
    ax[axx].set_xlabel('Hour', fontsize=14)
    ax[axx].set_ylabel('Occupancy', fontsize=14)

def errors_plotting(fig, ax, axx, scaled_proto, Prototype, real_day, day, limit_hour):
    #Computing Errors
    limit_hour = limit_hour*2
    tn_scaled_error = (np.absolute((np.array(scaled_proto) - np.array(real_day.values)))/max_value)*100
    mean_scaled_error = (np.absolute((np.array(Prototype) - np.array(real_day.values)))/max_value)*100

    tn_s_error_mean = [np.mean(tn_scaled_error[limit_hour:])]*len(tn_scaled_error)
    mean_s_error_mean = [np.mean(mean_scaled_error[limit_hour:])]*len(mean_scaled_error)

    #Second plot
#     time = time[limit_hour:]
    ax[axx].plot(time[limit_hour:], tn_scaled_error[limit_hour:], color='tomato', label='TN scaling error')
    ax[axx].plot(time[limit_hour:],tn_s_error_mean[limit_hour:], '--',color='tomato', label='TN Mean prop. error')
    ax[axx].plot(time[limit_hour:],mean_scaled_error[limit_hour:], color='blueviolet', label='Proto scaling error')
    ax[axx].plot(time[limit_hour:],mean_s_error_mean[limit_hour:], '--',color='blueviolet', label='Proto. Mean prop. error')
    ax[axx].grid(linestyle='dotted', linewidth='0.5', color='grey')
    ax[axx].axvline(x=limit_hour/2, linestyle='--', color='grey', linewidth=2, label='Moment of prediction')
    ax[axx].axvspan(0, limit_hour/2, facecolor='grey', alpha=0.2, label='Known Activity', zorder=4)
    ax[axx].legend(fontsize=12)
    ax[axx].set_ylim([0,1.1*max(max(tn_scaled_error[limit_hour:]),max(mean_scaled_error[limit_hour:]))])
    ax[axx].set_xlabel('Hour', fontsize=14)
    ax[axx].set_ylabel('Proportional error (%)', fontsize=14)

    print('Real ' + day + ' scaled prtotype error: ', round(100*mean_s_error_mean[0])/100, '%')
    print('Real ' + day + ' scaled prtotype STDV: ', np.std(mean_scaled_error[limit_hour:]))

    print('Real ' + day + ' scaled TN error: ', round(100*tn_s_error_mean[0])/100, '%')
    print('Real ' + day + ' scaled TN STDV: ', np.std(tn_scaled_error[limit_hour:]))
    print('_____________________________________________________________')

def get_scaling_factor(limit_hour, test_day, proto):
    if limit_hour < 6:
        return 1
    index = limit_hour*2
    current_real_data = test_day.values[index]
    proto_value = proto[index]
    scaling = current_real_data/proto_value
    return scaling


# In[280]:

    


# load prottypes :
with open('data/'+current_parking+'_proto.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [hist_weekday_proto, hist_friday_proto, hist_weekend_proto]= pickle.load(f)
    f.close() 

def plot_prototype():
    for i in range(0,len(tn_proto)):
        if tn_proto[i] < 0:
            tn_proto[i] = 0
    cont=0
    for i in range(0,len(t_days)):
        t_date=t_dates[cont]
        print(t_date)
        tn_scaling = get_scaling_factor(limit_hour, t_days[i], tn_proto)
        tn_arr_scaling = get_scaling_factor(limit_hour, t_days[i], tn_arr_proto)
        stat_scaling = get_scaling_factor(limit_hour, t_days[i], hist_weekday_proto.values)

        scaled_tn_proto = tn_proto * tn_scaling
        scaled_tn_arr_proto = tn_arr_proto * tn_arr_scaling
        scaled_tn_dep_proto = tn_dep_proto * tn_arr_scaling

        if max(scaled_tn_arr_proto)>max_value:
            cars_could_not_park=max(scaled_tn_arr_proto[scaled_tn_arr_proto >max_value])-max_value
            print(round(cars_could_not_park), "cars could not park")
            scaled_tn_arr_proto[scaled_tn_arr_proto >max_value]=max_value
            scaled_tn_dep_proto=scaled_tn_dep_proto/max(scaled_tn_dep_proto)*(max_value-weekday_offset)
        scaled_tn_proto2=scaled_tn_arr_proto-scaled_tn_dep_proto

        scaled_stat_proto = hist_weekday_proto.values * stat_scaling
        fig, ax = plt.subplots(1, 2)
        fig.set_figwidth(20)
        fig.set_figheight(5)
        axx=0;
        real_timing_predition(fig, ax, axx, day, tn_proto, t_days[i], scaled_tn_proto2, scaled_stat_proto, limit_hour, t_date)
        axx=1;
        errors_plotting(fig, ax, axx, scaled_tn_proto2, scaled_stat_proto, t_days[i], day, limit_hour)
        cont=cont+1


# #### MONDAY

# In[279]:


statistic_proto = hist_weekday_proto
tn_proto = tn_weekday
tn_arr_proto = tn_arr_weekday
tn_dep_proto = tn_dep_weekday
t_days = testing_mondays
t_dates=testing_mondays_dates
limit_hour = 8
day = 'Monday'
# Negative values to 0
for i in range(0,len(tn_proto)):
    if tn_proto[i] < 0:
        tn_proto[i] = 0
cont=0
for i in range(0,len(t_days)):
    t_date=t_dates[cont]
    print(t_date)
    tn_scaling = get_scaling_factor(limit_hour, t_days[i], tn_proto)
    tn_arr_scaling = get_scaling_factor(limit_hour, t_days[i], tn_arr_proto)
    stat_scaling = get_scaling_factor(limit_hour, t_days[i], hist_weekday_proto.values)
    
    scaled_tn_proto = tn_proto * tn_scaling
    scaled_tn_arr_proto = tn_arr_proto * tn_arr_scaling
    scaled_tn_dep_proto = tn_dep_proto * tn_arr_scaling
    
    if max(scaled_tn_arr_proto)>max_value:
        cars_could_not_park=max(scaled_tn_arr_proto[scaled_tn_arr_proto >max_value])-max_value
        print(round(cars_could_not_park), "cars could not park")
        scaled_tn_arr_proto[scaled_tn_arr_proto >max_value]=max_value
        scaled_tn_dep_proto=scaled_tn_dep_proto/max(scaled_tn_dep_proto)*(max_value-weekday_offset)
    scaled_tn_proto2=scaled_tn_arr_proto-scaled_tn_dep_proto
    
    scaled_stat_proto = hist_weekday_proto.values * stat_scaling
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    axx=0;
    real_timing_predition(fig, ax, axx, day, tn_proto, t_days[i], scaled_tn_proto2, scaled_stat_proto, limit_hour, t_date)
    axx=1;
    errors_plotting(fig, ax, axx, scaled_tn_proto2, scaled_stat_proto, t_days[i], day, limit_hour)
    cont=cont+1


# #### TUESDAY

# In[283]:


statistic_proto = hist_weekday_proto
tn_proto = tn_weekday
tn_arr_proto = tn_arr_weekday
tn_dep_proto = tn_dep_weekday
t_days = testing_tuesdays
t_dates= testing_tuesdays_dates
limit_hour = 8
day = 'Tuesday'
# Negative values to 0

plot_prototype()


# ### WEDNESDAY

# In[285]:


statistic_proto = hist_weekday_proto
tn_proto = tn_weekday
t_days = testing_wednesdays
t_dates= testing_wednesdays_dates
limit_hour = 8
day = 'Wednesday'

tn_arr_proto = tn_arr_weekday
tn_dep_proto = tn_dep_weekday
plot_prototype()


# ### THURSDAY

# In[286]:


statistic_proto = hist_weekday_proto
tn_proto = tn_weekday
t_days = testing_thursdays
t_dates= testing_thursdays_dates
limit_hour = 8
day = 'Thursday'

tn_arr_proto = tn_arr_weekday
tn_dep_proto = tn_dep_weekday
plot_prototype()


# ### FRIDAY

# In[289]:


statistic_proto = hist_friday_proto
tn_proto = tn_friday
t_days = testing_fridays
t_dates=testing_fridays_dates
limit_hour = 8
day = 'Friday'

tn_arr_proto = tn_arr_friday
tn_dep_proto = tn_dep_friday
plot_prototype()
    


# ### WEEKEND

# ### SATURDAY

# In[290]:


statistic_proto = hist_weekend_proto
tn_proto = tn1_weekend
t_days = testing_saturdays
t_dates= testing_saturdays_dates
limit_hour = 16
day = 'Saturday'
# Negative values to 0
for i in range(0,len(tn_proto)):
    if tn_proto[i] < 0:
        tn_proto[i] = 0
        
cont=0
for i in range(0,len(t_days)):
    t_date=t_dates[cont]
    print(t_date)
    tn_scaling = get_scaling_factor(limit_hour, t_days[i], tn_proto)
    stat_scaling = get_scaling_factor(limit_hour, t_days[i], statistic_proto.values)
    
    scaled_tn_proto = tn_proto * tn_scaling
    scaled_stat_proto = statistic_proto.values * stat_scaling
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(20)
    fig.set_figheight(6)
    axx=0;
    real_timing_predition(fig, ax, axx, day, tn_proto, t_days[i], scaled_tn_proto, scaled_stat_proto, limit_hour, t_date)
    #ax[0].set_ylim([0,axis_ylim_we])
    axx=1;
    errors_plotting(fig, ax, axx, scaled_tn_proto, scaled_stat_proto, t_days[i], day, limit_hour)
    cont=cont+1


# ### SUNDAY

# In[291]:


statistic_proto = hist_weekend_proto
tn_proto = tn1_weekend
t_days = testing_sundays
t_dates= testing_sundays_dates
limit_hour = 13
day = 'Sunday'
# Negative values to 0
for i in range(0,len(tn_proto)):
    if tn_proto[i] < 0:
        tn_proto[i] = 0
    
cont=0
for i in range(0,len(t_days)):
    t_date=t_dates[cont]
    print(t_date)
    tn_scaling = get_scaling_factor(limit_hour, t_days[i], tn_proto)
    stat_scaling = get_scaling_factor(limit_hour, t_days[i], statistic_proto.values)
    
    scaled_tn_proto = tn_proto * tn_scaling
    scaled_stat_proto = statistic_proto.values * stat_scaling
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(20)
    fig.set_figheight(6)
    axx=0;
    real_timing_predition(fig, ax, axx, day, tn_proto, t_days[i], scaled_tn_proto, scaled_stat_proto, limit_hour, t_date)
    #ax[0].set_ylim([0,axis_ylim_we])
    axx=1;
    errors_plotting(fig, ax, axx, scaled_tn_proto, scaled_stat_proto, t_days[i], day, limit_hour)
    cont=cont+1


# In[292]:


def printTimes(params,timeString='WEEKDAYS'):
    print("--------- "+timeString +" "+current_parking+" -----------")
    loc_ar = params[0]*24
    scale_ar = params[1]*24
    loc_de = params[2]*24
    scale_de = params[3]*24

    print(f'Mean Arrival Time   = {int(loc_ar):02.0f}:{int((loc_ar-int(loc_ar))*60):02.0f}h')
    print(f'stdv Arrival        = {int(scale_ar):2.0f}:{int((scale_ar-int(scale_ar))*60):02.0f}h')
    print(f'Mean Departure Time = {int(loc_de):02.0f}:{int((loc_de-int(loc_de))*60):02.0f}h')
    print(f'stdv Departure      = {int(scale_de):2.0f}:{int((scale_de-int(scale_de))*60):02.0f}h')
    if len(params)>4:
        thresh=params[4]
        if thresh<1:
            cdf_ar = tn_cdf(time_tn, loc_ar/24, scale_ar/24)
            time_parking_full= 0.5*np.argmax(cdf_ar>thresh)
            print(f'Parking full        = {int(time_parking_full):02.0f}:{int((time_parking_full-int(time_parking_full))*60):02.0f}h')


# In[293]:


printTimes(optimal_params_weekdaytn.x,'WEEKDAYS')
printTimes(optimal_params_fridaytn.x,'FRIDAYS')
printTimes(optimal_params_weekendtn.x,'WEEKENDS')


# # Store protos, params, areas and offsets

# In[179]:


# df_prototypes = pd.read_csv('data/prototypes_new.csv')
# index = 0

# weekday_tn_params = ','.join(str(e) for e in list(res_weekday.x))
# friday_tn_params  = ','.join(str(e) for e in list(res_friday.x))
# weekend_tn_params = ','.join(str(e) for e in list(res_weekend.x))

# weekday_mean_area = weekday_area
# friday_mean_area  = friday_area
# weekend_mean_area = weekend_area

# weekday_mean_offset = weekday_offset
# friday_mean_offset  = friday_offset
# weekend_mean_offset = weekend_offset

# total_wave_weekday_s = ','.join(str(e) for e in tn_weekday.tolist())
# total_wave_weekend_s = ','.join(str(e) for e in tn1_weekend.tolist())
# total_wave_friday_s  = ','.join(str(e) for e in tn_friday.tolist())

# historical_weekday_s = ','.join(str(e) for e in hist_weekday_proto.tolist())
# historical_weekend_s = ','.join(str(e) for e in hist_weekend_proto.tolist())
# historical_friday_s  = ','.join(str(e) for e in hist_friday_proto.tolist())


# df_prototypes.at[index,'CDF subtraction_weekday_proto'] = total_wave_weekday_s
# df_prototypes.at[index,'CDF subtraction_friday_proto']  = total_wave_friday_s
# df_prototypes.at[index,'CDF subtraction_weekend_proto'] = total_wave_weekend_s

# df_prototypes.at[index,'CDF subtraction_weekday_params'] = weekday_tn_params
# df_prototypes.at[index,'CDF subtraction_friday_params']  = friday_tn_params
# df_prototypes.at[index,'CDF subtraction_weekend_params'] = weekend_tn_params

# df_prototypes.at[index,'CDF subtraction_weekday_area'] = weekday_mean_area
# df_prototypes.at[index,'CDF subtraction_friday_area']  = friday_mean_area
# df_prototypes.at[index,'CDF subtraction_weekend_area'] = weekend_mean_area

# df_prototypes.at[index,'CDF subtraction_weekday_offset'] = weekday_mean_offset
# df_prototypes.at[index,'CDF subtraction_friday_offset']  = friday_mean_offset
# df_prototypes.at[index,'CDF subtraction_weekend_offset'] = weekend_mean_offset

# df_prototypes.at[index,'Historical_weekday_proto'] = historical_weekday_s
# df_prototypes.at[index,'Historical_weekend_proto'] = historical_weekend_s
# df_prototypes.at[index,'Historical_friday_proto']  = historical_friday_s

# df_prototypes.to_csv("data/final_prototypes.csv", index=False)
# df_prototypes


# In[180]:


# How to read the prorotypes: 
# string = exportable_df.at[0,'CDF subtraction_weekday_proto']
# list_of_strings = string.split(',')
# final_list = list(np.float_(list_of_strings))


# In[181]:


# import pandas as pd
# df_prototypes_2 = pd.read_csv('data/prototypes_new.csv')
# new_row = {'Location':'Cerdanyola'}
# #append row to the dataframe
# df_prototypes_2 = df_prototypes_2.append(new_row, ignore_index=True)
# df_prototypes_2


# In[182]:


# df_prototypes_2.to_csv("data/prototypes_new.csv", index=False)
# day = 'Monday'
# date = '2020-02-24'
# real_day = mean_of_day(day,date)
# real_day.values


# In[183]:


# hist_friday_proto.tolist()

