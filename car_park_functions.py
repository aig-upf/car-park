#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:49:06 2021

@author: andreas
"""

from matplotlib import pyplot as plt
from scipy.stats import truncnorm 
import numpy as np

import calendar

def getDayName(d):
    return calendar.day_name[d.weekday()]

def get_days_of_protos_normalized(proto_name, df_):
    data_temp = df_[df_['Profile_3'] == proto_name] 
    days = []
    for i in range(0,data_temp.shape[0], 48):
        day = data_temp['Normalized_occupancy'][i:i+48]
        if len(day) == 48:
            days.append(day)
        
    return days

def get_parkingfull_of_protos(proto_name, df_):
    data_temp = df_[df_['Profile_3'] == proto_name] 
    isfull = []
    for i in range(0,data_temp.shape[0], 48):
        intervallIsfull = data_temp['Free slots'][i:i+48]==0
        if len(intervallIsfull) == 48:
            isfull.append(max(intervallIsfull))
    return isfull

def classify_2_proto(x):
    weekday_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    if x in weekday_list:
        return 'Weekday'
    else:
        return 'Weekend'


def classify_3_proto(x):
    weekday_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
    weekend_list = ['Saturday', 'Sunday']
    if x in weekday_list:
        return 'Weekday'
    elif x in weekend_list:
        return 'Weekend'
    else:
        return 'Friday'
    

# 1 week data = 336 rows (48*7) WE ASUME WE HAVE COMPLETE DATA WITHOUT GAPS BETWEEN HOURS, AND DATA FROM 0.0 TO 23.5
# @from_end: if the testing data is retrieved from the last part of the df or from th beginning. By default, from the end
# @limit_date: if we want to get the train from specific date instead of by number of weeks

def split_data(df, n_test_weeks, limit_date = None, from_end=True): 
    if(limit_date != None):
        init_test = df[df['Date']==limit_date].index[0];
        end_test = df.shape[0]
        test_domain = range(init_test, end_test)
        test_domain = list(test_domain)

    else:
        if(from_end == True):
            end_test = df.shape[0]
            week_domain = n_test_weeks*336
            init_test = end_test-week_domain
            test_domain = range(init_test, end_test)
            test_domain = list(test_domain)
            
        elif(from_end == False):
            week_domain = n_test_weeks*336
            init_test = 0
            end_test = week_domain
            test_domain = range(0,week_domain)
            test_domain = list(test_domain)
            
    test_df = df[init_test:end_test]
    training_df = df.drop(test_domain)
    return training_df, test_df




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


def model_tn_areaN_args(params,trainining_norm,errors): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
 
    num_training_days = len(trainining_norm)
    time_tn = np.linspace(0,23.5,48)/24
    
    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)

    res = cdf_ar - cdf_de
    res_n = res/sum(res)
  

    #error = 0  
    for ii in range(0,num_training_days):
        day = trainining_norm[ii]
        #error += np.sum(np.power(res_n - day, 2))
        errors[ii,:] = np.power(res_n - day, 2)
    #return error
    return np.sum(errors)

def model_tn_th_max_args(params,trainining_norm,errors): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
    thresh=params[4]

    num_training_days = len(trainining_norm)
    time_tn = np.linspace(0,23.5,48)/24

    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)

    cdf_ar[cdf_ar>thresh] = thresh
    cdf_ar = cdf_ar/thresh

    res = cdf_ar - cdf_de
    res_n = res 
  
    for ii in range(0,num_training_days):
        day = trainining_norm[ii]
        errors[ii,:] = np.power(res_n - day, 2)
    return  np.sum(errors)


def model_tn_th_ind_max(params,trainining_norm,training_isfull,errors): 
    loc_ar = params[0]
    scale_ar = params[1]
    loc_de = params[2]
    scale_de = params[3]
 
    num_training_days = len(trainining_norm)
    time_tn = np.linspace(0,23.5,48)/24

    cdf_ar=tn_cdf(time_tn, loc_ar, scale_ar)
    cdf_de=tn_cdf(time_tn, loc_de, scale_de)


    res = cdf_ar - cdf_de
    res_n = res
    

    for ii in range(0,num_training_days):
        day = trainining_norm[ii]
        dayisFull=training_isfull[ii]
        
        if dayisFull:
            thresh=params[4+ii]
            cdf_ar_th=cdf_ar
            cdf_ar_th[cdf_ar>thresh] = thresh
            cdf_ar_th = cdf_ar_th/thresh
    
            res_th = cdf_ar_th - cdf_de
            res_th_n = res_th#/sum(res_th)
            
            errors[ii,:] = np.power(res_th_n - day, 2)
        else:
            errors[ii,:] = np.power(res_n - day,2)      
    return np.sum(errors)



def subplot_training(fig, ax, xx, yy, proto_data, test_days, day, proto_name,axis_ylim): 
    time = np.linspace(0,23.5,48)
    ax[xx,yy].plot(time, proto_data, linewidth=3, linestyle='dashed', label= proto_name + ' TN prediction')
    for i in range(0, len(test_days)): 
        ax[xx, yy].plot(time, test_days[i], linewidth=1, label='Testing ' + day )
        if i==0:
            ax[xx,yy].legend(fontsize=16)
    ax[xx,yy].grid(linestyle='dotted')
    ax[xx,yy].set_ylim(-2,axis_ylim)
    ax[xx,yy].set_xlabel('Time (hours)', fontsize=16)
    ax[xx,yy].set_ylabel('Occupancy', fontsize=16)
    
    
def plot_model_tn(loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1):
    # arrivals
    a_ar = -loc_ar/scale_ar
    b_ar = (1-loc_ar)/scale_ar
    
    # departures
    a_de = -loc_de/scale_de
    b_de = (1-loc_de)/scale_de
    
    time = np.linspace(0,23.5,48)
    time_tn=time/24
    
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
       
def plot_model_tn_th(loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1,thresh=.8):
    # arrivals
    a_ar = -loc_ar/scale_ar
    b_ar = (1-loc_ar)/scale_ar

    # departures
    a_de = -loc_de/scale_de
    b_de = (1-loc_de)/scale_de

    time = np.linspace(0,23.5,48)
    time_tn=time/24

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
    