#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:49:06 2021

@author: andreas
"""

from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
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


def get_days_of_protos_areanormalized(proto_name, df_):
    data_temp = df_[df_['Profile_3'] == proto_name]
    days = []
    for i in range(0,data_temp.shape[0], 48):
        day = data_temp['Area_Normalized_occupancy'][i:i+48]
        if len(day) == 48:
            days.append(day)

    return days


def get_days_of_protos(proto_name, df_):
    data_temp = df_[df_['Profile_3'] == proto_name]
    days = []
    for i in range(0,data_temp.shape[0], 48):
#        day = data_temp['Occupancy_mod'][i:i+48]
        day = data_temp['Occupancy'][i:i+48]
        if len(day) == 48:
            days.append(day)

    return days



def get_dates_of_protos(proto_name, df_):
    data_temp = df_[df_['Profile_3'] == proto_name]
    dates = []
    for i in range(0,data_temp.shape[0], 48):
        day = data_temp['Occupancy'][i:i+48]
        t_date = data_temp.iloc[i]['Date']
        if len(day) == 48:
            dates.append(t_date)
    return dates


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

def model_tn_max_args(params,trainining_norm,errors):
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


def plot_model_tn_thDisc(loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1,thresh=.8):
    # arrivals
    fsize=9

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
    t_parking_full=np.interp(thresh,cdf_ar,time)
    time_th=np.insert(time,ix_parking_full,t_parking_full)

    pdf_ar_orig=pdf_ar.copy()
    pdf_ar=np.insert(pdf_ar,ix_parking_full,np.interp(t_parking_full,time,pdf_ar))
    pdf_ar_excees=pdf_ar.copy()
    pdf_ar[ix_parking_full+1:] =0
    pdf_ar_excees[:ix_parking_full] =0

    masktn_arr = np.ones(len(pdf_ar), dtype=bool)
    masktn_arr[ix_parking_full+1:] =False
    masktn_arr_excees = np.ones(len(pdf_ar), dtype=bool)
    masktn_arr_excees[:ix_parking_full] =False


    cdf_ar=np.insert(cdf_ar,ix_parking_full,thresh)
    cdf_ar_withExcess=cdf_ar.copy()
    cdf_ar[cdf_ar>thresh] = thresh
    cdf_ar = cdf_ar/thresh
    cdf_ar_withExcess=cdf_ar_withExcess/thresh

    cdf_de = truncnorm.cdf(time_tn, a_de, b_de, loc=loc_de, scale=scale_de)
    cdf_de=np.insert(cdf_de,ix_parking_full,np.interp(t_parking_full,time,cdf_de))

    ymin=-0.01
    ymax=max(pdf_ar_orig/sum(pdf_ar_orig))*1.05

    #fig, ax = plt.subplots(2,figsize=(18,10))
    fig, ax = plt.subplots(2,figsize=(7,4))
    ax[0].plot(time_th[masktn_arr], pdf_ar[masktn_arr]/sum(pdf_ar_orig) , '-b',label='PDF arrivals')
    ax[0].plot(time, pdf_de/sum(pdf_de), '-r',label='PDF departures')
    ax[0].plot(t_parking_full*np.array([1, 1]),[ymin,ymax],'--',linewidth=2,label="Lot Full")
    ax[0].plot(time_th[masktn_arr_excees], pdf_ar_excees[masktn_arr_excees]/sum(pdf_ar_orig) , '-.k',
               label='Surplus (PDF)')
    ax[0].set_title('PDFs', fontsize=fsize)
    ax[0].legend(fontsize=fsize, loc="upper center");
    ax[0].set_ylabel('probability', fontsize = fsize)
    ax[0].grid(which='major',linestyle='dotted')
    ax[0].set_xlim([0,23.5])
    ax[0].set_ylim([ymin,ymax])
    ax[0].tick_params(axis='both', which='major', labelsize=fsize)

    ymin=-0.06
    ymax=max(cdf_ar_withExcess)*1.05
    #ax[1].plot(time_th, cdf_ar , '--b',label='CDF arrivals')
    h2,=ax[1].plot(time_th, cdf_de, '--r',label='CDF departures')
    h3,=ax[1].plot(time_th, cdf_ar-cdf_de, 'r',linewidth=3,label='TNL model')
    h4,=ax[1].plot(t_parking_full*np.array([1, 1]),[ymin,ymax],'--',linewidth=2,label="Lot Full")
    h5,=ax[1].plot(time_th[masktn_arr_excees], cdf_ar_withExcess[masktn_arr_excees] , '-.k',
               label='Surplus (CDF)')
    h1,=ax[1].plot(time_th, cdf_ar , '--b',label='CDF arrivals')
    ax[1].set_title('CDFs', fontsize=fsize)
    ax[1].legend(handles=[h1, h2, h3, h4, h5],fontsize=fsize, loc="upper left")
    ax[1].grid(which='major',linestyle='dotted')
    ax[1].set_xlim([0,23.5])
    ax[1].set_ylim([ymin,ymax])
    ax[1].set_xlabel('hour', fontsize = fsize)
    ax[1].set_ylabel('occupancy', fontsize = fsize)
    ax[1].tick_params(axis='both', which='major', labelsize=fsize)

    return fig

def printTimes(params,current_parking,timeString='WEEKDAYS'):
    print("--------- "+timeString +" "+current_parking+" -----------")
    loc_ar = params[0]*24
    scale_ar = params[1]*24
    loc_de = params[2]*24
    scale_de = params[3]*24

    time = np.linspace(0,23.5,48)
    time_tn=time/24

    print(f'Mean Arrival Time   = {int(loc_ar):02.0f}:{int((loc_ar-int(loc_ar))*60):02.0f}h')
    print(f'stdv Arrival        = {int(scale_ar):2.0f}:{int((scale_ar-int(scale_ar))*60):02.0f}h')
    print(f'Mean Departure Time = {int(loc_de):02.0f}:{int((loc_de-int(loc_de))*60):02.0f}h')
    print(f'stdv Departure      = {int(scale_de):2.0f}:{int((scale_de-int(scale_de))*60):02.0f}h')
    if len(params)>4:
        thresh=np.mean(params[4:])
        if thresh<1:
            cdf_ar = tn_cdf(time_tn, loc_ar/24, scale_ar/24)
            time_parking_full= 0.5*np.argmax(cdf_ar>thresh)
            print(f'Parking full        = {int(time_parking_full):02.0f}:{int((time_parking_full-int(time_parking_full))*60):02.0f}h')


def compute_testing_prop_errorM(testing_days, proto_data, m_value):
    errors = np.zeros(48)
    n_test_days = len(testing_days)
    proto = np.array(proto_data)

    for i in range(0, n_test_days):
        day = np.array(testing_days[i])
        er = np.array((np.absolute(proto - day)/m_value)*100)
        errors += er
    return errors/n_test_days

def compute_testing_prop_errorMstdv(testing_days, proto_data, m_value):
    errors = np.zeros(48)
    errors2=np.zeros(48)
    n_test_days = len(testing_days)
    proto = np.array(proto_data)

    for i in range(0, n_test_days):
        day = np.array(testing_days[i])
        er = np.array((np.absolute(proto - day)/m_value)*100)
        errors += er
        errors2 += er*er
    meanE=errors/n_test_days
    stdvE=np.sqrt(errors2/n_test_days-meanE*meanE)
    return [meanE, stdvE]

def compute_testing_prop_errorMstdv_fit(testing_days, proto_data, m_value):
    errors = np.zeros(48)
    errors2=np.zeros(48)
    n_test_days = len(testing_days)
    proto = np.array(proto_data)

    #fig, ax = plt.subplots(n_test_days,1)
    for i in range(0, n_test_days):
        day = testing_days[i]
        f_scaling = get_scaling_factor_and_constant(24, day, proto)
        scaled_proto = proto * f_scaling.x[1]+f_scaling.x[0]
        #time = np.linspace(0,23.5,48)
        #ax[i].plot(time,scaled_proto)
        #ax[i].plot(time,day)
        er = np.array((np.absolute(scaled_proto - day)/m_value)*100)
        errors += er
        errors2 += er*er
    meanE=errors/n_test_days
    stdvE=np.sqrt(errors2/n_test_days-meanE*meanE)
    return [meanE, stdvE]

def compute_testing_prop_errorMstdv_fitTh(testing_days, tn_arr_proto, tn_dep_proto, m_value):
    errors = np.zeros(48)
    errors2=np.zeros(48)
    n_test_days = len(testing_days)
    #proto = np.array(proto_data)

    #fig, ax = plt.subplots(n_test_days,1)
    for i in range(0, n_test_days):
        day = testing_days[i]

        tn_arr_scaling = get_scaling_factor_and_constantTH(24, day, tn_arr_proto)
        scaled_tn_arr_proto = tn_arr_proto * tn_arr_scaling.x[1]+tn_arr_scaling.x[0]
        scaled_tn_dep_proto = tn_dep_proto * tn_arr_scaling.x[1]


        if max(scaled_tn_arr_proto)>m_value:
            cars_could_not_park=max(scaled_tn_arr_proto[scaled_tn_arr_proto >m_value])-m_value
            print(round(cars_could_not_park), "cars could not park")
            scaled_tn_arr_proto[scaled_tn_arr_proto >m_value]=m_value
            scaled_tn_dep_proto=scaled_tn_dep_proto/max(scaled_tn_dep_proto)*(m_value-tn_arr_scaling.x[0])
        scaled_proto=scaled_tn_arr_proto-scaled_tn_dep_proto

        #time = np.linspace(0,23.5,48)
        #ax[i].plot(time,scaled_proto)
        #ax[i].plot(time,day)
        er = np.array((np.absolute(scaled_proto - day)/m_value)*100)
        errors += er
        errors2 += er*er
    meanE=errors/n_test_days
    stdvE=np.sqrt(errors2/n_test_days-meanE*meanE)
    return [meanE, stdvE]

def compute_testing_prop_errorMstdv_fitThdep(testing_days, tn_arr_proto, tn_dep_proto, m_value):
    errors = np.zeros(48)
    errors2=np.zeros(48)
    n_test_days = len(testing_days)
    #proto = np.array(proto_data)

    #fig, ax = plt.subplots(n_test_days,1)
    for i in range(0, n_test_days):
        day = testing_days[i]

        tn_arr_scaling = get_scaling_factor_and_constantTH(24, day, tn_arr_proto)
        scaled_tn_arr_proto = tn_arr_proto * tn_arr_scaling.x[1]+tn_arr_scaling.x[0]
        #scaled_tn_dep_proto = tn_dep_proto * tn_arr_scaling.x[1]

        tn_dep_scaling = get_scaling_factor_dep(24, day, tn_dep_proto,m_value)
        scaled_tn_dep_proto = tn_dep_proto * tn_dep_scaling.x[0]


        if max(scaled_tn_arr_proto)>m_value:
            cars_could_not_park=max(scaled_tn_arr_proto[scaled_tn_arr_proto >m_value])-m_value
            print(round(cars_could_not_park), "cars could not park")
            scaled_tn_arr_proto[scaled_tn_arr_proto >m_value]=m_value
            #scaled_tn_dep_proto=scaled_tn_dep_proto/max(scaled_tn_dep_proto)*(m_value-tn_arr_scaling.x[0])
        scaled_proto=scaled_tn_arr_proto-scaled_tn_dep_proto

        #time = np.linspace(0,23.5,48)
        #ax[i].plot(time,scaled_proto)
        #ax[i].plot(time,day)
        er = np.array((np.absolute(scaled_proto - day)/m_value)*100)
        errors += er
        errors2 += er*er
    meanE=errors/n_test_days
    stdvE=np.sqrt(errors2/n_test_days-meanE*meanE)
    return [meanE, stdvE]

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

def subplotCDFsubtractionStdvErr(fig, ax, axx, axy, x, error, std_error, mean, title, day,
                                 bolxlim=True,bol_ylabel=True,bol_xlabel=True, bol_legend=True):
    ax[axx,axy].plot(x, error, color="r", linewidth=2, zorder=10, label='Proportional error')
    ax[axx,axy].plot(x, error+std_error, color="b", linestyle='--', linewidth=1, zorder=10, label='± stdv')
    ax[axx,axy].plot(x, error-std_error, color="b", linestyle='--', linewidth=1, zorder=10)
    ax[axx,axy].plot(x, mean, linewidth=2, linestyle='-.' ,color='black', label='Mean error')
    ax[axx,axy].grid(linestyle='dotted')
    if bol_ylabel:
        ax[axx,axy].set_ylabel('Proportional error %', fontsize=20)
    if bol_xlabel:
        ax[axx,axy].set_xlabel('Time [h]', fontsize=20)
    if not bolxlim:
        ax[axx,axy].xaxis.set_ticklabels([])
    ax[axx,axy].set_title(title , fontsize=20, pad=10)
    ax[axx,axy].set_ylim((0,1.1*max(error+std_error)))
    ax[axx,axy].set_xlim((0,24))
    if bol_legend:
         ax[axx,axy].legend(bbox_to_anchor=(0.7,0.15), loc="lower right",
                bbox_transform=fig.transFigure, ncol=1,fontsize=16)
    ax[axx,axy].tick_params( labelsize=15)

def real_timing_predition(fig, ax, axx, day, tn_proto, real_day, scaled_proto, Prototype, limit_hour, t_date, current_parking="LABEL PARKING"):
    time = np.linspace(0,23.5,48)
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

def errors_plottingM(fig, ax, axx, scaled_proto, Prototype, real_day, day, limit_hour, m_value):
    #Computing Errors
    time = np.linspace(0,23.5,48)
    limit_hour = int(limit_hour*2)
    tn_scaled_error = (np.absolute((np.array(scaled_proto) - np.array(real_day.values)))/m_value)*100
    mean_scaled_error = (np.absolute((np.array(Prototype) - np.array(real_day.values)))/m_value)*100
    #normalizador=np.array(real_day.values)
    #normalizador[normalizador==0]=1;
    #tn_scaled_error = (np.absolute((np.array(scaled_proto) - np.array(real_day.values)))/normalizador)*100
    #mean_scaled_error = (np.absolute((np.array(Prototype) - np.array(real_day.values)))/normalizador)*100



    tn_s_error_mean = [np.mean(tn_scaled_error[limit_hour:])]*len(tn_scaled_error)
    mean_s_error_mean = [np.mean(mean_scaled_error[limit_hour:])]*len(mean_scaled_error)

    tn_s_error_median = np.median(tn_scaled_error[limit_hour:])
    mean_s_error_median = np.median(mean_scaled_error[limit_hour:])
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

    print('Real ' + day + ' scaled prototype error: ', round(100*mean_s_error_mean[0])/100, '%')
    print('Real ' + day + ' scaled prototype error (median): ', round(100*mean_s_error_median)/100, '%')
    print('Real ' + day + ' scaled prototype STDV:', np.std(mean_scaled_error[limit_hour:]))

    print('Real ' + day + ' scaled TN error: ', round(100*tn_s_error_mean[0])/100, '%')
    print('Real ' + day + ' scaled TN error (median): ', round(100*tn_s_error_median)/100, '%')
    print('Real ' + day + ' scaled TN STDV: ', np.std(tn_scaled_error[limit_hour:]))
    print('_____________________________________________________________')

def get_scaling_factor(limit_hour, test_day, proto):
    if limit_hour < 6:
        return 1
    index = int(limit_hour*2)
    current_real_data = test_day.values[index]
    proto_value = proto[index]
    scaling = current_real_data/proto_value
    return scaling



def generate_mean_variance(accumulated_date, accumulated_free_slots):
    aux_dict = {};
    for ii in np.arange(len(accumulated_date)):
        free_slots = list(accumulated_free_slots[ii].iat)
        hour = list(accumulated_free_slots[ii].index)
        for jj in np.arange(len(free_slots)):
            current_hour = hour[jj]
            if current_hour not in aux_dict:
                aux_dict[current_hour] = []
            aux_dict[current_hour].append(free_slots[jj])
    return aux_dict

def compute_mean_variance(aux_dict):
    domain = np.linspace(0,23,47)
    var_vec = []; mean_vec = []; hour_vec=[];
    for ii in domain:
        if ii in aux_dict:
            var_vec.append(np.var(list(aux_dict[ii])))
            mean_vec.append(np.mean(list(aux_dict[ii])))
            hour_vec.append(ii)
    return var_vec, mean_vec, hour_vec

def calcRunningPredcitionError(t_days,statistic_proto,tn_proto,max_value,starting_hour=7):
    limit_hour_vec = np.arange (starting_hour, 23, 0.5)
    tn_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))
    proto_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))

    for i in range(0,len(t_days)):
        cont=0
        for limit_hour in limit_hour_vec:
            tn_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], tn_proto)
            stat_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], statistic_proto.values)
            scaled_tn_proto = tn_proto * tn_scaling.x[1]+tn_scaling.x[0]
            scaled_stat_proto = statistic_proto.values * stat_scaling.x[1]+stat_scaling.x[0]
            [tn_running_error_vec[cont,i],proto_running_error_vec[cont,i]]=errors_calc_max(scaled_tn_proto, scaled_stat_proto, t_days[i], limit_hour, max_value)
            cont=cont+1
    return [tn_running_error_vec,proto_running_error_vec]

def calcRunningPredcitionErrorNow(t_days,statistic_proto,tn_proto,max_value,starting_hour=7,window_lenth=2,ending_hour=23):
    limit_hour_vec = np.arange (starting_hour, ending_hour, 0.5)
    tn_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))
    proto_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))

    for i in range(0,len(t_days)):
        cont=0
        for limit_hour in limit_hour_vec:
            tn_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], tn_proto)
            stat_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], statistic_proto.values)
            scaled_tn_proto = tn_proto * tn_scaling.x[1]+tn_scaling.x[0]
            scaled_stat_proto = statistic_proto.values * stat_scaling.x[1]+stat_scaling.x[0]
            [tn_running_error_vec[cont,i],proto_running_error_vec[cont,i]]=errors_calc_max_Now(scaled_tn_proto, scaled_stat_proto, t_days[i], limit_hour, max_value, window_lenth)
            cont=cont+1
    return [tn_running_error_vec,proto_running_error_vec]

def calcRunningPredcitionErrorNowTHv2(t_days,statistic_proto,tn_arr_proto,tn_dep_proto,
                                      max_value,starting_hour=7,window_lenth=2,ending_hour=23):
    limit_hour_vec = np.arange (starting_hour, ending_hour, 0.5)
    tn_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))
    proto_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))

    for i in range(0,len(t_days)):
        cont=0
        for limit_hour in limit_hour_vec:
            tn_arr_scaling = get_scaling_factor_and_constantTH(limit_hour, t_days[i], tn_arr_proto)
            stat_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], statistic_proto.values)

            scaled_tn_arr_proto = tn_arr_proto * tn_arr_scaling.x[1]+tn_arr_scaling.x[0]
            scaled_tn_dep_proto = tn_dep_proto * tn_arr_scaling.x[1]
            if max(scaled_tn_arr_proto)>max_value:
                scaled_tn_arr_proto[scaled_tn_arr_proto >max_value]=max_value
                scaled_tn_dep_proto=scaled_tn_dep_proto/max(scaled_tn_dep_proto)*(max_value-tn_arr_scaling.x[0])

            scaled_tn_proto2=scaled_tn_arr_proto-scaled_tn_dep_proto
            scaled_stat_proto = statistic_proto.values * stat_scaling.x[1]+stat_scaling.x[0]
            [tn_running_error_vec[cont,i],proto_running_error_vec[cont,i]]= \
			                errors_calc_max_Now(scaled_tn_proto2, scaled_stat_proto, t_days[i], limit_hour, max_value, window_lenth)
            cont=cont+1

    return [tn_running_error_vec,proto_running_error_vec]


def calcRunningPredcitionErrorMedian(t_days,statistic_proto,tn_proto,max_value,starting_hour=7):
    limit_hour_vec = np.arange (starting_hour, 23, 0.5)
    tn_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))
    proto_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))

    for i in range(0,len(t_days)):
        cont=0
        for limit_hour in limit_hour_vec:
            tn_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], tn_proto)
            stat_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], statistic_proto.values)
            scaled_tn_proto = tn_proto * tn_scaling.x[1]+tn_scaling.x[0]
            scaled_stat_proto = statistic_proto.values * stat_scaling.x[1]+stat_scaling.x[0]
            [tn_running_error_vec[cont,i],proto_running_error_vec[cont,i]]=errors_calc_max_median(scaled_tn_proto, scaled_stat_proto, t_days[i], limit_hour, max_value)
            cont=cont+1
    return [tn_running_error_vec,proto_running_error_vec]


def calcRunningPredcitionErrorTHv2(t_days,statistic_proto,tn_arr_proto,tn_dep_proto,
                                 max_value,starting_hour=7):
    limit_hour_vec = np.arange (starting_hour, 23, 0.5)
    tn_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))
    proto_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))

    for i in range(0,len(t_days)):
        cont=0
        for limit_hour in limit_hour_vec:
            tn_arr_scaling = get_scaling_factor_and_constantTH(limit_hour, t_days[i], tn_arr_proto)
            stat_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], statistic_proto.values)

            scaled_tn_arr_proto = tn_arr_proto * tn_arr_scaling.x[1]+tn_arr_scaling.x[0]
            scaled_tn_dep_proto = tn_dep_proto * tn_arr_scaling.x[1]

            if max(scaled_tn_arr_proto)>max_value:
                #cars_could_not_park=max(scaled_tn_arr_proto[scaled_tn_arr_proto >max_value])-max_value
                #print(round(cars_could_not_park), "cars could not park")
                scaled_tn_arr_proto[scaled_tn_arr_proto >max_value]=max_value
                scaled_tn_dep_proto=scaled_tn_dep_proto/max(scaled_tn_dep_proto)*(max_value-tn_arr_scaling.x[0])

            scaled_tn_proto2=scaled_tn_arr_proto-scaled_tn_dep_proto
            scaled_stat_proto = statistic_proto.values * stat_scaling.x[1]+stat_scaling.x[0]
            [tn_running_error_vec[cont,i],proto_running_error_vec[cont,i]]= \
                errors_calc_max(scaled_tn_proto2, scaled_stat_proto, t_days[i], limit_hour, max_value)
            cont=cont+1
    return [tn_running_error_vec,proto_running_error_vec]


def calcRunningPredcitionErrorTHdep(t_days,statistic_proto,tn_arr_proto,tn_dep_proto,
                                 max_value,starting_hour=7):
    limit_hour_vec = np.arange (starting_hour, 23, 0.5)
    tn_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))
    proto_running_error_vec=np.zeros((len(limit_hour_vec),len(t_days)))

    #find when we should rescale deprting curve
    #we do if more than a proportion of dep_th cars have left (to have enoungh data to fit)
    dep_th=0.05
    bol_index_dep=tn_dep_proto/max(tn_dep_proto)>dep_th
    index_dep=bol_index_dep.argmax()

    for i in range(0,len(t_days)):
        cont=0
        for limit_hour in limit_hour_vec:
            tn_arr_scaling = get_scaling_factor_and_constantTH(limit_hour, t_days[i], tn_arr_proto)
            stat_scaling = get_scaling_factor_and_constant(limit_hour, t_days[i], statistic_proto.values)

            scaled_tn_arr_proto = tn_arr_proto * tn_arr_scaling.x[1]+tn_arr_scaling.x[0]

            bol_rescale_dep= (limit_hour*2 > index_dep)
            if bol_rescale_dep:
                tn_dep_scaling = get_scaling_factor_dep(limit_hour, t_days[i], tn_dep_proto,max_value)
                scaled_tn_dep_proto = tn_dep_proto * tn_dep_scaling.x[0]
            else:
                scaled_tn_dep_proto = tn_dep_proto * tn_arr_scaling.x[1]

            if max(scaled_tn_arr_proto)>max_value:
                #cars_could_not_park=max(scaled_tn_arr_proto[scaled_tn_arr_proto >max_value])-max_value
                #print(round(cars_could_not_park), "cars could not park")
                scaled_tn_arr_proto[scaled_tn_arr_proto >max_value]=max_value
                if not(bol_rescale_dep):
                    scaled_tn_dep_proto=scaled_tn_dep_proto/max(scaled_tn_dep_proto)*(max_value-tn_arr_scaling.x[0])

            scaled_tn_proto2=scaled_tn_arr_proto-scaled_tn_dep_proto
            scaled_stat_proto = statistic_proto.values * stat_scaling.x[1]+stat_scaling.x[0]
            [tn_running_error_vec[cont,i],proto_running_error_vec[cont,i]]= \
                errors_calc_max(scaled_tn_proto2, scaled_stat_proto, t_days[i], limit_hour, max_value)
            cont=cont+1
    return [tn_running_error_vec,proto_running_error_vec]

def model_fit(params,data_curve,model_curve):
    const = params[0]
    scale_factor = params[1]

    errorV=data_curve-model_curve*scale_factor-const
    error = np.sum(np.power(errorV, 2))
    return error


def model_fitTH(params,data_curve,model_curve):
    const = params[0]
    scale_factor = params[1]

    max_index =data_curve.argmax()

    errorV=data_curve[:max_index+1]-model_curve[:max_index+1]*scale_factor-const
    error = np.sum(np.power(errorV, 2))
    return error


def model_fit_dep(params,data_curve,model_curve, max_value):
    scale_factor = params
    epsilon=1 #might be useful to find a heuristic for that
    bol_start_index =model_curve>epsilon
    start_index=bol_start_index.argmax()

    errorV=max_value-data_curve[start_index:]-model_curve[start_index:]*scale_factor
    error = np.sum(np.power(errorV, 2))
    return error



def get_scaling_factor_and_constant(limit_hour, test_day, proto):
    #if limit_hour < 6:
    #    return 1
    index = int(limit_hour*2)
    current_real_data = test_day.values[:index]
    proto_data = proto[:index]
    parameters_fit=[0,1]
    optimal_params_fit = minimize(model_fit,
                                    parameters_fit,
                                    args=(current_real_data, proto_data),
                                    method='Nelder-Mead',
                                    tol=1e-6, options={'disp': False, 'maxfev': 100000})
    return optimal_params_fit


def get_scaling_factor_and_constantTH(limit_hour, test_day, proto):
    #if limit_hour < 6:
    #    return 1
    index = int(limit_hour*2)
    current_real_data = test_day.values[:index]
    proto_data = proto[:index]
    parameters_fit=[0,1]
    optimal_params_fit = minimize(model_fitTH,
                                    parameters_fit,
                                    args=(current_real_data, proto_data),
                                    method='Nelder-Mead',
                                    tol=1e-6, options={'disp': False, 'maxfev': 100000})
    return optimal_params_fit


def get_scaling_factor_dep(limit_hour, test_day, proto, max_value):
    #if limit_hour < 6:
    #    return 1
    index = int(limit_hour*2)
    current_real_data = test_day.values[:index]
    proto_data = proto[:index]
    parameters_fit=1
    optimal_params_fit = minimize(model_fit_dep,
                                    parameters_fit,
                                    args=(current_real_data, proto_data, max_value),
                                    method='Nelder-Mead',
                                    tol=1e-6, options={'disp': False, 'maxfev': 100000})
    return optimal_params_fit

def errors_calc_max(tn_proto, Prototype, real_day, limit_hour, m_value):
    #Computing Errors
    limit_hour = int(limit_hour*2)
    tn_scaled_error = (np.absolute((np.array(tn_proto) - np.array(real_day.values)))/m_value)*100
    mean_scaled_error = (np.absolute((np.array(Prototype) - np.array(real_day.values)))/m_value)*100

    tn_s_error_mean = np.mean(tn_scaled_error[limit_hour:])
    mean_s_error_mean = np.mean(mean_scaled_error[limit_hour:])
    return [tn_s_error_mean,mean_s_error_mean]

def errors_calc_max_Now(tn_proto, Prototype, real_day, limit_hour, m_value, window_lenth):
    #Computing Errors
    limit_hour = int(limit_hour*2)
    tn_scaled_error = (np.absolute((np.array(tn_proto) - np.array(real_day.values)))/m_value)*100
    mean_scaled_error = (np.absolute((np.array(Prototype) - np.array(real_day.values)))/m_value)*100

    tn_s_error_mean = np.mean(tn_scaled_error[limit_hour:(limit_hour+int(window_lenth*2)+1)])
    #print(tn_s_error_mean)
    #print(np.mean(tn_scaled_error[limit_hour:]))
    #print('-----')

    mean_s_error_mean = np.mean(mean_scaled_error[limit_hour:(limit_hour+int(window_lenth*2)+1)])
    return [tn_s_error_mean,mean_s_error_mean]

def errors_calc_max_median(tn_proto, Prototype, real_day, limit_hour, m_value):
    #Computing Errors
    limit_hour = int(limit_hour*2)
    tn_scaled_error = (np.absolute((np.array(tn_proto) - np.array(real_day.values)))/m_value)*100
    mean_scaled_error = (np.absolute((np.array(Prototype) - np.array(real_day.values)))/m_value)*100

    tn_s_error_median = np.median(tn_scaled_error[limit_hour:])
    mean_s_error_medina = np.median(mean_scaled_error[limit_hour:])
    return [tn_s_error_median,mean_s_error_medina]

def errors_calc(tn_proto, Prototype, real_day, limit_hour, m_value):
    #Computing Errors
    limit_hour = int(limit_hour*2)

    normalizador=np.array(real_day.values)
    normalizador[normalizador==0]=1;
    tn_scaled_error = (np.absolute((np.array(tn_proto) - np.array(real_day.values)))/normalizador)*100
    mean_scaled_error = (np.absolute((np.array(Prototype) - np.array(real_day.values)))/normalizador)*100

    tn_s_error_mean = np.mean(tn_scaled_error[limit_hour:])
    mean_s_error_mean = np.mean(mean_scaled_error[limit_hour:])
    return [tn_s_error_mean,mean_s_error_mean]

def plotRunningPredcitionError(tn_running_error_vec,proto_running_error_vec,starting_hour,day,current_parking) :
    fsize=20
    limit_hour_vec = np.arange (starting_hour, 23, 0.5)
    plt.figure(figsize=(18,10));
    plt.plot(limit_hour_vec,np.mean(tn_running_error_vec,axis=1),label='TN')
    plt.plot(limit_hour_vec,np.mean(proto_running_error_vec,axis=1),label='Prototype')
    plt.title("Avearge proportional Prediction Error " + day +  ' ('+ current_parking+')', fontsize=fsize)
    plt.ylabel("Proportional Prediction Error %",fontsize=fsize);
    plt.xlabel("Hour of the day",fontsize=fsize);
    plt.yticks(fontsize=fsize)
    plt.xticks(fontsize=fsize);
    plt.grid(linestyle='dotted', linewidth='0.5', color='grey')
    plt.legend(fontsize=fsize, loc="best",ncol=2);

def plotRunningPredcitionErrorSTDV(tn_running_error_vec,proto_running_error_vec,starting_hour,day,current_parking) :
    fsize=20
    limit_hour_vec = np.arange (starting_hour, 23, 0.5)
    fig=plt.figure(figsize=(18,10));
    plt.plot(limit_hour_vec,np.mean(tn_running_error_vec,axis=1),color='b',label='TN')
    plt.plot(limit_hour_vec,np.mean(tn_running_error_vec,axis=1)+np.std(tn_running_error_vec,axis=1),
             linestyle='dashed',color='b',label='TN±stdv')
    plt.plot(limit_hour_vec,np.mean(tn_running_error_vec,axis=1)-np.std(tn_running_error_vec,axis=1),
             linestyle='dashed',color='b')
    plt.plot(limit_hour_vec,np.mean(proto_running_error_vec,axis=1),color='r',label='Prototype')
    #plt.plot(limit_hour_vec,np.mean(proto_running_error_vec,axis=1)+np.std(proto_running_error_vec,axis=1),
    #         linestyle='dashed',color='r',label='Prototype±stdv')
    #plt.plot(limit_hour_vec,np.mean(proto_running_error_vec,axis=1)-np.std(proto_running_error_vec,axis=1),
    #         linestyle='dashed',color='r')
    plt.title("Avearge proportional Prediction Error " + day +  ' ('+ current_parking+')', fontsize=fsize)
    plt.ylabel("Proportional Prediction Error %",fontsize=fsize);
    plt.xlabel("Hour of the day",fontsize=fsize);
    plt.yticks(fontsize=fsize)
    plt.xticks(fontsize=fsize);
    plt.grid(linestyle='dotted', linewidth='0.5', color='grey')
    plt.legend(fontsize=fsize, loc="best",ncol=2);
    return fig

def plotRunningPredcitionErrorAgg(tn_running_error_wd,proto_running_error_wd,tn_running_error_fr,
                                  proto_running_error_fr,tn_running_error_we,proto_running_error_we,
                                  starting_hour,current_parking,bol_plotstdv=False):
    fsize=20
    #default colors:
    bd='#1f77b4'
    rd='#ff7f0e'
    gd='#2ca02c'
    limit_hour_vec = np.arange (starting_hour, 23, 0.5)
    fig=plt.figure(figsize=(18,10));
    plt.plot(limit_hour_vec,np.mean(tn_running_error_wd,axis=1),color=bd,label='Weekday: TN')
    plt.plot(limit_hour_vec,np.mean(proto_running_error_wd,axis=1),linestyle='dashdot',color=bd,
             label='Weekday: Prototype')
    #plt.plot(limit_hour_vec,np.mean(proto_running_error_wd,axis=1)+np.std(proto_running_error_wd,axis=1),
    #         linestyle='dashed',color='r',label='Prototype±stdv')
    #plt.plot(limit_hour_vec,np.mean(proto_running_error_wd,axis=1)-np.std(proto_running_error_wd,axis=1),
    #         linestyle='dashed',color='r')
    plt.plot(limit_hour_vec,np.mean(tn_running_error_fr,axis=1),color=rd,label='Friday: TN')

    plt.plot(limit_hour_vec,np.mean(proto_running_error_fr,axis=1),linestyle='dashdot',color=rd,
             label='Friday: Prototype')

    plt.plot(limit_hour_vec,np.mean(tn_running_error_we,axis=1),color=gd,label='Weekend: TN')

    if bol_plotstdv:
        plt.plot(limit_hour_vec,np.mean(tn_running_error_wd,axis=1)+np.std(tn_running_error_wd,axis=1),
             linestyle='dashed',color=bd,label='Weekday: TN±stdv')
        plt.plot(limit_hour_vec,np.mean(tn_running_error_wd,axis=1)-np.std(tn_running_error_wd,axis=1),
             linestyle='dashed',color=bd)
        plt.plot(limit_hour_vec,np.mean(tn_running_error_fr,axis=1)+np.std(tn_running_error_fr,axis=1),
             linestyle='dashed',color=rd,label='Friday: TN±stdv')
        plt.plot(limit_hour_vec,np.mean(tn_running_error_fr,axis=1)-np.std(tn_running_error_fr,axis=1),
             linestyle='dashed',color=rd)
        plt.plot(limit_hour_vec,np.mean(tn_running_error_we,axis=1)+np.std(tn_running_error_we,axis=1),
             linestyle='dashed',color=gd,label='Weekend: TN±stdv')
        plt.plot(limit_hour_vec,np.mean(tn_running_error_we,axis=1)-np.std(tn_running_error_we,axis=1),
             linestyle='dashed',color=gd)

    plt.plot(limit_hour_vec,np.mean(proto_running_error_we,axis=1),linestyle='dashdot',color=gd,
             label='Weekend: Prototype')
    plt.title("Avearge proportional Prediction Error ("+ current_parking+')', fontsize=fsize)
    plt.ylabel("Proportional Prediction Error %",fontsize=fsize);
    plt.xlabel("Hour of the day",fontsize=fsize);
    plt.yticks(fontsize=fsize)
    plt.xticks(fontsize=fsize);
    plt.xlim([starting_hour,22.5])
    plt.grid(linestyle='dotted', linewidth='0.5', color='grey')
    plt.legend(fontsize=fsize, loc="best",ncol=3);
    return fig


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    fsize=20
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=fsize)
    ax.tick_params(axis='y', labelsize=fsize)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.grid(linestyle='dotted', linewidth='0.5', color='grey')
    ax.set_xlabel('Type of day and model', fontsize=fsize)
    ax.set_ylabel('Proportional Error (%)', fontsize=fsize)

def calcRunningPredcitionErrorNowReg(t_days,training_days,max_value,starting_hour=7,window_lenth=2,ending_hour=23):
    #starting_hour=7
    #window_lenth=2
    #ending_hour=23
    training_matrix=np.zeros((len(training_days), 48))
    i=0
    for day_list in training_days:
        temp=day_list.tolist()
        training_matrix[i,:]=temp
        i=i+1
    training_diff_matrix=np.diff(training_matrix,axis=1,prepend=0)

    testing_matrix=np.zeros((len(t_days), 48))
    i=0
    for day_list in t_days:
        temp=day_list.tolist()
        testing_matrix[i,:]=temp
        i=i+1
    testing_diff_matrix=np.diff(testing_matrix,axis=1,prepend=0)

    limit_indx_vec = np.arange (starting_hour*2, ending_hour*2, 1)
    reg_running_error_vec=np.zeros((len(limit_indx_vec),len(t_days)))
    cont=0
    for limit_indx in limit_indx_vec:
        X_trainT=training_diff_matrix[:,0:limit_indx]
        y_trainT=training_matrix[:,(limit_indx+window_lenth)]
        model = LinearRegression().fit(X_trainT, y_trainT)

        X_testT=testing_diff_matrix[:,0:limit_indx]
        y_testT=testing_matrix[:,(limit_indx+window_lenth)]
        y_pred = model.predict(X_testT)
        reg_running_error_vec[cont,:] = abs(y_pred-y_testT)/max_value
        cont=cont+1
    return reg_running_error_vec
