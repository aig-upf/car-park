#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import calendar
from scipy import integrate


# In[2]:


available_parkings = ['Vilanova', 'SantSadurni', 'SantBoi', 'QuatreCamins',
                      'Cerdanyola','Granollers','Martorell','Mollet',
                      'SantQuirze','PratDelLlobregat']
df_column_name=['Parking Vilanova Renfe','Parking Sant Sadurní Renfe','Parking Sant Boi de Llobregat',
              'Parking Quatre Camins','Cerdanyola Universitat Renfe','Parking Granollers Renfe',
                'Parking Martorell FGC','Parking Mollet Renfe','Parking Sant Quirze FGC',
               'Parking Prat del Ll.']
current_parking_ix=0
for current_parking_ix in range(0,len(available_parkings)):
#for current_parking_ix in range(0,1):    
# parkings which fill up: 3 QuatreCamins, 7 Mollet, 1 SantSadurni (sometimes),
# problems on Weekend with 2 SantBoi, 4 Cerdanyola, 
# bad data: 6 Martorell, 8 SantQuirze DO NOT USE
#good 0 Vilanova, 1 SantSadurni, 3 QuatreCamins, 5 Granollers, 7 Mollet, 9 PratDelLlobregat
    current_parking = available_parkings[current_parking_ix]
    print(current_parking)
    current_column_name=df_column_name[current_parking_ix]
    
    df = pd.read_csv('data/'+current_parking+"_Estable.csv", delimiter=";")
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.head(6)
    
    
    # # Data processing and preliminary analysis
    
    # In[3]:
    
    
    def getDayName(d):
        return calendar.day_name[d.weekday()]
    
    df = pd.read_csv('data/'+current_parking+"_Estable.csv", delimiter=";")
    df = df.dropna()
    df['Date'] = df['DateTime'].apply(lambda x: x.split(' ')[0])
    df['Time'] = df['DateTime'].apply(lambda x: x.split(' ')[1])
    df['Free slots'] = df[current_column_name +' plazas totales'].apply(lambda x: int(x.split(',')[0]))
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    df.sort_values(by="Date")
    df['Weekday'] = df.apply(lambda x: getDayName(x['Date']),axis=1)
    df['Date'] = [d.date().strftime("%Y-%m-%d") for d in df['Date']]
    df = df.drop(['DateTime'], axis=1)
    df = df.drop([current_column_name +' plazas totales'], axis=1)
    
    max_value = df["Free slots"].max()
    min_value = df["Free slots"].min()
    df['Occupancy'] = df['Free slots'].map(lambda x: max_value-x)
    axis_ylim = max_value+20
    axis_ylim_low = 0
    #print('Y lim: ' ,axis_ylim)
    df.head()
    
    
    # In[4]:
    
    
    # GLOBAL VARIABLES THAT MUST BE FILLED ALONG THE Jup. NOTEBOOK FOR THE EXPORT
    max_capacity = max_value
    
    hist_weekday_proto = []
    hist_friday_proto  = []
    hist_weekend_proto = []
    
    tn_weekday_pars = []
    tn_friday_pars  = []
    tn_weekend_pars = []
    
    tn_weekday_proto = []
    tn_friday_proto  = []
    tn_weekend_proto = []
    
    time = np.linspace(0,23.5,48)
    
    
    # In[5]:
    
    
    x_date=pd.to_datetime(df['Date']+' '+df['Time'])
    y_occ=df['Occupancy']
    
    
    # ### Delete days that have not to be taken into account
    
    # In[6]:
    
    
    #only optimized for Vilanova and QuatreCamins
    
    if current_parking == "QuatreCamins":
        days_list = ['2020-01-01', '2020-01-06', '2020-01-18', '2020-01-19', '2020-01-26', '2020-02-07', 
                     '2020-02-08', '2020-02-09', '2020-03-14', '2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18', 
                     '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25', 
                     '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30']
    elif current_parking == "Cerdanyola":  
        days_list = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-11',
                     '2020-01-12', '2020-01-18', '2020-01-19', '2020-01-25', '2020-01-26', 
                     '2020-02-09', '2020-02-24', '2020-02-29', '2020-03-01', '2020-03-09', '2020-03-14', '2020-03-15', 
                     '2020-03-16', 
                     '2020-03-17', '2020-03-18', '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', 
                     '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', 
                     '2020-03-31']
    elif current_parking == "Granollers":  
        days_list = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06',  
                     '2020-01-25', '2020-01-26', '2020-02-02', '2020-02-08', '2020-02-09', '2020-02-22', '2020-02-23', 
                     '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-14', '2020-03-15', '2020-03-16', 
                     '2020-03-17', '2020-03-18', '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', 
                     '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', 
                     '2020-03-31']
    elif current_parking == "Mollet": 
        days_list = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-11', 
                     '2020-01-12', '2020-02-07', '2020-02-08', '2020-02-09', '2020-03-16', '2020-03-17', '2020-03-18', 
                     '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25', 
                     '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31']     
    elif current_parking == "PratDelLlobregat": 
        days_list = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-17', 
                     '2020-01-18', '2020-01-19', '2020-01-20', '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24',
                     '2020-01-25', '2020-01-26', '2020-02-07', '2020-02-08', '2020-02-09', '2020-02-21', '2020-02-22', 
                     '2020-02-23', '2020-03-16', '2020-03-17', '2020-03-18', 
                     '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25', 
                     '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31']     
    elif (current_parking == "Vilanova") | (current_parking == "SantSadurni") :  
        days_list = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-06', '2020-02-07', '2020-02-08', '2020-02-09', 
                     '2020-03-16', '2020-03-17', '2020-03-18', 
                     '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25', 
                     '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31']
    elif (current_parking == "SantBoi") :  
        days_list = ['2020-01-01', '2020-03-14', '2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18', '2020-03-19', 
                     '2020-01-20', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25', 
                     '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31']
    else:
        days_list = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-06', '2020-02-07', '2020-03-16', '2020-03-17', 
                     '2020-03-18', '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', 
                     '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31']
    
    #days_list = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-06', '2020-02-06', '2020-02-07', '2020-02-23', '2020-02-28', 
    #             '2020-03-16', '2020-03-17', '2020-03-18', '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23',
    #             '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-02-29',
    #             '2020-01-19']
    
    def checkDataValidty(date):
        if date in days_list: 
            return False
        else:
            return True
    
    
    
    
    # In[8]:
    
    
    # In[9]:
    
    
    df['Stable Data'] = df['Date'].apply(lambda x: checkDataValidty(x))
    df_holidays = df[df['Stable Data'] == False]
    df = df[df["Stable Data"] == True]
    
    
    # ## Mean free slots by weekday
    
    # In[10]:
    
    
    cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    df_days = df.groupby([df['Weekday']], as_index=False).mean()
    
    # REORDER BY DAY
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']
    mapping = {day: i for i, day in enumerate(days)}
    key = df_days['Weekday'].map(mapping)
    df_days = df_days.iloc[key.argsort()]
    df_days.reset_index(inplace=True, drop=True) 
    
    #df_bla = df.groupby('Weekday').agg({'score': ['mean', 'std']})
    
    
    
    
    cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_days_ = df.groupby('Weekday').agg({'Occupancy': ['mean', 'std']})
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']
    mapping = {day: i for i, day in enumerate(days)}
    df_days_.reset_index(inplace=True, drop=False) 
    key = df_days_['Weekday'].map(mapping)
    df_days_ = df_days_.iloc[key.argsort()]
    df_days_.reset_index(inplace=True, drop=True) 
    means = df_days_['Occupancy']['mean']
    stds = df_days_['Occupancy']['std']
    
    
    df = df.reset_index(drop = True)
    
    
    # In[13]:
    
    
    
    # In[14]:
    
    
    df_days['Occupancy']
    
    
    # ## Compute the mean for different days
    
    # In[15]:
    
    
    def toAbsHour(hour):
        beginning = [int(s) for s in hour.split(':')]
        absol_hour = (beginning[0] + beginning[1]/60 )
        return absol_hour
    
    df_byhour = df
    df_byhour['ABS_Hour'] = df_byhour['Time'].apply(lambda x: toAbsHour(x) )
    
    df_hours = df_byhour
    df_hours['hour'] = df_hours['ABS_Hour'].map(lambda x: (int(2*x))/2)
    
    
    # # Normalization
    
    # In[29]:
    
    
    def Area_by_date(date):
        df_day = df[(df['Date'] == date)]
        Profile = df_day['Occupancy'].values
        Area = integrate.simps(Profile)
        return Area
    
    df_hours['Area'] = df_hours['Date'].apply(lambda x: Area_by_date(x))
    
    
    def Max_by_date(date):
        df_day = df[(df['Date'] == date)]
        Profile = df_day['Occupancy'].values
        MaxV = max(Profile)
        return MaxV
    
    df_hours['MaxV'] = df_hours['Date'].apply(lambda x: Max_by_date(x))
    
    def df_normalization(occ, area):
        if area == 0:
            return 'error'
        return occ/area
    
    df_hours['Normalized_occupancy'] = df_hours.apply(lambda x: df_normalization(x.Occupancy, x.MaxV), axis=1)
    
    #df_hours = df_hours.drop(['Occupancy'], axis=1)
    #df_hours['Occupancy'] = df_hours['Normalized_occupancy'].apply(lambda x: x)
    df_mean_slots = df_hours.groupby(by=['Weekday','hour'], axis = 0, group_keys=True).mean()
    
    
    # In[30]:
    
    
    df_hours
    
    
    # In[26]:
    
    
    #pd.set_option('display.max_rows',10)
    
    
    # In[31]:
    
    
    mean_occupancy = df_mean_slots['Occupancy']
    
    
    # In[32]:
    
    
    Monday_occ = mean_occupancy['Monday']
    Tuesday_occ = mean_occupancy['Tuesday']
    Wednesday_occ = mean_occupancy['Wednesday']
    Thursday_occ = mean_occupancy['Thursday']
    Friday_occ = mean_occupancy['Friday']
    Saturday_occ = mean_occupancy['Saturday']
    Sunday_occ = mean_occupancy['Sunday']
    
    
    
    
    # ## Variance computation
    
    # In[34]:
    
    
    def mean_day(Weekday,date):
        df = df_hours[(df_hours['Weekday'] == Weekday)]
        df_day = df[(df['Date'] == date)]
        df_day_mean = df_day.groupby(by=['Weekday','hour'], axis = 0, group_keys=True).mean()
        mean_free_slots = df_day_mean['Occupancy']
        Day = mean_free_slots[Weekday]
        return Day
    
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
    
    
    # ### Visualize days
    
    # In[36]:
    
    
    Weekday = 'Monday'
    Day_df = df_hours[(df_hours['Weekday'] == Weekday)]
    dates = Day_df['Date']
    dates_options = set(dates[:][:])
    dates_options = list(dates_options)
    lockdown_dates=[];
    #lockdown_dates.append('06/01/2020')
    # lockdown_dates.append('16/01/2020')
    # lockdown_dates.append('30/01/2020')
    # lockdown_dates.append('23/01/2020')
    
    
    
    # In[37]:
    
    
    Weekday = 'Friday'
    Day_df = df_hours[(df_hours['Weekday'] == Weekday)]
    dates = Day_df['Date']
    dates_options = set(dates[:][:])
    dates_options = list(dates_options)
    lockdown_dates=[];
    #lockdown_dates.append('06/01/2020')
    # lockdown_dates.append('16/01/2020')
    # lockdown_dates.append('30/01/2020')
    
    
    
    # # PREDICTION 
    
    # ### Prepare the data for the prediction
    
    # In[38]:
    
    
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
    
    
    # In[39]:
    
    
    df_byhour['MonthNumber']= df_byhour['Date'].apply(lambda x: x.split('-')[1])
    df_prediction_mean = df_byhour.groupby(by=['Date','hour','MonthNumber','Weekday'], axis = 0, as_index=False).mean()
    df_prediction_mean['Profile_2'] = df_prediction_mean['Weekday'].apply(lambda x: classify_2_proto(x))
    df_prediction_mean['Profile_3'] = df_prediction_mean['Weekday'].apply(lambda x: classify_3_proto(x))
    
    
    # ### Split data in Training/Testing df
    
    # In[40]:
    
    
    number_of_testing_weeks = 3
    df_training, df_testing = split_data(df_prediction_mean, number_of_testing_weeks)
    df_train_data = df_training
    df_train_data.head(5)
    df_testing.head()
    
    
    # ## PREDICTION BY MEAN
    
    # In[41]:
    
    
    def mean_day_profile(Profile, df_aux):
        df = df_aux[(df_aux['Profile_2'] == Profile)]
        df_day_mean = df.groupby(by=['Profile_2','hour'], axis = 0, group_keys=True).mean()
        mean_free_slots = df_day_mean['Occupancy']
        Day = mean_free_slots[Profile]
        return Day
    
    def mean_of_day(Weekday,date):
        df = df_hours[(df_hours['Weekday'] == Weekday)]
        df_day = df[(df['Date'] == date)]
        df_day_mean = df_day.groupby(by=['Weekday','hour'], axis = 0, group_keys=True).mean()
        mean_free_slots = df_day_mean['Occupancy']
        Day = mean_free_slots[Weekday]
        return Day
    
    # Get all days of the type (Monday, Tuesday...) and return the mean of them
    def get_days(dayname, df_):
        data_temp = df_[df_['Weekday'] == dayname] 
        days = []
        for i in range(0,data_temp.shape[0], 48):
            day = data_temp['Occupancy'][i:i+48]
            if len(day) == 48:
                days.append(day)
            
        return days
    
    # Get unique dates of the type (Monday, Tuesday...) 
    def get_dates(dayname, df_):
        data_temp = df_[df_['Weekday'] == dayname] 
        dates = []
        for i in range(0,data_temp.shape[0], 48):
            day = data_temp['Occupancy'][i:i+48]
            t_date = data_temp.iloc[i]['Date']
            if len(day) == 48:
                dates.append(t_date)
            
        return dates
    
    def get_days_of_protos(proto_name, df_):
        data_temp = df_[df_['Profile_3'] == proto_name] 
        days = []
        for i in range(0,data_temp.shape[0], 48):
            day = data_temp['Occupancy'][i:i+48]
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
    
    
    # In[42]:
    
    
    # ------------------------ NEW TESTING DATA -----------------------------
    # these variables are arrays of days
    testing_mondays    = get_days("Monday", df_testing)
    testing_tuesdays   = get_days("Tuesday", df_testing)
    testing_wednesdays = get_days("Wednesday", df_testing)
    testing_thursdays  = get_days("Thursday", df_testing)
    testing_fridays    = get_days("Friday", df_testing)
    testing_saturdays  = get_days("Saturday", df_testing)
    testing_sundays    = get_days("Sunday", df_testing)
    
    
    testing_mondays_dates    = get_dates("Monday", df_testing)
    testing_tuesdays_dates   = get_dates("Tuesday", df_testing)
    testing_wednesdays_dates = get_dates("Wednesday", df_testing)
    testing_thursdays_dates  = get_dates("Thursday", df_testing)
    testing_fridays_dates    = get_dates("Friday", df_testing)
    testing_saturdays_dates  = get_dates("Saturday", df_testing)
    testing_sundays_dates    = get_dates("Sunday", df_testing)
    
    
    # In[43]:
    
   
    
    # In[44]:
    
    
    def mean_day_profile_3(Profile, df_aux):
        df = df_aux[(df_aux['Profile_3'] == Profile)]
        df_day_mean = df.groupby(by=['Profile_3','hour'], axis = 0, group_keys=True).mean()
        mean_free_slots = df_day_mean['Occupancy']
        profile = mean_free_slots[Profile]
        return profile
    
    
    # In[45]:
    
    
    # Obtain the 3 protoypes. IMPORTANT! We assume all the data in df is valid, robust and correct
    def train_statistical_model(df): 
        df_day_mean = df.groupby(by=['Profile_3','hour'], axis = 0, group_keys=True).mean()
        mean_free_slots = df_day_mean['Occupancy']
        return mean_free_slots['Weekday'], mean_free_slots['Friday'], mean_free_slots['Weekend'] 
    
    weekdays = ['Monday', 'Thursday', 'Wednesday', 'Tuesday']
    weekend= ['Saturday', 'Sunday']
    
    # Simply return the proper prototype
    def predict_full_day_statistical(day): 
        if day in weekdays: 
            return hist_weekday_proto
        elif day in weekend: 
            return hist_weekend_proto
        else:
            return hist_friday_proto
    
    
    
    def get_scale_factor(current_data, proto):
        index = len(current_data)
        proto_val = proto[index-1]
        last_hour_val = current_data[-1]
        scaling = last_hour_val/proto_val
        return scaling 
    
    
    
    # In[46]:
    
    
    hist_weekday_proto, hist_friday_proto, hist_weekend_proto = train_statistical_model(df_train_data)
    prediction = predict_full_day_statistical('Monday')
    
    
    # #### Plotting  prototypes
    
    # In[47]:
    
    
    Weekday_proto = mean_day_profile_3('Weekday',df_train_data)
    Weekend_proto = mean_day_profile_3('Weekend',df_train_data)
    df_fri = df_train_data #.drop(df_train_data[df_train_data['Date'] == '2020-02-07' ].index)
    Friday_proto = mean_day_profile_3('Friday',df_fri)
    
    # fig.tight_layout(pad=3.0)
    
    
    # In[33]:
    
    
    Weekday_proto = mean_day_profile_3('Weekday',df_train_data)
    Weekend_proto = mean_day_profile_3('Weekend',df_train_data)
    df_fri = df_train_data #.drop(df_train_data[df_train_data['Date'] == '2020-02-07' ].index)
    Friday_proto = mean_day_profile_3('Friday',df_fri)
    
    
    
    # In[48]:
    
    
    
    # #### Computing errors for 3 prototypes
    
    # In[51]:
    
    
    def compute_testing_prop_error(testing_days, proto_data):
        errors = np.zeros(48)
        n_test_days = len(testing_days)
        proto = np.array(proto_data)
        
        for i in range(0, n_test_days):
            day = np.array(testing_days[i])
            er = np.array((np.absolute(proto - day)/max_value)*100)
            errors += er
        return errors/n_test_days
    
    error_monday_stat = compute_testing_prop_error(testing_mondays, Weekday_proto.values)
    error_tuesday_stat = compute_testing_prop_error(testing_tuesdays, Weekday_proto.values)
    error_wednesday_stat = compute_testing_prop_error(testing_wednesdays, Weekday_proto.values)
    error_thursday_stat = compute_testing_prop_error(testing_thursdays, Weekday_proto.values)
    error_friday_stat = compute_testing_prop_error(testing_fridays, Friday_proto.values)
    error_saturday_stat = compute_testing_prop_error(testing_saturdays, Weekend_proto.values)
    error_sunday_stat = compute_testing_prop_error(testing_sundays, Weekend_proto.values)
    
    
    # In[52]:
    
    
    
    mean_Monday =  [np.mean(error_monday_stat)]*len(time)
    
    
    mean_Tuesday =  [np.mean(error_tuesday_stat)]*len(time)
    
    
    mean_Wednesday =  [np.mean(error_wednesday_stat)]*len(time)
    
    
    mean_Thursday =  [np.mean(error_thursday_stat)]*len(time)
    
    
    mean_Friday =  [np.mean(error_friday_stat)]*len(time)
    
    
    mean_Saturday =  [np.mean(error_saturday_stat)]*len(time)
    
    
    mean_Sunday =  [np.mean(error_sunday_stat)]*len(time)
    
    
    
   # print('______MEAN________')
   # print(mean_Monday[0])
   # print(mean_Tuesday[0])
   # print(mean_Wednesday[0])
   # print(mean_Thursday[0])
   # print(mean_Friday[0])
   # print(mean_Saturday[0])
   # print(mean_Sunday[0])
    
   # print('______STDV________')
    
   # print(np.std(error_monday_stat))
   # print(np.std(error_tuesday_stat))
   # print(np.std(error_wednesday_stat))
   # print(np.std(error_thursday_stat))
   # print(np.std(error_friday_stat))
   # print(np.std(error_saturday_stat))
   # print(np.std(error_sunday_stat))
    
    
    
    
    
    # ## NORMALIZATION
    
    # In[64]:
    
    
    def Area_by_date(date):
        df = df_prediction_mean
        df_day = df[(df['Date'] == date)]
        day_occ = df_day['Occupancy_mod'].values
        Area = integrate.simps(day_occ)
        return Area
    
    def Max_by_date(date):
        df = df_prediction_mean
        df_day = df[(df['Date'] == date)]
        day_occ = df_day['Occupancy_mod'].values
        MaxV = max(day_occ)
        return MaxV
    
    
    def df_normalization(occ, area):
        if area == 0:
            return occ
        return occ/area
    
    def compute_min(date):
        df_day = df[(df['Date'] == date)]
        day_occ = df_day['Occupancy'].values
        return min(day_occ)
    
    
    df_prediction_mean['Min_value'] = df_prediction_mean['Date'].apply(lambda x: compute_min(x))
    df_mean_offset = df_prediction_mean.groupby(['Profile_3'], as_index=False).mean() 
    df_mean_offset[['Profile_3','Min_value']]
    
    friday_offset = df_mean_offset.iloc[0]['Min_value']
    weekday_offset = df_mean_offset.iloc[1]['Min_value']
    weekend_offset = df_mean_offset.iloc[2]['Min_value']
    
    df_prediction_mean['Occupancy_mod'] = df_prediction_mean['Occupancy'] - df_prediction_mean['Min_value']
    
    
    # In[62]:
    
    
    df_prediction_mean
    
    
    # In[66]:
    
    
    df_prediction_mean['Area'] = df_prediction_mean['Date'].apply(lambda x: Area_by_date(x))
    df_prediction_mean['MaxV'] = df_prediction_mean['Date'].apply(lambda x: Max_by_date(x))
    df_prediction_mean['Normalized_occupancy'] = df_prediction_mean.apply(lambda x: df_normalization(x.Occupancy_mod, x.MaxV), axis=1)
    df_prediction_mean['Area_Normalized_occupancy'] = df_prediction_mean.apply(lambda x: df_normalization(x.Occupancy_mod, x.Area), axis=1)
    df_normalized = df_prediction_mean[['Date', 'hour','MonthNumber', 
                                        'Normalized_occupancy','Area_Normalized_occupancy', 'Weekday', 'Profile_3', 
                                        'Occupancy_mod', 'Area', 'MaxV', 'Occupancy','Free slots']].copy()
    
    
    
    df_normalized.to_pickle('data/'+current_parking+'_normalized.pkl')
    
    
    df_normalized = pd.read_pickle('data/'+current_parking+'_normalized.pkl')
    
    
    import pickle

    # obj0, obj1, obj2 are created here...
    
    # Saving the objects:
    with open('data/'+current_parking+'_normalized.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([df_normalized, weekday_offset, friday_offset, weekend_offset, max_value], f)
        f.close()
    
    # Getting back the objects:
    with open('data/'+current_parking+'_normalized.pkl','rb') as f:  # Python 3: open(..., 'rb')
        df_normalized, weekday_offset, friday_offset,  weekend_offset, max_value= pickle.load(f)
        f.close()
    
    # Saving the objects:
    with open('data/'+current_parking+'_testing.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([testing_mondays, testing_mondays_dates, testing_tuesdays, testing_tuesdays_dates, 
                     testing_wednesdays, testing_wednesdays_dates, testing_thursdays, testing_thursdays_dates, 
                     testing_fridays, testing_fridays_dates, testing_saturdays, testing_saturdays_dates,
                     testing_sundays, testing_sundays_dates], f)
        f.close(), 
    
    # Getting back the objects:
    with open('data/'+current_parking+'_testing.pkl','rb') as f:  # Python 3: open(..., 'rb')
        [testing_mondays, testing_mondays_dates, testing_tuesdays, testing_tuesdays_dates, 
        testing_wednesdays, testing_wednesdays_dates, testing_thursdays, testing_thursdays_dates, 
        testing_fridays, testing_fridays_dates, testing_saturdays, testing_saturdays_dates,
        testing_sundays, testing_sundays_dates]= pickle.load(f)
        f.close()  
  
 
    # Saving the objects:
    with open('data/'+current_parking+'_proto.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([hist_weekday_proto, hist_friday_proto, hist_weekend_proto], f)
        f.close(), 
    

    # Getting back the objects:
    with open('data/'+current_parking+'_proto.pkl','rb') as f:  # Python 3: open(..., 'rb')
        [hist_weekday_proto, hist_friday_proto, hist_weekend_proto]= pickle.load(f)
        f.close() 
