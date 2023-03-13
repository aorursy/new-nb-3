import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

from collections import Counter



import warnings

warnings.filterwarnings("ignore")



from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from statsmodels.tsa.arima_model import ARIMA, ARMAResults 



from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline
datapath   = '../input/covid19-global-forecasting-week-2/'

train      = pd.read_csv(datapath+'train.csv',)

test       = pd.read_csv(datapath+'test.csv')
print("Train dataset: ", train.head())

print("Train period: ", train.Date.min(), train.Date.max())

print("Test dataset: ", test.head())

print("Test period: ", test.Date.min(), test.Date.max())
# check metadata of train

train.info()
# check metadata of test

test.info()
train['Date'] = train['Date'].astype('datetime64[ns]')

test['Date'] = test['Date'].astype('datetime64[ns]')



print("Train Date type: ", train['Date'].dtype)

print("Test Date type: ",test['Date'].dtype)
train.columns = ['id','state','country','date','ConfirmedCases','Fatalities']

test.columns  = ['ForecastId', 'state','country','date']
train['place'] = train['state'].fillna('') + '_' + train['country']

test['place'] = test['state'].fillna('') + '_' + test['country']
print('How many places?: ', 'Train: ', len(train['place'].unique()), 

      'Test: ', len(test['place'].unique()))

print('Unique place similar as test?: ',(train['place'].unique() == test['place'].unique()).sum())
fig,ax = plt.subplots(2,1, sharex=True)

ax[0].plot(train.groupby('date')['ConfirmedCases'].sum(),color='blue')

ax[1].plot(train.groupby('date')['Fatalities'].agg(sum),color='red')



ax[0].set_ylabel('Frequency of cases')

ax[1].set_ylabel('Death count')

ax[1].set_xlabel('Date')

plt.xticks(rotation=45)



ax[0].set_title('Total confirmed cases and fatalities (Jan 22 2020-)')

plt.show()
china_cases     = train[train['place'].str.contains('China')][['date',

                                                               'ConfirmedCases',

                                                               'Fatalities']].reset_index(drop=True)

restworld_cases = train[-train['place'].str.contains('China')][['date',

                                                                'ConfirmedCases',

                                                                'Fatalities']].reset_index(drop=True)
#plot total confirmed cases and fatalities in China



fig,ax = plt.subplots(2,1, sharex=True)

ax[0].plot(china_cases.groupby('date')['ConfirmedCases'].sum(), marker='o',color='b', 

            linestyle='--')

ax[1].plot(china_cases.groupby('date')['Fatalities'].sum(), marker='v',color='r',

            linestyle='--')

ax[0].set_ylabel('Frequency of cases')

ax[1].set_ylabel('Death count')

ax[1].set_xlabel('Date')

plt.xticks(rotation=45)



ax[0].set_title('Total confirmed cases and fatalities in China (Jan 22 2020-)')

plt.show()
# plot total confirmed cases and fatalities outside of China



fig,ax = plt.subplots(2,1, sharex=True)

ax[0].plot(restworld_cases.groupby('date')['ConfirmedCases'].sum(), marker='o',color='b', 

            linestyle='--')

ax[1].plot(restworld_cases.groupby('date')['Fatalities'].sum(), marker='v',color='r',

            linestyle='--')

ax[0].set_ylabel('Frequency of cases')

ax[1].set_ylabel('Death count')

ax[1].set_xlabel('Date')

plt.xticks(rotation=45)



ax[0].set_title('Total confirmed cases and fatalities outside of China (Jan 22 2020-)')

plt.show()
top10cases = train.groupby('place')['ConfirmedCases'].sum().sort_values(ascending=False).head(10)



plt.barh(top10cases.index, top10cases)

plt.ylabel('Places')

plt.xlabel('Total confirmed cases')

plt.title('Top 10 places with highest confirmed cases')

plt.show()
# let's look at US states



us_cases     = train[train['place'].str.contains('US')][['date','place',

                                                         'ConfirmedCases',

                                                               'Fatalities']].reset_index(drop=True)
top10uscases = us_cases.groupby('place')['ConfirmedCases'].sum().sort_values(ascending=False).head(10)



plt.barh(top10uscases.index, top10cases)

plt.ylabel('Places')

plt.xlabel('Total confirmed cases')

plt.title('Top 10 US States with highest confirmed cases')

plt.show()
def RMSLE(predicted, actual):

    return np.sqrt(np.mean(np.power((np.log(predicted+1)-np.log(actual+1)),2)))
train_sub = train[['id','place','date','ConfirmedCases','Fatalities']] 

train_sub['logConfirmedCases'] = np.log(train_sub['ConfirmedCases'])

train_sub = train_sub.set_index('date')
list= []

# using rolling window = 3 days



for place in train_sub.place.unique():    

    a = train_sub[train_sub['place']==place]

    a['z_cases'] = (a['logConfirmedCases']- a['logConfirmedCases'].rolling(window=3).mean())/a['logConfirmedCases'].rolling(window=3).std()

    a['zp_cases']= a['z_cases']- a['z_cases'].shift(3)

    a['z_death'] =(a['Fatalities']-a['Fatalities'].rolling(window=3).mean())/a['Fatalities'].rolling(window=3).std()

    a['zp_death']= a['z_death']- a['z_death'].shift(3)

    list.append(a)

    

rolling_df = pd.concat(list)
def plot_rolling(df, variable, z, zp):

    fit, ax= plt.subplots(2, figsize=(10,9), sharex=True)

    ax[0].plot(df.index, df[variable], label='raw data')

    ax[0].plot(df[variable].rolling(window=3).mean(), label="rolling mean");

    ax[0].plot(df[variable].rolling(window=3).std(), label="rolling std (x10)");

    ax[0].legend()

    

    ax[1].plot(df.index, df[z], label="de-trended data")

    ax[1].plot(df[z].rolling(window=3).mean(), label="rolling mean");

    ax[1].plot(df[z].rolling(window=3).std(), label="rolling std (x10)");

    ax[1].legend()

    

    ax[1].set_xlabel('Date')

    plt.xticks(rotation=45)

    ax[0].set_title('{}'.format(place))

    

    plt.show()

    plt.close()
# rolling plots for Confirmed Cases



for place in rolling_df.place.unique()[:5]:

    plot_rolling(df= rolling_df[rolling_df['place']==place], 

                 variable='logConfirmedCases', z= 'z_cases', 

                                 zp= 'zp_cases')
# rolling plots for Fatalities



for place in rolling_df.place.unique()[:5]:

    plot_rolling(df= rolling_df[rolling_df['place']==place], 

                 variable='Fatalities', z= 'z_death', 

                                 zp= 'zp_death')
stationary_data =[]

for place in train_sub.place.unique():

    a= rolling_df[(rolling_df['place']==place) & (rolling_df['logConfirmedCases'] > 0)]['logConfirmedCases'].dropna()

    try:   

        dftest = adfuller(a, autolag='AIC')

        if (dftest[1] < 0.001):

            stationary_data.append(place)

        else: 

            pass

    except:

        pass

    

print(len(stationary_data))
station_death_data =[]

for place in train_sub.place.unique():

    dftest = adfuller(rolling_df[rolling_df['place']==place]['Fatalities'], autolag='AIC')

    if (dftest[1] < 0.001):

        station_death_data.append(place)

    else: 

        pass

    

print(len(station_death_data))
# ACF and PACF plots for Confirmed Cases

for place in stationary_data:

    fig,ax = plt.subplots(2,figsize=(12,6))

    ax[0] = plot_acf(rolling_df[rolling_df['place']==place]['logConfirmedCases'].dropna(), ax=ax[0], lags=2)

    ax[1] = plot_pacf(rolling_df[rolling_df['place']==place]['logConfirmedCases'].dropna(), ax=ax[1], lags=2)

    plt.title('{}'.format(place))
# ACF and PACF plots for Fatalities

for place in stationary_data:

    fig,ax = plt.subplots(2,figsize=(12,6))

    ax[0] = plot_acf(np.log(rolling_df[rolling_df['place']==place]['Fatalities']).dropna(), ax=ax[0], lags=2)

    ax[1] = plot_pacf(np.log(rolling_df[rolling_df['place']==place]['Fatalities']).dropna(), ax=ax[1], lags=2)

    plt.title('{}'.format(place))
# list of places with lags for Confirmed Cases

confirmedc_lag = ['Anhui_China', 'Chongqing_China','Guangdong_China',

                  'Guizhou_China', 'Hainan_China', 'Hebei_China','Hubei_China',

                 'Ningxia_China','Shandong_China','Shanxi_China', 'Sichuan_China']
# list of places with non-stationary confirmed cases data

allplaces = train_sub.place.unique().tolist()

non_stationary_data = [ele for ele in allplaces]



for place in confirmedc_lag:

    if place in allplaces:

        non_stationary_data.remove(place)



print(len(non_stationary_data))
# list of places with lags for Fatality

fatalities_lag = ['Hubei_China']
# list of places with non-stationary fatalities data

non_stationary_death_data = [ele for ele in allplaces]



for place in fatalities_lag:

    if place in allplaces:

        non_stationary_death_data.remove(place)



print(len(non_stationary_death_data))
from numpy import inf

train_sub['logConfirmedCases']= train_sub['logConfirmedCases'].replace(to_replace=-inf,

                                                                      value=0)
poly_data = train[['date','place',

                  'ConfirmedCases','Fatalities']].merge(test[['date','place']], 

                                                      how='outer', 

                                                        on=['date','place']).sort_values(['place',

                                                                                          'date'])



print(poly_data.date.min(), test.date.min(), train.date.max(), poly_data.date.max())
# create label for each date by each place

label = []

for place in poly_data.place.unique():

    labelrange = range(1,len(poly_data[poly_data['place']==place])+1)

    label.append([i for i in labelrange])

lab = [item for lab in label for item in lab]

poly_data['label'] = lab

poly_data.head()
XYtrain = poly_data[(poly_data['date']>'2020-01-21')&((poly_data['date']<'2020-04-01'))]

print(XYtrain.date.min(), XYtrain.date.max(), XYtrain.isna().sum())
XYtest = poly_data[(poly_data['date']>'2020-03-18')&(poly_data['date']<'2020-05-01')]

print(XYtest.date.min(), XYtest.date.max(), XYtest.isna().sum())
XYtrain['intercept']= -1



result=pd.DataFrame()

for place in poly_data.place.unique():

    for degree in [2,3,4,5,6]:

        features  = XYtrain[XYtrain['place']==place][['label','intercept']]

        target    = XYtrain[XYtrain['place']==place]['ConfirmedCases']

        model  = make_pipeline(PolynomialFeatures(degree), Ridge())

        model.fit(np.array(features), target)

        y_pred = model.predict(np.array(features))

        rmsle  = RMSLE(y_pred, target)

        result = result.append(pd.DataFrame({'place':[place],

                                             'degree':[degree],'RMSLE': [rmsle]}))

    

# if you want to look at the plot

        #plt.plot(features, y_pred, 

        #         label= "degree %d" % degree

        #         +';$RMSLE: %.2f' % RMSLE(y_pred, target))

    #plt.legend(loc='upper left')

    #plt.xlabel('date')

    #plt.ylabel('predictedcase')

    #plt.title("Polynomial model for confirmed cases in {}".format(place) )

    #plt.show()
best_degree = pd.DataFrame()

for place in result.place.unique():

    a = result[result['place']==place]

    best_degree = best_degree.append(a[a['RMSLE'] == a['RMSLE'].min()])

print(best_degree.groupby('degree')['place'].nunique())

print('Zero polynomial (no fit): ',best_degree[best_degree['RMSLE']<0.00001]['place'].unique())
fit_best_degree = best_degree[best_degree['RMSLE']>0.00001]

twodeg_places   = fit_best_degree[fit_best_degree['degree']==2]['place'].unique()

threedeg_places = fit_best_degree[fit_best_degree['degree']==3]['place'].unique()

fourdeg_places  = fit_best_degree[fit_best_degree['degree']==4]['place'].unique()

fivedeg_places  = fit_best_degree[fit_best_degree['degree']==5]['place'].unique()

sdeg_places  = fit_best_degree[fit_best_degree['degree']==6]['place'].unique()

nofit_places1    = best_degree[best_degree['RMSLE']<0.00001]['place'].unique()

print(fit_best_degree.nunique())

print(len(twodeg_places), len(threedeg_places), 

      len(fourdeg_places), len(fivedeg_places), len(sdeg_places), len(nofit_places1))
XYtest = XYtest.reset_index(drop=True)

XYtest['intercept'] = -1
poly_predicted_confirmedcases = pd.DataFrame() 

for place in twodeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['ConfirmedCases']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(2), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    a = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred),columns=['place','ConfirmedCases'])

    poly_predicted_confirmedcases = poly_predicted_confirmedcases.append(a)

    

for place in threedeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['ConfirmedCases']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(3), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    b = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','ConfirmedCases'])

    poly_predicted_confirmedcases = poly_predicted_confirmedcases.append(b)

    

    

for place in fourdeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['ConfirmedCases']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(4), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    c = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','ConfirmedCases'])

    poly_predicted_confirmedcases = poly_predicted_confirmedcases.append(c)

    

    

for place in fivedeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['ConfirmedCases']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(5), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    d = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','ConfirmedCases'])

    poly_predicted_confirmedcases = poly_predicted_confirmedcases.append(d)

    

for place in sdeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['ConfirmedCases']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(6), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    e = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','ConfirmedCases'])

    poly_predicted_confirmedcases = poly_predicted_confirmedcases.append(e)



fatalities_result=pd.DataFrame()

for place in poly_data.place.unique():

    for degree in [2,3,4,5,6]:

        features  = XYtrain[XYtrain['place']==place][['label','intercept']]

        target    = XYtrain[XYtrain['place']==place]['Fatalities']

        model  = make_pipeline(PolynomialFeatures(degree), Ridge())

        model.fit(np.array(features), target)

        y_pred = model.predict(np.array(features))

        rmsle  = RMSLE(y_pred, target)

        fatalities_result = fatalities_result.append(pd.DataFrame({'place':[place],

                                             'degree':[degree],'RMSLE': [rmsle]}))
fat_best_degree = pd.DataFrame()

for place in fatalities_result.place.unique():

    a = fatalities_result[fatalities_result['place']==place]

    fat_best_degree = fat_best_degree.append(a[a['RMSLE'] == a['RMSLE'].min()])

print(fat_best_degree.groupby('degree')['place'].nunique())

print('Zero polynomial (no fit): ',

      fat_best_degree[fat_best_degree['RMSLE']<0.000001]['place'].unique())
fit_best_degree = fat_best_degree[fat_best_degree['RMSLE']>0.000001]

twodeg_places   = fit_best_degree[fit_best_degree['degree']==2]['place'].unique()

threedeg_places = fit_best_degree[fit_best_degree['degree']==3]['place'].unique()

fourdeg_places  = fit_best_degree[fit_best_degree['degree']==4]['place'].unique()

fivedeg_places  = fit_best_degree[fit_best_degree['degree']==5]['place'].unique()

sevdeg_places  = fit_best_degree[fit_best_degree['degree']==6]['place'].unique()

nofit_places2    = fat_best_degree[fat_best_degree['RMSLE']<0.000001]['place'].unique()

print(fit_best_degree.nunique())

print(len(twodeg_places), len(threedeg_places), 

      len(fourdeg_places), len(fivedeg_places), len(sevdeg_places), len(nofit_places2))
poly_predicted_fatalities = pd.DataFrame() 

for place in twodeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['Fatalities']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(2), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    a = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','Fatalities'])

    poly_predicted_fatalities = poly_predicted_fatalities.append(a)

    

for place in threedeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['Fatalities']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(3), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    b = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','Fatalities'])

    poly_predicted_fatalities = poly_predicted_fatalities.append(b)

    

    

for place in fourdeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['Fatalities']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(4), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    c = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','Fatalities'])

    poly_predicted_fatalities = poly_predicted_fatalities.append(c)

    

    

for place in fivedeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['Fatalities']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(5), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    d = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','Fatalities'])

    poly_predicted_fatalities = poly_predicted_fatalities.append(d)



for place in sevdeg_places:

    features  = XYtrain[XYtrain['place']==place][['label','intercept']]

    target    = XYtrain[XYtrain['place']==place]['Fatalities']

    Xtest     = XYtest[XYtest['place']==place][['label','intercept']]

    model  = make_pipeline(PolynomialFeatures(6), Ridge())

    model.fit(np.array(features), target)

    y_pred = model.predict(np.array(Xtest))

    e = pd.DataFrame(zip(XYtrain[XYtrain['place']==place]['place'], 

                              y_pred.tolist()),columns=['place','Fatalities'])

    poly_predicted_fatalities = poly_predicted_fatalities.append(e)
# forward fill no fit places for confirmed cases

for place in nofit_places1:

    e = poly_data[(poly_data['place']==place) & (poly_data['date']>'2020-03-18')]

    f = e['ConfirmedCases'].fillna(method = 'ffill')

    g = pd.DataFrame(zip(e['place'], f),columns=['place','ConfirmedCases'])

    poly_predicted_confirmedcases = poly_predicted_confirmedcases.append(g)



# forward fill no fit places for fatalities

for place in nofit_places2:

    h = poly_data[(poly_data['place']==place) & (poly_data['date']>'2020-03-18')]

    i = h['Fatalities'].fillna(method = 'ffill')

    j = pd.DataFrame(zip(h['place'], i),columns=['place','Fatalities'])

    poly_predicted_fatalities = poly_predicted_fatalities.append(j)
print(poly_predicted_confirmedcases.shape, poly_predicted_fatalities.shape)
poly_predicted_confirmedcases2= pd.DataFrame({'date':XYtest.date,

                                              'place':poly_predicted_confirmedcases['place'].tolist(),

                                              'ConfirmedCases':poly_predicted_confirmedcases['ConfirmedCases'].tolist()})

poly_predicted_confirmedcases2.head()
poly_predicted_confirmedcases2.shape
poly_predicted_fatalities2= pd.DataFrame({'date':XYtest.date,

                                              'place':poly_predicted_fatalities['place'].tolist(),

                                              'Fatalities':poly_predicted_fatalities['Fatalities'].tolist()})

poly_predicted_fatalities2.head()
poly_predicted_fatalities2.shape
poly_compiled = poly_predicted_confirmedcases2.merge(poly_predicted_fatalities2, how='inner', on=['place','date'])
test_poly_compiled= test.merge(poly_compiled, how='inner', on=['place','date'])

test_poly_compiled= test_poly_compiled.set_index('date')

test_poly_compiled
df_compiled = pd.DataFrame()

for place in test_poly_compiled.place.unique():

        a = test_poly_compiled[test_poly_compiled['place']==place]

        ind_max_confirmedcases = np.argmax(a['ConfirmedCases'])

        a = a.replace(to_replace=a.loc[(a.index>ind_max_confirmedcases),'ConfirmedCases'].tolist(),

                      value=a.loc[ind_max_confirmedcases,'ConfirmedCases'])

        

        ind_max_fatatities     = np.argmax(a['Fatalities'])

        a = a.replace(to_replace=a.loc[(a.index>ind_max_fatatities),'Fatalities'].tolist(),

                      value=a.loc[ind_max_fatatities,'Fatalities'])

        df_compiled = df_compiled.append(a)



df_compiled[df_compiled['place']=='_Zimbabwe'].tail()
# for place in df_compiled.place.unique():

#     fig, ax = plt.subplots(2,1, sharex=True)

#     ax[0].plot(df_compiled[df_compiled['place']==place].groupby('date')['ConfirmedCases'].sum(), 

#                marker='o',color='b', linestyle='--')

#     ax[1].plot(df_compiled[df_compiled['place']==place].groupby('date')['Fatalities'].sum(), 

#                marker='v',color='r',linestyle='--')

#     ax[0].set_ylabel('Predicted cases')

#     ax[1].set_ylabel('Predicted deaths')

#     ax[1].set_xlabel('Date')

#     plt.xticks(rotation=45)



#     ax[0].set_title('Total predicted cases and fatalities in {}'.format(place))

#     plt.show()
submission= pd.read_csv(datapath+'submission.csv')
sub2 = submission[['ForecastId']].merge(df_compiled[['ForecastId','ConfirmedCases','Fatalities']],

                                      how='left',on='ForecastId') 
sub2['ConfirmedCases'] = sub2['ConfirmedCases'].round(1)

sub2['Fatalities'] = sub2['Fatalities'].round(1).abs()
sub2
sub2.to_csv('submission.csv', index=False)