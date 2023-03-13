# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import random



from tqdm.notebook import tqdm



from functools import partial

from scipy.optimize import minimize

import matplotlib.pyplot as plt
def seed_everything(seed=0):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)



seed_everything(seed=0)
train = False
weights_lambda = 0.90 # reflect the weight decay for distant days
print([weights_lambda ** i for i in range(100)])
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

train_df = train_df.fillna('')

train_df.tail()
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

test_df = test_df.fillna('')

test_df.tail()
num_dates_total = len(np.unique(list(train_df['Date']) + list(test_df['Date'])))

print(num_dates_total)

num_dates_test = len(np.unique(list(test_df['Date'])))

print(num_dates_test)

num_dates_train = len(np.unique(list(train_df['Date'])))

print(num_dates_train)
sample_submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')

sample_submission.head()
len(sample_submission) / 3
cases = train_df["TargetValue"][train_df["Target"] == 'ConfirmedCases'].values.reshape((-1, num_dates_train))

cases
fatalities = train_df["TargetValue"][train_df["Target"] == 'Fatalities'].values.reshape((-1, num_dates_train))

fatalities
#making the weights equal

fatalities = fatalities * 10
population = train_df["Population"].values[::num_dates_train * 2]

population

len(population)
weignts_cases = train_df["Weight"][train_df["Target"] == 'ConfirmedCases'].values[::num_dates_train]

weignts_cases

len(weignts_cases)
import lightgbm as lgbm

params = {

    "metric":"mse",

}

f = lgbm.LGBMRegressor(**params)
if train:

    days_to_predict = 45

else:

    days_to_predict = 31

train_days_used = num_dates_total - days_to_predict

assert train_days_used <= num_dates_train





features_days = 40

test_days_predict = 40

train_data = []

target_data = []

test_data = []

weights = []





fat_index_add = features_days

fat_index_add_test = test_days_predict 
for i in range(len(population)):

    for j in range(train_days_used - 1 - features_days):

        full_train = True

        

        data = []

        data = data + [population[i]]

        max_cases = cases[i][:j+features_days].max()

        data = data + [max_cases]

        max_fatalities = fatalities[i][:j+features_days].max()

        data = data + [max_fatalities]

        data = data + list(cases[i][j:j+features_days]/(max_cases+1))

        data = data + list(fatalities[i][j:j+features_days]/(max_fatalities+1))

        train_data.append(data)



        target = []

        target_list = list(cases[i][j+features_days:j+features_days + test_days_predict]/(max_cases+1))

        target_list = target_list + [None]*(test_days_predict - len(target_list))

        target = target + target_list



        target_list = list(fatalities[i][j+features_days:j+features_days + test_days_predict]/(max_fatalities+1))

        target_list = target_list + [None]*(test_days_predict - len(target_list))

        target = target + target_list           



        target_data.append(target)

        weights.append(weignts_cases[i]*weights_lambda**(train_days_used - test_days_predict - features_days - j - 1))

    

    test = []

    test = test + [population[i]]

    max_cases = cases[i][:train_days_used].max()

    test = test + [max_cases]

    max_fatalities = fatalities[i][:train_days_used].max()

    test = test + [max_fatalities]

    test = test + list(cases[i][train_days_used - features_days:train_days_used]/(max_cases+1))

    test = test + list(fatalities[i][train_days_used - features_days:train_days_used]/(max_fatalities+1))

    test_data.append(test)        



initial_train_data = np.stack(train_data)

target_data = np.stack(target_data)

max_features_days = features_days

max_test_days_predict = test_days_predict

initial_test_data = np.stack(test_data)

weights = np.array(weights)    
i = 0

test = []

test = test + [population[i]]

max_cases = cases[i][:train_days_used].max()

test = test + [max_cases]

max_fatalities = fatalities[i][:train_days_used].max()

test = test + [max_fatalities]

test = test + list(cases[i][train_days_used - features_days:train_days_used]/(max_cases+1))

test = test + list(fatalities[i][train_days_used - features_days:train_days_used]/(max_fatalities+1))

test_data.append(test)   
predictions_cases_global = []

predictions_cases_max_global = np.zeros((len(population), days_to_predict))

predictions_cases_min_global = np.zeros((len(population), days_to_predict)) + 1000000



predictions_fatalities_global = []

predictions_fatalities_max_global = np.zeros((len(population), days_to_predict))

predictions_fatalities_min_global = np.zeros((len(population), days_to_predict)) + 1000000
for features_days in tqdm(range(5, 40, 10)):

    fat_index_add = features_days

    train_data = np.concatenate([initial_train_data[:,:3], 

                                 initial_train_data[:,3+max_features_days-features_days:3+max_features_days],

                                 initial_train_data[:,3+2*max_features_days-features_days:3+2*max_features_days],

                                ], axis = 1).copy()

    

    case_predictors = []

    for i in range(max_test_days_predict):

        f = lgbm.LGBMRegressor(**params)

        mask = np.logical_not(pd.isnull(target_data[:,i]))

        f.fit(train_data[mask], target_data[:,i][mask],sample_weight = weights[mask], verbose=False)

        case_predictors.append(f)



    fatalities_predictors = []

    for i in range(max_test_days_predict):

        f = lgbm.LGBMRegressor(**params)

        mask = np.logical_not(pd.isnull(target_data[:,max_test_days_predict + i]))

        f.fit(train_data[mask], target_data[:,max_test_days_predict + i][mask],sample_weight = weights[mask], verbose=False)

        fatalities_predictors.append(f)

        

    

    for test_days_predict in range(1,32, 5):

        

        test_data = np.concatenate([initial_test_data[:,:3], 

                             initial_test_data[:,3+max_features_days-features_days:3+max_features_days],

                             initial_test_data[:,3+2*max_features_days-features_days:3+2*max_features_days],

                            ], axis = 1).copy()

        

        predictions_cases_sum = np.zeros((len(population), days_to_predict))

        predictions_cases_max = np.zeros((len(population), days_to_predict))

        predictions_cases_min= np.zeros((len(population), days_to_predict))+ 1000000

        predictions_cases_counts = np.zeros(days_to_predict)





        predictions_fatalities_sum = np.zeros((len(population), days_to_predict))

        predictions_fatalities_max = np.zeros((len(population), days_to_predict))

        predictions_fatalities_min= np.zeros((len(population), days_to_predict))+ 1000000

        predictions_fatalities_counts = np.zeros(days_to_predict)



        for step in range(days_to_predict - test_days_predict + 1):

            predictions_cases_local = np.zeros((len(population), test_days_predict))

            predictions_fatalities_local = np.zeros((len(population), test_days_predict))



            for i in range(test_days_predict):

                predictions_cases_local[:,i] = case_predictors[i].predict(test_data) * (test_data[:,1] + 1)



            for i in range(test_days_predict):

                predictions_fatalities_local[:,i] = fatalities_predictors[i].predict(test_data) * (test_data[:,2] + 1)





            predictions_cases_sum[:,step:step+test_days_predict] += predictions_cases_local

            predictions_cases_max[:,step:step+test_days_predict] = np.maximum(predictions_cases_max[:,step:step+test_days_predict], predictions_cases_local)

            predictions_cases_min[:,step:step+test_days_predict] = np.minimum(predictions_cases_min[:,step:step+test_days_predict], predictions_cases_local)

            predictions_cases_counts[step:step+test_days_predict] += 1



            current_predictions_cases = predictions_cases_sum[:,step] / predictions_cases_counts[step]





            predictions_fatalities_sum[:,step:step+test_days_predict] += predictions_fatalities_local

            predictions_fatalities_max[:,step:step+test_days_predict] = np.maximum(predictions_fatalities_max[:,step:step+test_days_predict], predictions_fatalities_local)

            predictions_fatalities_min[:,step:step+test_days_predict] = np.minimum(predictions_fatalities_min[:,step:step+test_days_predict], predictions_fatalities_local)

            predictions_fatalities_counts[step:step+test_days_predict] += 1



            current_predictions_fatalities = predictions_fatalities_sum[:,step] / predictions_fatalities_counts[step]



            test_data[:,3:3+features_days-1] = test_data[:,4:3+features_days]

            new_max = np.maximum(test_data[:,1], current_predictions_cases)

            test_data[:,3:3+features_days -1] *= ((test_data[:,1] + 1) / (new_max + 1)).reshape((-1,1))

            test_data[:,2+features_days] = current_predictions_cases / (new_max+1)

            test_data[:,1] = new_max







            test_data[:,3+features_days:3+features_days-1+features_days] = test_data[:,4+features_days:3+features_days+features_days]

            new_max = np.maximum(test_data[:,2], current_predictions_fatalities)

            test_data[:,3+features_days:3+features_days+features_days-1] *= ((test_data[:,2] + 1) /(new_max + 1) ).reshape((-1,1))

            test_data[:,2+features_days+features_days] = current_predictions_fatalities / (new_max+1)

            test_data[:,2] = new_max





        predictions_cases_global.append(predictions_cases_sum / predictions_cases_counts)

        predictions_cases_min_global = np.minimum(predictions_cases_min_global, predictions_cases_min)

        predictions_cases_max_global = np.maximum(predictions_cases_max_global, predictions_cases_max)



        predictions_fatalities_global.append(predictions_fatalities_sum / predictions_fatalities_counts)

        predictions_fatalities_min_global = np.minimum(predictions_fatalities_min_global, predictions_fatalities_min)

        predictions_fatalities_max_global = np.maximum(predictions_fatalities_max_global, predictions_fatalities_max)

final_predictions_cases = np.stack(predictions_cases_global).mean(0)

final_predictions_fatalities = np.stack(predictions_fatalities_global).mean(0)
final_predictions_cases_std = np.stack(predictions_cases_global).std(0)

final_predictions_fatalities_std = np.stack(predictions_fatalities_global).std(0)
valid_len = num_dates_train + days_to_predict - num_dates_total

valid_len
valid_true_cases = cases[:,-valid_len:]

valid_true_fatalities = fatalities[:, -valid_len:]
predict_mean_cases = final_predictions_cases[:,:valid_len].copy()

predict_min_cases = predictions_cases_min_global[:,:valid_len].copy()

predict_max_cases = predictions_cases_max_global[:,:valid_len].copy()



predict_mean_fatalities = final_predictions_fatalities[:,:valid_len].copy()

predict_min_fatalities = predictions_fatalities_min_global[:,:valid_len].copy()

predict_max_fatalities = predictions_fatalities_max_global[:,:valid_len].copy()
def compute_loss_L(true_array, predicted_array, tau, weight):

    array = predicted_array * (predicted_array > 0)

    abs_diff = np.absolute(true_array - array)

    result = abs_diff * (1 -tau) * (array > true_array) + abs_diff * (tau) * (array <= true_array)

    result = (result.mean(1)) * weight

#     print(result.mean())

    return result.mean()
def compute_loss(true_array, mean_array, min_array, max_array, weights):

    result = (compute_loss_L(true_array, max_array, 0.95, weights) + 

              compute_loss_L(true_array, min_array, 0.05, weights) + 

              compute_loss_L(true_array, mean_array, 0.5, weights))

    return result / 3
x0 = [1,0,0,0]



def normalize(x, mean_array, min_array, max_array, base = None):

    if base is None:

        base = np.zeros_like(mean_array)

        lamb = 0

    else:

        lamb = x[3]

    deviation = np.array([lamb * n for n in range(base.shape[1])])

    new_array = base + (x[0] * mean_array + x[1] * min_array + x[2]*max_array) * (deviation + 1).reshape((1,-1))

#     print(deviation)

    return new_array
def fun(x, mean_array, min_array, max_array, true_array, weights, tau, base = None):

    new_array = normalize(x, mean_array, min_array, max_array, base = base)

    return compute_loss_L(true_array, new_array, tau, weights)
from scipy.optimize import minimize

from functools import partial
x = [1.13952572e+00,  5.49225865e-02, -2.41205637e-01,  0]

if train:

    part_func = partial(fun, mean_array = predict_mean_cases, min_array = predict_min_cases, max_array = predict_max_cases, weights = weignts_cases, tau = 0.5, true_array = valid_true_cases)

    res = minimize(part_func, x0 = [1,0,0,0], method='Powell', tol=1e-6)

    print(res.fun)

    print(res.x)

    x = res.x



new_final_predictions_cases = normalize(x, final_predictions_cases, predictions_cases_min_global, predictions_cases_max_global)





x = [ 0.15951811, -0.16219168, -0.45721537, -0.01818924]

if train:

    part_func = partial(fun, mean_array = predict_mean_cases, min_array = predict_min_cases, max_array = predict_max_cases, weights = weignts_cases, tau = 0.05, true_array = valid_true_cases, 

                        base = new_final_predictions_cases[:,:valid_len])

    res = minimize(part_func, x0 = [1,0,0, 0], method='Powell', tol=1e-6)

    print(res.fun)

    print(res.x)

    x = res.x



new_predictions_cases_min_global = normalize(x, final_predictions_cases, predictions_cases_min_global, predictions_cases_max_global, base = new_final_predictions_cases)





x = [0.69243738, -0.25191857,  0.14696969,  0.00829207]

if train:

    part_func = partial(fun, mean_array = predict_mean_cases, min_array = predict_min_cases, max_array = predict_max_cases, weights = weignts_cases, tau = 0.95, true_array = valid_true_cases,

                       base = new_final_predictions_cases[:,:valid_len])

    res = minimize(part_func, x0 = [1,0,0, 0], method='Powell', tol=1e-6)

    print(res.fun)

    print(res.x)

    x = res.x



new_predictions_cases_max_global = normalize(x, final_predictions_cases, predictions_cases_min_global, predictions_cases_max_global, base = new_final_predictions_cases)



final_predictions_cases = new_final_predictions_cases

predictions_cases_min_global = new_predictions_cases_min_global

predictions_cases_max_global = new_predictions_cases_max_global
x = [ 8.14860611e-01,  8.80132452e-02, -1.60911727e-01,  0]

if train:

    part_func = partial(fun, mean_array = predict_mean_fatalities, min_array = predict_min_fatalities, max_array = predict_max_fatalities, weights = weignts_cases, 

                        tau = 0.5, true_array = valid_true_fatalities)

    res = minimize(part_func, x0 = [1,0,0, 0], method='Powell', tol=1e-6)

    print(res.fun)

    print(res.x)

    x = res.x



new_final_predictions_fatalities = normalize(x, final_predictions_fatalities, predictions_fatalities_min_global, predictions_fatalities_max_global)





x = [-0.92953006,  0.998952,   -0.40161728, -0.04630638]

if train:

    part_func = partial(fun, mean_array = predict_mean_fatalities, min_array = predict_min_fatalities, max_array = predict_max_fatalities, weights = weignts_cases, 

                        tau = 0.05,

                        true_array = valid_true_fatalities, base = new_final_predictions_fatalities[:,:valid_len])

    res = minimize(part_func, x0 = [1,0,0, 0], method='Powell', tol=1e-6)

    print(res.fun)

    print(res.x)

    x = res.x



new_predictions_fatalities_min_global = normalize(x, final_predictions_fatalities, predictions_fatalities_min_global, predictions_fatalities_max_global,

                                                  base = new_final_predictions_fatalities)





x = [0.8965989,   0.02662769, -0.01948303,  0.00444532]

if train:

    part_func = partial(fun, mean_array = predict_mean_fatalities, min_array = predict_min_fatalities, max_array = predict_max_fatalities, weights = weignts_cases, 

                        tau = 0.95, 

                        true_array = valid_true_fatalities, base = new_final_predictions_fatalities[:,:valid_len])

    res = minimize(part_func, x0 = [1,0,0, 0], method='Powell', tol=1e-6)

    print(res.fun)

    print(res.x)

    x = res.x



new_predictions_fatalities_max_global = normalize(x, final_predictions_fatalities, predictions_fatalities_min_global, predictions_fatalities_max_global,

                                                  base = new_final_predictions_fatalities)



final_predictions_fatalities = new_final_predictions_fatalities

predictions_fatalities_min_global = new_predictions_fatalities_min_global

predictions_fatalities_max_global = new_predictions_fatalities_max_global
predict_mean_cases = final_predictions_cases[:,:valid_len].copy()

predict_min_cases = predictions_cases_min_global[:,:valid_len].copy()

predict_max_cases = predictions_cases_max_global[:,:valid_len].copy()



predict_mean_fatalities = final_predictions_fatalities[:,:valid_len].copy()

predict_min_fatalities = predictions_fatalities_min_global[:,:valid_len].copy()

predict_max_fatalities = predictions_fatalities_max_global[:,:valid_len].copy()
if train:

    total_loss = (compute_loss(valid_true_cases, predict_mean_cases, predict_min_cases, 

                                   predict_max_cases , weignts_cases) + 

                     compute_loss(valid_true_fatalities, predict_mean_fatalities, predict_min_fatalities, 

                                  predict_max_fatalities, weignts_cases)) / 2



    print(total_loss)
submission_mean_cases = np.zeros((len(population), 45))

submission_min_cases = np.zeros((len(population), 45))

submission_max_cases = np.zeros((len(population), 45))



submission_mean_fatalities = np.zeros((len(population), 45))

submission_min_fatalities = np.zeros((len(population), 45))

submission_max_fatalities = np.zeros((len(population), 45))
submission_mean_cases[:, -days_to_predict:] = final_predictions_cases

submission_min_cases[:, -days_to_predict:] = predictions_cases_min_global

submission_max_cases[:, -days_to_predict:] = predictions_cases_max_global





submission_mean_fatalities[:, -days_to_predict:] = final_predictions_fatalities / 10

submission_min_fatalities[:, -days_to_predict:] = predictions_fatalities_min_global / 10

submission_max_fatalities[:, -days_to_predict:] = predictions_fatalities_max_global / 10
submission_mean_cases[:, :-days_to_predict] = cases[:, num_dates_total-45:num_dates_total-days_to_predict]

submission_min_cases[:, :-days_to_predict] = cases[:, num_dates_total-45:num_dates_total-days_to_predict]

submission_max_cases[:, :-days_to_predict] = cases[:, num_dates_total-45:num_dates_total-days_to_predict]





submission_mean_fatalities[:, :-days_to_predict] = fatalities[:, num_dates_total-45:num_dates_total-days_to_predict] / 10

submission_min_fatalities[:, :-days_to_predict] = fatalities[:, num_dates_total-45:num_dates_total-days_to_predict] / 10

submission_max_fatalities[:, :-days_to_predict] = fatalities[:, num_dates_total-45:num_dates_total-days_to_predict] / 10 
loss_1 = compute_loss(fatalities[:,-14:] / 10, submission_mean_fatalities[:,:14] , submission_min_fatalities[:,:14], 

                                   submission_max_fatalities[:,:14],weignts_cases * 10)
loss_2 = compute_loss(cases[:,-14:], submission_mean_cases[:,:14] , submission_min_cases[:,:14], 

                                   submission_max_cases[:,:14],weignts_cases)
loss_1
loss_2
(loss_1 + loss_2) / 2
#0.25734741800417726
submission_file = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
submission_file
submission = []

for i in range(len(submission_mean_cases)):

    for j in range(len(submission_mean_cases[0])):

        submission.append(submission_min_cases[i][j])

        submission.append(submission_mean_cases[i][j])

        submission.append(submission_max_cases[i][j])

        

        submission.append(submission_min_fatalities[i][j])

        submission.append(submission_mean_fatalities[i][j])

        submission.append(submission_max_fatalities[i][j])
submission = [max(0,x) for x in submission]
submission_file['TargetValue'] = submission
submission_file.to_csv('submission.csv', index = False)
submission_file
import matplotlib.pyplot as plt

def plot_results(true_array, mean_array, max_array, min_array):

    nans = np.array([None]*len(true_array))

    plt.plot(true_array)

    plt.plot(np.concatenate([nans,mean_array[-days_to_predict:]]))

    plt.plot(np.concatenate([nans,min_array[-days_to_predict:]]))

    plt.plot(np.concatenate([nans,max_array[-days_to_predict:]]))

    

    plt.show()
for i in np.random.randint(0, len(cases),30):

    plot_results(cases[i], submission_mean_cases[i], submission_max_cases[i], submission_min_cases[i])
for i in np.random.randint(0, len(cases),10):

    plot_results(fatalities[i], submission_mean_fatalities[i], submission_max_fatalities[i], submission_min_fatalities[i])