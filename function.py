import pandas as pd
import random
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV , cross_validate




def log_transform (df_clean):
    a = df_clean.skew(axis = 0, skipna = True).sort_values(ascending=False)
    ls_sk = list(a[a>2].index)
    df_clean_log = df_clean.copy()
    for i in ls_sk :
        df_clean_log[i]=np.log(1+df_clean_log[i])
        
    return df_clean_log
    

def analyse_modele (X,y,ls_model,dico_model) : 

    ls_rmse=[]
    ls_r2=[]
    ls_mae =[] 

    for model in  ls_model:
        score = cross_validate(model,X,y,cv=4,scoring=('neg_root_mean_squared_error','neg_mean_absolute_error','r2'))
        rmse =  score['test_neg_root_mean_squared_error'].mean()
        mae = score['test_neg_mean_absolute_error'].mean()
        r2 = score['test_r2'].mean()
        
        ls_rmse.append(-rmse)
        ls_mae.append(-mae)
        ls_r2.append(r2)
        
        
    df_RMSE = pd.DataFrame(ls_rmse)
    df_RMSE = df_RMSE.rename({0: "RMSE"},axis='columns')

    df_MAE = pd.DataFrame(ls_mae)
    df_MAE = df_MAE.rename({0: "MAE"},axis='columns')

    df_R2 = pd.DataFrame(ls_r2)
    df_R2 = df_R2.rename({0: "R2"},axis='columns')

    result_un = pd.concat([df_RMSE,df_MAE,df_R2],axis=1)
    result_un = result_un.rename(dico_model, axis='index')
    
    return result_un
    
    
def plot_metrics1 (analyse_un,analyse_un_log,metric) :
    ls_model = list(analyse_un.index)
    x=np.arange(len(ls_model))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    rects1 = ax.bar(x + 0.00, analyse_un[metric].values, color = 'b', width = 0.25)
    rects2 = ax.bar(x + 0.25, analyse_un_log[metric].values, color = 'g', width = 0.25)

    plt.xticks(range(0, len(ls_model)),ls_model)
    plt.title(metric)
    ax.legend(labels=['var normal', 'var log'])
    
    return
    

    
    
def plot_metrics2 (analyse_un_log,analyse_opt,metric) :
    ls_model = list(analyse_un_log.index)
    x=np.arange(len(ls_model))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    rects1 = ax.bar(x + 0.00, analyse_un_log[metric].values, color = 'b', width = 0.25)
    rects2 = ax.bar(x + 0.25, analyse_opt[metric].values, color = 'g', width = 0.25)

    plt.xticks(range(0, len(ls_model)),ls_model)
    plt.title(metric)
    ax.legend(labels=['modele simple', 'model opt'])
    
    return

def plot_metrics3 (analyse_sansES,analyse_avecES) :
    ls_metric = list(analyse_sansES.columns)
    x=np.arange(len(ls_metric))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    rects1 = ax.bar(x + 0.00, analyse_sansES.values[0], color = 'b', width = 0.25)
    rects2 = ax.bar(x + 0.25, analyse_avecES.values[0], color = 'g', width = 0.25)

    plt.xticks(range(0, len(ls_metric)),ls_metric)
    plt.title('Comparaison sans et avec ENERGYSTARScore')
    ax.legend(labels=['Sans ENERGYSTARScore', 'Avec ENERGYSTARScore'])
    
    return

