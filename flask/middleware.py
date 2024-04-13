import pandas as pd
import numpy as np
from flask import Flask, request, redirect, url_for, session, abort, flash, jsonify, g, render_template
from sklearn.preprocessing import LabelEncoder
import joblib
import datetime as dt
from datetime import datetime,timedelta
from datetime import date
from itertools import islice
import ast
import random as rn



def home_page():
    if request.method == 'POST':
        test_dict = eval(request.form["arguments"])
        test_df = pd.DataFrame(test_dict,index=[0])
        print(test_df.shape)
        cols = ['Voltage', 'Current', 'Charge_Discharge_Efficiency', 'Charge_Rate', 'Voltage_Sag', 'EIS_Data', 'Idle_Time', 'Charging_Time', 'Payload_Distribution', 'Acceleration', 'Cycle_Life', 'Chemical_Composition', 'Rate_of_Charge_Peak', 'Tire_Pressure', 'Regenerative_Braking']
        test_df.columns = cols
        for col in test_df.columns:
            test_df[col] = [int(x) for x in test_df[col]]

        test_df = test_df[['Voltage', 'Current', 'Charge_Discharge_Efficiency', 'Charge_Rate',
       'Voltage_Sag', 'EIS_Data', 'Cycle_Life', 'Chemical_Composition',
       'Rate_of_Charge_Peak', 'Tire_Pressure', 'Regenerative_Braking',
       'Idle_Time', 'Charging_Time', 'Payload_Distribution', 'Acceleration']]

        #loading model
        ensemble_model = joblib.load('../models/xgboost_model.joblib')

        variable = ensemble_model.predict(test_df)
        variable_prob = ensemble_model.predict_proba(test_df)

        attrition_rate_yes = str(variable_prob[0][1])
        attrition_rate_no = str(variable_prob[0][0])

        print(variable_prob)

        return  jsonify({
                'Class' : str(variable[0]),
                'Rate_yes': attrition_rate_yes[:5],
                'Rate_no':attrition_rate_no[:5] 
                 })
    else:
        return render_template('Home_page.html')



def home_page_date():
    return render_template('Home_page.html')