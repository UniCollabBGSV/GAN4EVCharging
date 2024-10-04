import gc
from tqdm import tqdm
import hashlib
import yaml
import warnings
from yaml import Loader

import pandas as pd
import numpy as np
import joblib
import random
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.exceptions import InconsistentVersionWarning

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)  

stream = open('configs//configs.yaml')
dictionary = yaml.load(stream, Loader=Loader)
LEN_SESSIONS = dictionary['LSTMGAN']["parammeters"]["len_session"]
cat_ohe = joblib.load('models//datapreprocessing//encoder//cat_ohe.pkl')
cat_oe = joblib.load('models//datapreprocessing//encoder//cat_oe.pkl')

pred_case_map = {
    "base": "use_base",
    "single_pred": "charge_speed_prediction",
    "part_pred_2": "charge_speed_prediction_part_2",
    "part_pred_5": "charge_speed_prediction_part_5",
}

th_dict_test = {
    "acn_office": {"decision_function_min": -3.0, "decision_function_mid": 0.0, "predict_proba_min": 0.5, "predict_proba_mid": 0.3},
    "acn_caltech": {"decision_function_min": -0.7, "decision_function_mid": 0.1, "predict_proba_min": 0.8, "predict_proba_mid": 0.5},
    "acn_jpl": {"decision_function_min": -2.0, "decision_function_mid": 0.0, "predict_proba_min": 0.64, "predict_proba_mid": 0.3}}


def discretize_hour_only(hour):
    if hour in [9, 10, 11, 12, 13, 14, 15, 16]:
        return "Work"
    elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:
        return "Sleep"
    elif hour in [17, 18, 19, 20, 21, 6, 7, 8]:
        return "Play"


def discretize_hour_ts(ts):
    hour = ts.hour
    return discretize_hour_only(hour)


def discretize_day_is_work(ts):
    day = ts.weekday()
    if day in [5, 6]:  # weekend
        return False
    else:
        return True


def discretize_hour_day(ts):
    hour = ts.hour
    day = ts.weekday()
    if day in [5, 6]:  # weekend
        if hour in [22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return "High-Home"
        elif hour in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
            return "High-Home"
            return "High-Leisure"
    else:
        if hour in [22, 23, 0, 1, 2, 3, 4, 5, 6, 7]:
            return "High-Home"
        elif hour in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
            return "High-Work"
        elif hour in [17, 18, 19, 20, 21]:
            return "High-Home"
            return "High-Leisure"
        else:
            raise Exception("discretize_hour_only", hour)


def discretize_hour_balancing(ts):
    hour = ts.hour
    if hour in [7, 8, 9]:
        return "peak"
    elif hour in [16, 17, 18, 19, 20, 21]:
        return "peak"
    elif hour in [22, 23, 24, 0, 1, 2, 3, 4, 5, 6]:
        return "low"
    elif hour in [10, 11, 12, 13, 14, 15]:
        return "low"


def get_date_exog(sum_df):
    sum_df['dayofweek'] = sum_df.index.to_series().dt.dayofweek
    sum_df['hour'] = sum_df.index.to_series().dt.hour
    sum_df['discretize_hour_only'] = sum_df.index.to_series().apply(
        discretize_hour_ts)
    sum_df['discretize_hour_day'] = sum_df.index.to_series().apply(
        discretize_hour_day)
    sum_df['discretize_day_is_work'] = sum_df.index.to_series().apply(
        discretize_day_is_work)
    sum_df['discretize_hour_balancing'] = sum_df.index.to_series().apply(
        discretize_hour_balancing)
    return sum_df.copy()


def do_get_opt_cols(_config=None, _lft=None, part_preds=False, only_clf=None):
    if _config == "acn_office" and _lft == "None" and part_preds == "use_base" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_office_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 30, 'clf__algorithm': 'auto'}, 'f1': 0.89193, 'cols': ['capacity_connected', 'dayofweek_end', 'discretize_hour_day_end_High-Home', 'hour_end'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'capacity_connected', 'dayofweek_end', 'discretize_hour_day_end_High-Home', 'hour_end']}}
    elif _config == "acn_office" and _lft == "None" and part_preds == "charge_speed_prediction" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_office_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 30, 'clf__algorithm': 'auto'}, 'f1': 0.95011, 'cols': ['charge_speed_prediction_diff_max', 'charge_speed_prediction_median', 'charge_speed_prediction_summary_diff_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'actual_charge_rel', 'charge_speed_median', 'discretize_hour_only_median_Play', 'charge_speed_prediction_diff_max', 'charge_speed_prediction_median', 'charge_speed_prediction_summary_diff_max']}}
    elif _config == "acn_office" and _lft == "None" and part_preds == "charge_speed_prediction_part_5" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_office_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 30, 'clf__algorithm': 'auto'}, 'f1': 0.9595, 'cols': ['charge_speed_prediction+2_sum', 'charge_speed_prediction_diff+4_median_relative', 'charge_speed_prediction_summary_diff+4_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'actual_charge_rel', 'charge_speed_median', 'discretize_hour_only_median_Play', 'charge_speed_prediction+2_sum', 'charge_speed_prediction_diff+4_median_relative', 'charge_speed_prediction_summary_diff+4_max']}}

    elif _config == "acn_caltech" and _lft == "None" and part_preds == "use_base" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_caltech_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 2, 'clf__novelty': True, 'clf__n_neighbors': 20, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 30, 'clf__algorithm': 'auto'}, 'f1': 0.8937, 'cols': ['charge_speed_mean', 'charge_speed_sum', 'charge_time_end', 'cp_id', 'dayofweek_median', 'discretize_hour_balancing_end_low', 'distinct_charge_speeds', 'hour_median'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_mean', 'charge_speed_sum', 'charge_time_end', 'cp_id', 'dayofweek_median', 'discretize_hour_balancing_end_low', 'distinct_charge_speeds', 'hour_median']}}
    elif _config == "acn_caltech" and _lft == "None" and part_preds == "charge_speed_prediction" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_caltech_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 2, 'clf__novelty': True, 'clf__n_neighbors': 20, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 30, 'clf__algorithm': 'auto'}, 'f1': 0.95616, 'cols': ['charge_speed_prediction_diff_max', 'charge_speed_prediction_max', 'charge_speed_prediction_summary_diff_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_mean', 'charge_speed_sum', 'charge_speed_prediction_diff_max', 'charge_speed_prediction_max', 'charge_speed_prediction_summary_diff_max']}}
    elif _config == "acn_caltech" and _lft == "None" and part_preds == "charge_speed_prediction_part_5" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_caltech_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 2, 'clf__novelty': True, 'clf__n_neighbors': 20, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 30, 'clf__algorithm': 'auto'}, 'f1': 0.95138, 'cols': ['charge_speed_prediction_diff+5_max', 'charge_speed_prediction_diff+5_median_relative', 'charge_speed_prediction_sum', 'charge_speed_prediction_summary_diff+3_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_mean', 'charge_speed_sum', 'charge_speed_prediction_diff+5_max', 'charge_speed_prediction_diff+5_median_relative', 'charge_speed_prediction_sum', 'charge_speed_prediction_summary_diff+3_max']}}

    elif _config == "acn_jpl" and _lft == "None" and part_preds == "use_base" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_jpl_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 40, 'clf__algorithm': 'auto'}, 'f1': 0.88699, 'cols': ['charge_time_end', 'charge_time_start', 'dayofweek_end', 'discretize_day_is_work_end_True', 'discretize_hour_day_median_High-Home', 'hour_end'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_time_end', 'charge_time_start', 'dayofweek_end', 'discretize_day_is_work_end_True', 'discretize_hour_day_median_High-Home', 'hour_end']}}
    elif _config == "acn_jpl" and _lft == "None" and part_preds == "charge_speed_prediction" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_jpl_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 40, 'clf__algorithm': 'auto'}, 'f1': 0.96505, 'cols': ['charge_speed_prediction_diff_max', 'charge_speed_prediction_sum', 'charge_speed_prediction_summary_diff_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_sum', 'charge_speed_prediction_diff_max', 'charge_speed_prediction_sum', 'charge_speed_prediction_summary_diff_max']}}
    elif _config == "acn_jpl" and _lft == "None" and part_preds == "charge_speed_prediction_part_5" and only_clf == "LocalOutlierFactor":
        opt_cols_d = {'acn_jpl_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 40, 'clf__algorithm': 'auto'}, 'f1': 0.96639, 'cols': ['charge_speed_prediction+4_median', 'charge_speed_prediction+4_sum', 'charge_speed_prediction_diff+4_mean', 'charge_speed_prediction_diff+4_sum_relative', 'charge_speed_prediction_diff+5_sum_relative', 'charge_speed_prediction_summary_diff+3_min', 'charge_speed_prediction_summary_diff+4_min', 'charge_speed_prediction_summary_diff_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_sum', 'charge_speed_prediction+4_median', 'charge_speed_prediction+4_sum', 'charge_speed_prediction_diff+4_mean', 'charge_speed_prediction_diff+4_sum_relative', 'charge_speed_prediction_diff+5_sum_relative', 'charge_speed_prediction_summary_diff+3_min', 'charge_speed_prediction_summary_diff+4_min', 'charge_speed_prediction_summary_diff_max']}}

    # elif _config == "elaadnl" and _lft == "None" and part_preds == "use_base" and only_clf == "LocalOutlierFactor":
    #     opt_cols_d = {'elaadnl_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 40, 'clf__algorithm': 'auto'}, 'f1': 0.89697, 'cols': ['charge_time_end', 'charge_time_start', 'cp_id', 'dayofweek_start', 'discretize_hour_only_start_Work', 'hour_median'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_time_end', 'charge_time_start', 'cp_id', 'dayofweek_start', 'discretize_hour_only_start_Work', 'hour_median']}}
    # elif _config == "elaadnl" and _lft == "None" and part_preds == "charge_speed_prediction" and only_clf == "LocalOutlierFactor":
    #     opt_cols_d =  {'elaadnl_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 40, 'clf__algorithm': 'auto'}, 'f1': 0.88458, 'cols': ['charge_speed_prediction_diff_max', 'charge_speed_prediction_diff_median_relative', 'charge_speed_prediction_median_relative', 'charge_speed_prediction_summary_diff_mean_relative'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_mean', 'charge_speed_sum', 'charge_speed_prediction_diff_max', 'charge_speed_prediction_diff_median_relative', 'charge_speed_prediction_median_relative', 'charge_speed_prediction_summary_diff_mean_relative']}}
    # elif _config == "elaadnl" and _lft == "None" and part_preds == "charge_speed_prediction_part_5" and only_clf == "LocalOutlierFactor":
    #     opt_cols_d =  {'elaadnl_None': {'clf': 'LocalOutlierFactor', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__p': 1, 'clf__novelty': True, 'clf__n_neighbors': 10, 'clf__n_jobs': 6, 'clf__metric_params': None, 'clf__metric': 'minkowski', 'clf__leaf_size': 40, 'clf__algorithm': 'auto'}, 'f1': 0.89592, 'cols': ['charge_speed_prediction+2_sum', 'charge_speed_prediction+5_mean', 'charge_speed_prediction_summary_diff+5_sum'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_mean', 'charge_speed_sum', 'charge_speed_prediction+2_sum', 'charge_speed_prediction+5_mean', 'charge_speed_prediction_summary_diff+5_sum']}}

    elif _config == "acn_office" and _lft == "None" and part_preds == "use_base" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_office_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.89523, 'cols': ['charge_speed_changes', 'discretize_hour_day_median_High-Home', 'discretize_hour_day_median_High-Work'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_changes', 'discretize_hour_day_median_High-Home', 'discretize_hour_day_median_High-Work']}}
    elif _config == "acn_office" and _lft == "None" and part_preds == "charge_speed_prediction" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_office_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.988, 'cols': ['charge_speed_prediction_diff_max', 'charge_speed_prediction_diff_min', 'charge_speed_prediction_summary_diff_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'actual_charge_time', 'charge_speed_sum', 'distinct_charge_speeds', 'charge_speed_prediction_diff_max', 'charge_speed_prediction_diff_min', 'charge_speed_prediction_summary_diff_max']}}
    elif _config == "acn_office" and _lft == "None" and part_preds == "charge_speed_prediction_part_5" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_office_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.98647, 'cols': ['charge_speed_prediction+5_max', 'charge_speed_prediction_summary_diff+2_max', 'charge_speed_prediction_summary_diff_median', 'charge_speed_prediction_summary_diff_median_relative'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'actual_charge_time', 'charge_speed_sum', 'distinct_charge_speeds', 'charge_speed_prediction+5_max', 'charge_speed_prediction_summary_diff+2_max', 'charge_speed_prediction_summary_diff_median', 'charge_speed_prediction_summary_diff_median_relative']}}

    elif _config == "acn_caltech" and _lft == "None" and part_preds == "use_base" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_caltech_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.8937, 'cols': ['actual_charge_rel', 'actual_charge_time', 'discretize_hour_only_end_Play'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'actual_charge_rel', 'actual_charge_time', 'discretize_hour_only_end_Play']}}
    elif _config == "acn_caltech" and _lft == "None" and part_preds == "charge_speed_prediction" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_caltech_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.97846, 'cols': ['charge_speed_prediction_mean', 'charge_speed_prediction_summary_diff_max', 'charge_speed_prediction_summary_diff_median'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_changes', 'charge_speed_mean', 'charge_speed_sum', 'distinct_charge_speeds', 'charge_speed_prediction_mean', 'charge_speed_prediction_summary_diff_max', 'charge_speed_prediction_summary_diff_median']}}
    elif _config == "acn_caltech" and _lft == "None" and part_preds == "charge_speed_prediction_part_5" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_caltech_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.97928, 'cols': ['charge_speed_prediction_diff+2_sum', 'charge_speed_prediction_diff_median', 'charge_speed_prediction_summary_diff+2_median_relative', 'charge_speed_prediction_summary_diff_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_changes', 'charge_speed_mean', 'charge_speed_sum', 'distinct_charge_speeds', 'charge_speed_prediction_diff+2_sum', 'charge_speed_prediction_diff_median', 'charge_speed_prediction_summary_diff+2_median_relative', 'charge_speed_prediction_summary_diff_max']}}

    elif _config == "acn_jpl" and _lft == "None" and part_preds == "use_base" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_jpl_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.88964, 'cols': ['discretize_day_is_work_median_True'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'discretize_day_is_work_median_True']}}
    elif _config == "acn_jpl" and _lft == "None" and part_preds == "charge_speed_prediction" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_jpl_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.98035, 'cols': ['charge_speed_prediction_diff_mean_relative', 'charge_speed_prediction_summary_diff_max', 'charge_speed_prediction_summary_diff_mean'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_changes', 'charge_speed_mean', 'distinct_charge_speeds', 'charge_speed_prediction_diff_mean_relative', 'charge_speed_prediction_summary_diff_max', 'charge_speed_prediction_summary_diff_mean']}}
    elif _config == "acn_jpl" and _lft == "None" and part_preds == "charge_speed_prediction_part_5" and only_clf == "MLPClassifier":
        opt_cols_d = {'acn_jpl_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (100, 100), 'clf__epsilon': 1e-09, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.98047, 'cols': ['charge_speed_prediction+5_median_relative', 'charge_speed_prediction_summary_diff_max', 'charge_speed_prediction_summary_diff_median_relative'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'charge_speed_changes', 'charge_speed_mean', 'distinct_charge_speeds', 'charge_speed_prediction+5_median_relative', 'charge_speed_prediction_summary_diff_max', 'charge_speed_prediction_summary_diff_median_relative']}}

    # elif _config == "elaadnl" and _lft == "None" and part_preds == "use_base" and only_clf == "MLPClassifier":
    #     opt_cols_d = {'elaadnl_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (50, 50), 'clf__epsilon': 1e-08, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.89712, 'cols': ['actual_charge_rel'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'actual_charge_rel']}}
    # elif _config == "elaadnl" and _lft == "None" and part_preds == "charge_speed_prediction" and only_clf == "MLPClassifier":
    #     opt_cols_d = {'elaadnl_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (50, 50), 'clf__epsilon': 1e-08, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.90488, 'cols': ['charge_speed_prediction_diff_max', 'charge_speed_prediction_diff_mean', 'charge_speed_prediction_diff_median_relative', 'charge_speed_prediction_sum', 'charge_speed_prediction_summary_diff_max', 'charge_speed_prediction_summary_diff_mean_relative'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'actual_charge_time', 'charge_speed_changes', 'charge_speed_mean', 'distinct_charge_speeds', 'charge_speed_prediction_diff_max', 'charge_speed_prediction_diff_mean', 'charge_speed_prediction_diff_median_relative', 'charge_speed_prediction_sum', 'charge_speed_prediction_summary_diff_max', 'charge_speed_prediction_summary_diff_mean_relative']}}
    # elif _config == "elaadnl" and _lft == "None" and part_preds == "charge_speed_prediction_part_5" and only_clf == "MLPClassifier":
    #     opt_cols_d = {'elaadnl_None': {'clf': 'MLPClassifier', 'pipe': 'RandomizedSearchCVPipelineStandardScaler', 'params': {'clf__warm_start': False, 'clf__verbose': False, 'clf__validation_fraction': 0.1, 'clf__shuffle': True, 'clf__random_state': 12345, 'clf__nesterovs_momentum': True, 'clf__n_iter_no_change': 10, 'clf__max_iter': 200000, 'clf__max_fun': 1500000, 'clf__hidden_layer_sizes': (50, 50), 'clf__epsilon': 1e-08, 'clf__batch_size': 'auto', 'clf__alpha': 0.0001}, 'f1': 0.91186, 'cols': ['charge_speed_prediction+2_mean_relative', 'charge_speed_prediction_diff+2_max', 'charge_speed_prediction_summary_diff+5_max', 'charge_speed_prediction_summary_diff_max'], 'cols_full': ['charge_amount', 'distinct_charge_speeds_p', 'charge_speed_changes_p', 'charge_speed_sum_rel', 'actual_charge_time', 'charge_speed_changes', 'charge_speed_mean', 'distinct_charge_speeds', 'charge_speed_prediction+2_mean_relative', 'charge_speed_prediction_diff+2_max', 'charge_speed_prediction_summary_diff+5_max', 'charge_speed_prediction_summary_diff_max']}}
    else:
        raise ValueError("Unk params do_get_opt_cols:", _config, _lft, part_preds, only_clf)
    return opt_cols_d


def get_session_features(CONFIG, dff, df_pred_file_n, ret_type="df", try_new_cols=["use_base"], disable_tqdm=False):
    FINAL_TIME_STEP_MIN = 15 if CONFIG == 'elaadnl' else 1

    if type(df_pred_file_n) is pd.DataFrame:
        if "use_base" not in try_new_cols:
            df_pred = df_pred_file_n
            df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
            df_pred = df_pred_file_n.set_index('timestamp')

    if "charge_speed_prediction_part_5" in try_new_cols:
        df_pred.rename(columns={"charge_speed_prediction+1": "charge_speed_prediction"}, inplace=True)

    df = dff
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = dff.set_index('timestamp')
    df = get_date_exog(df)
    df["charge_speed_should_diff"] = df["charge_speed"].replace(0, 1) / df["charge_speed_should"].replace(0, 1)

    if "charge_speed_prediction" in try_new_cols:
        df["charge_speed_prediction"] = df_pred["charge_speed_prediction"]
        del df_pred
        gc.collect()
        df["charge_speed_prediction_diff"] = (df["charge_speed"] - df["charge_speed_prediction"]).abs()

    elif "charge_speed_prediction_part_2" in try_new_cols or "charge_speed_prediction_part_5" in try_new_cols:  # timestamp	charge_speed_prediction	session_id	charge_speed_prediction+2	charge_speed_prediction+3	charge_speed_prediction+4	charge_speed_prediction+5
        all_df_pred = []
        for sid in df["session_id"].drop_duplicates()[:]:
            df_pred_sid = df_pred[df_pred["session_id"] == sid]
            if len(df_pred_sid) != 0:
                df_pred_1 = df_pred_sid.iloc[[0]]
                df_pred_1 = df_pred_1.reset_index()
                df_pred_1["timestamp"] = df_pred_1["timestamp"] - pd.Timedelta(str(FINAL_TIME_STEP_MIN) + "Min")
                df_pred_1 = df_pred_1.set_index("timestamp")
                df_pred_1["charge_speed_prediction"] = np.nan
                df_pred_sid = pd.concat((df_pred_1, df_pred_sid))
            else:
                df_pred_sid = df_pred_sid.copy()
                df_pred_sid.loc[df[df["session_id"] == sid].index.to_list()[0]] = np.nan
                df_pred_sid["session_id"] = sid
                # print(df_pred_sid)
            all_df_pred.append(df_pred_sid)
        df_pred = pd.concat(all_df_pred)
        try:
            df["charge_speed_prediction"] = df_pred["charge_speed_prediction"]
        except Exception as e:
            print(dff, df_pred_file_n)
            print(df)
            print(df_pred)
            print(e)
            raise e
        df.loc[df["charge_speed_prediction"].isna(), "charge_speed_prediction"] = df[df["charge_speed_prediction"].isna()]["charge_speed"]
        df["charge_speed_prediction_diff"] = (df["charge_speed"] - df["charge_speed_prediction"]).abs()
        if "charge_speed_prediction_part_2" in try_new_cols:
            _x_range = range(2, 3)
        elif "charge_speed_prediction_part_5" in try_new_cols:
            _x_range = range(2, 6)
        for x in _x_range:
            df["charge_speed_prediction+"+str(x)] = df_pred["charge_speed_prediction+"+str(x)]
            df.loc[df["charge_speed_prediction+"+str(x)].isna(), "charge_speed_prediction+"+str(x)] = df[df["charge_speed_prediction+"+str(x)].isna()]["charge_speed"]
            df["charge_speed_prediction_diff+"+str(x)] = (df["charge_speed"] - df["charge_speed_prediction+"+str(x)]).abs()
        del df_pred
        gc.collect()

    df["charge_amount"] = df["charge_speed"] * (FINAL_TIME_STEP_MIN / 60)
    all_sids = df["session_id"].drop_duplicates()[:]
    df["capacity_connected"] = df["capacity_connected"].fillna(0)
    features_list = []

    for sid in tqdm(all_sids, desc=dff, disable=disable_tqdm):
        dfs = df[df["session_id"] == sid]
        features = {
            "_sid": sid,
            "cp_id": int(hashlib.sha256(dfs["cpID"].mode()[0].encode('utf-8')).hexdigest(), 16) % 10**8,
            "charge_amount": dfs["energy_register"][-1] - dfs["energy_register"][0],
            "charge_time_diff": dfs.index[-1].timestamp() - dfs.index[0].timestamp(),
            "charge_time_start": dfs.index[0].timestamp(),
            "charge_time_end": dfs.index[-1].timestamp(),

            "actual_charge_time": len(dfs[dfs["charge_speed"] > 0]) * FINAL_TIME_STEP_MIN * 60,  # in s
            "actual_charge_rel": len(dfs[dfs["charge_speed"] > 0]) / len(dfs),

            "capacity": dfs["capacity"].median(),
            "capacity_connected": dfs["capacity_connected"].median(),

            "distinct_charge_speeds": len(dfs["charge_speed"].round(1).drop_duplicates()),
            "charge_speed_changes": (dfs["charge_speed"].round(1).diff() != 0.0).sum(),
            "_is_attack": dfs["is_attack"].mean(),
            "_is_attack_2": ((dfs["charge_speed_should_diff"] >= 2) | (dfs["charge_speed_should_diff"] <= 0.5)).sum(),
            "num_values": len(dfs),
            "discretize_hour_only_median": dfs["discretize_hour_only"].mode()[0],
            "discretize_hour_day_median": dfs["discretize_hour_day"].mode()[0],
            "discretize_day_is_work_median": dfs["discretize_day_is_work"].mode()[0],
            "discretize_hour_balancing_median": dfs["discretize_hour_balancing"].mode()[0],
            "dayofweek_median": dfs["dayofweek"].median(),
            "hour_median": dfs["hour"].median(),
            "discretize_hour_only_start": dfs["discretize_hour_only"][0],
            "discretize_hour_day_start": dfs["discretize_hour_day"][0],
            "discretize_day_is_work_start": dfs["discretize_day_is_work"][0],
            "discretize_hour_balancing_start": dfs["discretize_hour_balancing"][0],
            "dayofweek_start": dfs["dayofweek"][0],
            "hour_start": dfs["hour"][0],
            "discretize_hour_only_end": dfs["discretize_hour_only"][-1],
            "discretize_hour_day_end": dfs["discretize_hour_day"][-1],
            "discretize_day_is_work_end": dfs["discretize_day_is_work"][-1],
            "discretize_hour_balancing_end": dfs["discretize_hour_balancing"][-1],
            "dayofweek_end": dfs["dayofweek"][-1],
            "hour_end": dfs["hour"][-1],
        }

        if "charge_speed_prediction" in try_new_cols or "charge_speed_prediction_part_2" in try_new_cols or "charge_speed_prediction_part_5" in try_new_cols:
            try:
                features["charge_speed_prediction_diff_mape"] = mean_absolute_percentage_error(df["charge_speed"], df["charge_speed_prediction"])
            except Exception as e:
                print(e)
                print(dff)
                print(df_pred_file_n)
                print(df["charge_speed"])
                print(df["charge_speed_prediction"])
                raise e

            features["charge_speed_prediction_diff_rmse"] = mean_squared_error(df["charge_speed"], df["charge_speed_prediction"], squared=False)

        if "charge_speed_prediction_part_2" in try_new_cols or "charge_speed_prediction_part_5" in try_new_cols:
            for x in _x_range:
                features["charge_speed_prediction_diff_mape+"+str(x)] = mean_absolute_percentage_error(df["charge_speed"], df["charge_speed_prediction+"+str(x)])
                features["charge_speed_prediction_diff_rmse+"+str(x)] = mean_squared_error(df["charge_speed"], df["charge_speed_prediction+"+str(x)], squared=False)

        cs_cols = ["charge_speed", "_charge_speed_should"]  # "_charge_speed_smgw"

        if "charge_speed_prediction" in try_new_cols or "charge_speed_prediction_part_2" in try_new_cols or "charge_speed_prediction_part_5" in try_new_cols:
            cs_cols += ["charge_speed_prediction", "charge_speed_prediction_diff"]
        if "charge_speed_prediction_part_2" in try_new_cols or "charge_speed_prediction_part_5" in try_new_cols:
            for x in _x_range:
                cs_cols += ["charge_speed_prediction+"+str(x), "charge_speed_prediction_diff+"+str(x)]

        for cs in cs_cols:
            features[cs+"_mean"] = dfs[cs.strip("_")].mean()
            features[cs+"_median"] = dfs[cs.strip("_")].median()
            features[cs+"_sum"] = dfs[cs.strip("_")].sum()
            features[cs+"_max"] = dfs[cs.strip("_")].max()
            features[cs+"_min"] = dfs[cs.strip("_")].min()

            if features[cs+"_sum"] != 0:
                features[cs+"_mean_relative"] = features[cs+"_mean"] / features[cs+"_sum"]
                features[cs+"_median_relative"] = features[cs+"_median"] / features[cs+"_sum"]
            else:
                features[cs+"_mean_relative"] = features[cs+"_mean"] / 1
                features[cs+"_median_relative"] = features[cs+"_median"] / 1

            features[cs+"_sum_relative"] = features[cs+"_sum"] / features["num_values"]

        if "charge_speed_prediction" in try_new_cols or "charge_speed_prediction_part_2" in try_new_cols or "charge_speed_prediction_part_5" in try_new_cols:
            for cs_t in ["_mean", "_median", "_sum", "_max", "_min", "_mean_relative", "_median_relative", "_sum_relative"]:
                if features["charge_speed_prediction"+cs_t] != 0:
                    features["charge_speed_prediction_summary_diff"+cs_t] = features["charge_speed"+cs_t] / features["charge_speed_prediction"+cs_t]
                else:
                    features["charge_speed_prediction_summary_diff"+cs_t] = features["charge_speed"+cs_t] / 1

        if "charge_speed_prediction_part_2" in try_new_cols or "charge_speed_prediction_part_5" in try_new_cols:
            for x in _x_range:
                for cs_t in ["_mean", "_median", "_sum", "_max", "_min", "_mean_relative", "_median_relative", "_sum_relative"]:
                    if features["charge_speed_prediction+"+str(x)+cs_t] != 0:
                        features["charge_speed_prediction_summary_diff+"+str(x)+cs_t] = features["charge_speed"+cs_t] / features["charge_speed_prediction+"+str(x)+cs_t]
                    else:
                        features["charge_speed_prediction_summary_diff+"+str(x)+cs_t] = features["charge_speed"+cs_t] / 1

        features["distinct_charge_speeds_p"] = features["distinct_charge_speeds"] / features["num_values"]
        features["charge_speed_changes_p"] = features["charge_speed_changes"] / features["num_values"]
        features["charge_time_diff"] += FINAL_TIME_STEP_MIN * 60
        features["charge_speed_sum_rel"] = features["charge_speed_sum"] / features["charge_time_diff"]
        features_list.append(features)

    if ret_type == "df":
        feat_df = pd.DataFrame(features_list)
    elif ret_type == "dict":
        feat_df = features_list
    return feat_df


def run_get_dfs(dff, x_type, param_set, insert_atks=False, add_atk_id_cols=False):
    if x_type == "df":
        dfs_dict = dff
        c_exo_all_dummies = dfs_dict.select_dtypes(exclude=[np.number]).columns
        dfs_dict = pd.get_dummies(dfs_dict, columns=c_exo_all_dummies)
        cols = [c for c in dfs_dict.columns if not c.startswith("_")]
        if param_set == "Min":
            cols_min = ['cp_id', 'charge_amount', 'charge_time_diff', 'distinct_charge_speeds_p', 'charge_speed_changes_p', "charge_speed_sum_rel"]
            cols = [c for c in cols if c in cols_min]
        elif param_set == "All":
            pass
        else:
            c_insert = [c for c in param_set if c not in cols]
            if c_insert:
                # print("inserting:", str(c_insert))
                dfs_dict[c_insert] = 0
                cols = [c for c in dfs_dict.columns if not c.startswith("_")]
            cols = [c for c in cols if c in param_set]
        if insert_atks or add_atk_id_cols:
            cols.append("_is_attack")
            cols.append("_is_attack_2")
        dfs_dict = dfs_dict[sorted(cols)]
        v = None
    else:
        raise ValueError(x_type)
    return dfs_dict, v


def create_prediction_file_1(df, CONFIG):
    """Create a single next charge speed prediction file. """
    model = joblib.load(f'models//regression_model//{CONFIG}//single.pkl')
    df_new = pd.DataFrame()
    df_new['timestamp'] = df['timestamp']
    df_new['session_id'] = df['session_id']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    flag = dictionary['regressor']['flag'][CONFIG]
    df = get_date_exog(df)
    y = np.array([])
    for sid in df["session_id"].drop_duplicates()[:]:
        df_sid = df[df["session_id"] == sid]
        for i in range(1, flag + 1):
            df_sid['charge_speed_lag_' + str(i)] = df_sid['charge_speed'].shift(i)
        df_sid.fillna(0.0)
        x_lag = np.array(df_sid[['charge_speed_lag_' + str(i) for i in range(1, flag + 1)]])
        x_num_pad = np.array(df_sid[['energy_register', 'capacity_connected']])
        x_num_pad = np.pad(x_num_pad[:-1], ((1, 0), (0, 0)), mode='constant', constant_values=0)
        x_num = np.array(df_sid[['dayofweek', 'hour']])
        x_cat_ohe = np.array(cat_ohe.transform(df_sid[['discretize_hour_only']]).toarray())
        x_cat_oe = np.array(cat_oe.transform(df_sid[['discretize_hour_day', 'discretize_day_is_work', 'discretize_hour_balancing']]))
        x_train = np.concatenate((x_lag, x_num_pad, x_num, x_cat_ohe, x_cat_oe), axis=1)
        prediction = model.predict(x_train)
        y = np.concatenate([y, prediction])
    df_new['charge_speed_prediction'] = y.astype(np.float32)
    return pd.DataFrame(df_new)


def create_prediction_file_5(df, CONFIG):
    """Create a 5-next charge speed prediction file. """
    model_1 = joblib.load(f'models//regression_model//{CONFIG}//part1.pkl')
    model_2 = joblib.load(f'models//regression_model//{CONFIG}//part2.pkl')
    model_3 = joblib.load(f'models//regression_model//{CONFIG}//part3.pkl')
    model_4 = joblib.load(f'models//regression_model//{CONFIG}//part4.pkl')
    model_5 = joblib.load(f'models//regression_model//{CONFIG}//part5.pkl')
    flag = dictionary['regressor']['flag'][CONFIG]
    models = [model_1, model_2, model_3, model_4, model_5]
    df_new = pd.DataFrame()
    data = df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = get_date_exog(df)
    predictions = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
    for sid in df["session_id"].drop_duplicates()[:]:
        df_sid = df[df["session_id"] == sid]
        num_fill = len(df_sid) - 1
        for i in range(1, 6):
            df_sid_i = df_sid[:-i]
            if len(df_sid_i) == 0:
                prediction = np.insert(np.array([]), 0, [np.nan] * num_fill)
                predictions[i-1] = np.concatenate([predictions[i-1], prediction])
            else:
                for idx in range(0, flag):
                    df_sid_i['charge_speed_lag_' + str(idx)] = df_sid_i['charge_speed'].shift(idx)
                df_sid_i.fillna(0.0)
                x_lag = np.array(df_sid_i[['charge_speed_lag_' + str(i) for i in range(0, flag)]])
                x_num_pad = np.array(df_sid_i[['energy_register', 'capacity_connected']])
                x_num = np.array(df_sid_i[['dayofweek', 'hour']])
                x_cat_ohe = np.array(cat_ohe.transform(df_sid_i[['discretize_hour_only']]).toarray())
                x_cat_oe = np.array(cat_oe.transform(df_sid_i[['discretize_hour_day', 'discretize_day_is_work', 'discretize_hour_balancing']]))
                x_train = np.concatenate((x_lag, x_num_pad, x_num, x_cat_ohe, x_cat_oe), axis=1)
                prediction = models[i-1].predict(x_train)
                prediction = np.insert(prediction, 0, [np.nan] * (i-1))
                predictions[i-1] = np.concatenate([predictions[i-1], prediction])
    for i in range(5):
        df_new['charge_speed_prediction+' + str(i+1)] = predictions[i]

    def remove_first_n_row(group, n):
        return group.iloc[n:]

    data = data.groupby('session_id').apply(remove_first_n_row, n=1, include_groups=True).reset_index(drop=True)
    df_new['timestamp'] = data['timestamp']
    df_new['session_id'] = data['session_id']
    return pd.DataFrame(df_new)


def extract_features_from_raw_data(df_base_file, df_pred_single, df_pred_part_5, CONFIG, do_clf, ):
    """Extract feature from csv file before feeding to ML models. """
    pred_cases = ["single_pred", "part_pred_5"]
    dfs_train_d = dict()
    lft = "None"
    opt_cols = do_get_opt_cols(_config=CONFIG.rstrip("2"), _lft=lft, part_preds=pred_case_map[pred_cases[0]], only_clf=do_clf)
    best = opt_cols[CONFIG.rstrip("2")+"_"+lft]
    df_feat = get_session_features(CONFIG=CONFIG, dff=df_base_file, ret_type="df", try_new_cols=[pred_case_map[pred_cases[0]]],
                                   df_pred_file_n=df_pred_single, disable_tqdm=True)
    dfs_train, v = run_get_dfs(df_feat, x_type='df', param_set=best["cols_full"], insert_atks=False, add_atk_id_cols=True)
    dfs_train_d[pred_cases[0]] = dfs_train

    opt_cols = do_get_opt_cols(_config=CONFIG.rstrip("2"), _lft=lft, part_preds=pred_case_map[pred_cases[1]], only_clf=do_clf)
    best = opt_cols[CONFIG.rstrip("2")+"_"+lft]
    df_feat = get_session_features(CONFIG=CONFIG, dff=df_base_file, ret_type="df", try_new_cols=[pred_case_map[pred_cases[1]]],
                                   df_pred_file_n=df_pred_part_5, disable_tqdm=True)
    dfs_train, v = run_get_dfs(df_feat, x_type='df', param_set=best["cols_full"], insert_atks=False, add_atk_id_cols=True)
    dfs_train_d[pred_cases[1]] = dfs_train

    dfs_train = dfs_train_d[pred_cases[0]]

    if len(pred_cases) > 1:
        for k in pred_cases[1:]:
            v = dfs_train_d[k]
            dfs_train_d_cols = [c for c in v.columns if "charge_speed_prediction" in c]
            dfs_train[[k+"_"+c for c in dfs_train_d_cols]] = v[dfs_train_d_cols]
    dfs_train["_y"] = 1
    dfs_train.loc[dfs_train["_is_attack"] > 0, "_y"] = -1
    dfs_train.drop(columns=['_is_attack', '_is_attack_2'], inplace=True)
    return pd.DataFrame(dfs_train)


def ensemble_model(df_base_file, CONFIG):
    """End-to-end ensemble model to obtain the final prediction from ML models. """
    df_pred_single = create_prediction_file_1(df_base_file, CONFIG)
    df_pred_part_5 = create_prediction_file_5(df_base_file, CONFIG)
    data_clf = extract_features_from_raw_data(df_base_file=df_base_file, df_pred_single=df_pred_single, df_pred_part_5=df_pred_part_5, CONFIG=CONFIG, do_clf='MLPClassifier')
    data_nov = extract_features_from_raw_data(df_base_file=df_base_file, df_pred_single=df_pred_single, df_pred_part_5=df_pred_part_5, CONFIG=CONFIG, do_clf='LocalOutlierFactor')

    decision_function_min = th_dict_test[CONFIG]["decision_function_min"]
    predict_proba_min = th_dict_test[CONFIG]["predict_proba_min"]
    decision_function_mid = th_dict_test[CONFIG]["decision_function_mid"]
    predict_proba_mid = th_dict_test[CONFIG]["predict_proba_mid"]

    model_clf = joblib.load(f'models//regressor_model//{CONFIG}//clf.pkl')
    model_nov = joblib.load(f'models//regressor_model//{CONFIG}//nov.pkl')

    predict_probas = model_clf.predict_proba(data_clf[list(c for c in data_clf.columns if c != "_y" and c != "Unnamed: 0" and c != "charge_time_diff" and c != "cp_id")])
    decision_functions = model_nov.decision_function(data_nov[list(c for c in data_nov.columns if c != "_y" and c != "Unnamed: 0" and c != "charge_time_diff" and c != "cp_id")])

    classes_new = [-1, 1]
    y_pred = []
    for decision_function, predict_proba in zip(decision_functions.tolist(), predict_probas.tolist()):
        pred_new = 0
        pred_old = 0
        classes_atk = 0 if classes_new[0] == -1 else 1
        if type(classes_new) is not list:
            raise Exception("type(classes_new) != list")
        if decision_function < decision_function_min:
            pred_new = -1
        elif decision_function < decision_function_mid:
            pred_new = 0
        else:
            pred_new = 1
        if predict_proba[classes_atk] > predict_proba_min:
            pred_old = classes_new[classes_atk]
        elif predict_proba[classes_atk] > predict_proba_mid:
            pred_old = 0
        else:
            pred_old = classes_new[classes_atk]*-1

        if pred_new == -1 or pred_old == -1:
            y_pred.append(-1)
        elif pred_new == 0 and pred_old == 0:
            y_pred.append(-1)
        else:
            y_pred.append(1)
    y_pred = np.array(y_pred)
    y_pred[y_pred == -1] = 0
    return y_pred


def localoutlierfactor(df_base_file, CONFIG):
    """End-to-end LocalOutlierFactor model to obtain the final prediction. """
    df_pred_single = create_prediction_file_1(df_base_file, CONFIG)
    df_pred_part_5 = create_prediction_file_5(df_base_file, CONFIG)
    data_nov = extract_features_from_raw_data(df_base_file=df_base_file, df_pred_single=df_pred_single, df_pred_part_5=df_pred_part_5, CONFIG=CONFIG, do_clf='LocalOutlierFactor')
    model_nov = joblib.load(f'models//ML_models//{CONFIG}//localoutlierfactor.pkl')
    y_pred = model_nov.predict(data_nov[list(c for c in data_nov.columns if c != "_y" and c != "Unnamed: 0")])
    y_pred[y_pred == -1] = 0
    return y_pred


def randomforestclassifier(df_base_file, CONFIG):
    """End-to-end RandomForestClassifier model to obtain the final prediction. """
    df_pred_single = create_prediction_file_1(df_base_file, CONFIG)
    df_pred_part_5 = create_prediction_file_5(df_base_file, CONFIG)
    data_clf = extract_features_from_raw_data(df_base_file=df_base_file, df_pred_single=df_pred_single, df_pred_part_5=df_pred_part_5, CONFIG=CONFIG, do_clf='MLPClassifier')
    model_clf = joblib.load(f'models//ML_models//{CONFIG}//randomforestclassifier.pkl')
    y_pred = model_clf.predict(data_clf[list(c for c in data_clf.columns if c != "_y" and c != "Unnamed: 0")])
    y_pred[y_pred == -1] = 0
    return y_pred


def multilayerperceptron(df_base_file, CONFIG):
    """End-to-end MultiLayerPerceptron model to obtain the final prediction. """
    df_pred_single = create_prediction_file_1(df_base_file, CONFIG)
    df_pred_part_5 = create_prediction_file_5(df_base_file, CONFIG)
    data_clf = extract_features_from_raw_data(df_base_file=df_base_file, df_pred_single=df_pred_single, df_pred_part_5=df_pred_part_5, CONFIG=CONFIG, do_clf='MLPClassifier')
    model_clf = joblib.load(f'models//ML_models//{CONFIG}//multilayerperceptron.pkl')
    y_pred = model_clf.predict(data_clf[list(c for c in data_clf.columns if c != "_y" and c != "Unnamed: 0")])
    y_pred[y_pred == -1] = 0
    return y_pred


def find_sid(df):
    """Finding charging sessions ID. """
    sids = []
    for sid in df["session_id"].drop_duplicates()[:]:
        df_sid = df[df["session_id"] == sid]
        if df_sid.shape[0] < LEN_SESSIONS:
            continue
        sids.append(sid)
    return np.array(sids)


def data_processing_after(generated_data, len_session, sids, scaler, df):
    """Data processing after obtaining from generator. """
    df_data = []
    for i in range(len(generated_data)):
        num = random.choice(sids)
        df_sid = df[df["session_id"] == num][:len_session]
        data = scaler.inverse(np.array(generated_data.to('cpu').detach().numpy()[i]))
        df_new = pd.DataFrame(columns=['timestamp', 'session_id', 'capacity', 'capacity_connected', 'charge_speed', 'charge_speed_should', 'energy_register', 'cpID', 'is_attack'])
        df_new['timestamp'] = df_sid['timestamp']
        df_new['session_id'] = np.zeros(len_session) + i
        df_new['capacity'] = df_sid['capacity']
        df_new['capacity_connected'] = data[:, 0]
        df_new['charge_speed'] = data[:, 1]
        df_new['charge_speed_should'] = data[:, 1]
        df_new['energy_register'] = data[:, 2]
        df_new['cpID'] = df_sid['cpID']
        df_new['is_attack'] = np.zeros(len_session)
        df_data.append(df_new)
    if len(df_data) == 1:
        return df_data
    else:
        return pd.concat(df_data, ignore_index=True)


def data_processing_wo_detach_after(generated_data, len_session, sids, scaler, df):
    """Data processing after obtaining from generator. """
    df_data = []
    for i in range(len(generated_data)):
        num = random.choice(sids)
        df_sid = df[df["session_id"] == num][:len_session]
        data = scaler.inverse(np.array(generated_data.numpy()[i]))
        df_new = pd.DataFrame(columns=['timestamp', 'session_id', 'capacity', 'capacity_connected', 'charge_speed', 'charge_speed_should', 'energy_register', 'cpID', 'is_attack'])
        df_new['timestamp'] = df_sid['timestamp']
        df_new['session_id'] = np.zeros(len_session) + i
        df_new['capacity'] = df_sid['capacity']
        df_new['capacity_connected'] = data[:, 0]
        df_new['charge_speed'] = data[:, 1]
        df_new['charge_speed_should'] = data[:, 1]
        df_new['energy_register'] = data[:, 2]
        df_new['cpID'] = df_sid['cpID']
        df_new['is_attack'] = np.zeros(len_session)

        df_data.append(df_new)
    return pd.concat(df_data, ignore_index=True)


def data_processing_before(df, len_session, scaler):
    """Data processing before feeding to GAN to generate adversarial data. """
    data = []
    labels = []
    for i, sid in enumerate(df["session_id"].drop_duplicates()[:]):
        df_sid = df[df["session_id"] == sid]
        if df_sid.shape[0] < len_session:
            continue
        df_sid = df_sid[:len_session]
        if np.array_equal(df_sid["charge_speed"].values, df_sid["charge_speed_should"].values):
            labels.append(1)
        else:
            labels.append(0)
        data.append(scaler.scaler(df_sid[['capacity_connected', 'charge_speed', 'energy_register']]))
    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
