from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import  accuracy_score,confusion_matrix,cohen_kappa_score
#from sklearn.preprocessing import StandardScaler as scaler
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pandas as pd
from os.path import isfile
from tsraster.prep import set_common_index, set_df_index,set_df_mindex, image_to_series_simple, seriesToRaster, arrayToRaster, image_to_array
from tsraster import random
import pickle
import numpy as np
from xgboost import XGBRegressor , XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import sklearn.metrics as skmetrics

import matplotlib.pyplot as plt

import copy
from contextlib import redirect_stdout
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri


def zeroMasker(row):
    ''' used as apply statement for masking in elastic_YearPredictor
    :return: 0 if cell is masked out, PredRisk otherwise
    '''
    if row['mask'] == 0:
        return 0
    else:
        return row['PredRisk']

def get_data(obj, test_size=0.33,scale=False,stratify=None,groups=None):
    '''
       :param obj: path to csv or name of pandas dataframe  with yX, or list holding dataframes [y,X]
       :param test_size: percentage to hold out for testing (default 0.33)
       :param scale: should data be centered and scaled True or False (default False)
       :param stratify: should the sample be stratified by the dependent value True or False (default None)
       :param groups:  group information defining domain specific stratifications of the samples, ex pixel_id, df.index.get_level_values('index') (default None)
    
       :return: X_train, X_test, y_train, y_test splits
    '''
    
    # read in inputs
    print("input should be csv or pandas dataframe with yX, or [y,X]")
    if str(type(obj)) == "<class 'pandas.core.frame.DataFrame'>":
        df = obj
    
    elif type(obj) == list and len(obj) == 2:
        print('reading in list concat on common index, inner join')
        obj = set_common_index(obj[0], obj[1])
        df = pd.concat([obj[0],obj[1]],axis=1, join='inner') # join Y and X
        df = df.iloc[:,~df.columns.duplicated()]  # remove any repeated columns, take first
    
    elif isfile(obj):
        df = pd.read_csv(obj)
        try:
            set_df_index(df)
        except:
            set_df_mindex(df)
    else:
        print("input format not dataframe, csv, or list")
    
    # remove potential index columns in data 
    df = df.drop(['Unnamed: 0','pixel_id','time'], axis=1,errors ='ignore')  # clear out unknown columns
    
    # check if center and scale
    if scale == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(np_scaled)
    
    y = df.iloc[:,0]
    X = df.iloc[:,1:]
    
    # handle stratification by dependent variable
    if stratify == True:
        stratify = y
    else:
        stratify = None
    
    if groups is not None: 
        print('ERROR: need to figure out groups with stratification by y')
        # test train accounting for independent groups
        train_inds, test_inds = next(GroupShuffleSplit().split(X, groups=groups)) 
        X_train, X_test, y_train, y_test = X.iloc[train_inds,:], X.iloc[test_inds,:], y.iloc[train_inds], y.iloc[test_inds]
        
    else:
        # ungrouped test train split 
        X_train, X_test, y_train, y_test = tts(X, y,
                                               test_size=test_size,
                                               stratify=stratify,
                                               random_state=42)
    
    return X_train, X_test, y_train, y_test

def dataFrame_to_r(in_frame):
    #convert pandas dataframe to r dataframe
    base = importr('base')

    from rpy2.robjects import pandas2ri
    pandas2ri.activate()


    #Convert pandas to r
    r_in_frame = pandas2ri.py2ri(in_frame)
    

    #calling function under base package
    #print(base.summary(r_combined_Data))
    #print(type(r_combined_Data))
    #print(r_combined_Data)
    return r_in_frame


def R_GAM_YearPredictor_Class_Regional(combined_Data_Training, target_Data_Training, 
                              preMasked_Data_Path, outPath, year_List, periodLen, 
                              DataFields, mask,
                              splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                              familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                              region = None, # if not none, iterate across all values of region, to calculate regional models
                              k = 5): # number of smooths to allow per parameter - -1 for no maximum
  if region != None:
    regionList = unique(combined_Data_Training[region])

    for x in regionList:
      combined_Data_Regional = combined_Data_Training[combined_Data_Training[region] == x]
      target_Data_Regional = target_Data_Training[combined_Data_Training[region] == x]

      R_GAM_YearPredictor_Class(combined_Data_Training = combined_Data_Regional, target_Data_Training = target_Data_Regional, 
                                preMasked_Data_Path = preMasked_Data_Path,
                                outPath = outPath + "Region_" + str(x) + '_',
                                year_List = year_List, 
                                periodLen = periodLen, 
                                DataFields = DataFields, 
                                mask = mask,
                                splineType = splineType, # list for creating space for identifing optimal wifggliness penalization:
                                familyType = familyType, #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                                k = k )




def R_GAM_YearPredictor_Class(combined_Data_Training, target_Data_Training, 
                                preMasked_Data_Path, outPath, year_List, periodLen, 
                                DataFields, mask,
                               splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                               familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                                k = -1): # number of smooths to allow per parameter - -1 for no maximum)
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param splineType: type of splines to use - default is cs, which is cubic with shrinkage
    :param familyType: type of GAM to use: default binomial
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    
    model_List = []
    
    
    
    #set up formla for model
    iter_formula = 'value~'
    for iterparam in DataFields:
        iter_formula += " + s(" + iterparam + ', k = ' + str(k) + ',  bs = \"' + splineType + '"'  ')'
    
    
    for iterYear in year_List:
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        combined_Data_iter_train['value'] = target_Data_iter_train['value']
        
        
        r_combined_Data_iter_train = dataFrame_to_r(combined_Data_iter_train)
        
        
       
        
        #run model 
        model = mgcv.gam(formula = base.eval(base.parse(text=iter_formula)),family = base.eval(base.parse(text=familyType)), data = r_combined_Data_iter_train)
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X = full_X.loc[:, DataFields]
        full_X = dataFrame_to_r(full_X)

        
        predict_proba = stats.predict(model,full_X, type = 'response')
        predict_proba = np.array(predict_proba)
        


        print(predict_proba.shape)
        # data_risk[1]represents predictedprobability  risk of fire, 
        #data_risk[2] represents probability of no fire
        data = pd.DataFrame(predict_proba, columns = ['PredRisk'])  
        index_mask = image_to_series_simple(mask)             ###########
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredRisk_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "PredRisk_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "LogGam_Class.tif")
 
def R_Gam_YearPredictor_regional(combined_Data_Training, target_Data_Training, 
                              preMasked_Data_Path, outPath, year_List, periodLen, 
                              DataFields, mask,
                              splineType = 'cs', 
                              familyType = "binomial",
                              region = 'Region',
                              null_regions = []):
    
    regionList = pd.unique(combined_Data_Training['Region']).tolist()
    regionList = list(set(regionList) - set(null_regions))

    
    
    for x in regionList:
   
        combined_Data_Regional = combined_Data_Training[combined_Data_Training[region] == x]
        target_Data_Regional = target_Data_Training[combined_Data_Training[region] == x]
        print('Region = ', x)
        print('length = ', len(combined_Data_Regional))
        try:
          R_GAM_YearPredictor_Class(combined_Data_Regional, target_Data_Regional, 
                                  preMasked_Data_Path, outPath + "Region_" + str(x) + '_',
                                  year_List, periodLen, 
                                  DataFields, mask,
                                  splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                                  familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values,
          k = 5)
                            
        except:
              try:
                 R_GAM_YearPredictor_Class(combined_Data_Regional, target_Data_Regional,
                                  preMasked_Data_Path, outPath + "Region_" + str(x) + '_',
                                  year_List, periodLen,
                                  DataFields,
                                  mask,
                                  splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                                  familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values,
                                  k = 5)
              except:
                   pass

        

def predict_Test(model, testData, threshold, suffix = ''):
    '''iterates predictions across pixels_years, pixels, and years 
    to conduct model diagnostics across different forms of cross-validation:
    is called by R_GAM in 2dim cross validation
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    r_testData = dataFrame_to_r(testData)
    
    predict_proba = stats.predict(model,r_testData, type = 'response')
    predict_risk = np.array(predict_proba)
    predict_test = copy.deepcopy(predict_risk)
    
    predict_test = np.where(predict_test >threshold, 1.0, predict_test)
    predict_test = np.where(predict_test <=threshold, 0.0, predict_test)
    
    
    y_test = np.array(testData['value'])
    
    r_summary = base.summary(model)
    
    r2 = np.array(r_summary.rx('r.sq')[0])[0]
    se = np.array(r_summary.rx('se')[0])[0]
    
    Accuracy = skmetrics.accuracy_score(y_test, predict_test)
    BalancedAccuracy = skmetrics.balanced_accuracy_score(y_test, predict_test)
    f1_binary = skmetrics.f1_score(y_test, predict_test, average = 'binary')
    f1_macro = skmetrics.f1_score(y_test, predict_test, average = 'macro')
    f1_micro = skmetrics.f1_score(y_test, predict_test, average = 'micro')
    log_loss = skmetrics.log_loss(y_test, predict_test, labels = [0,1])
    recall_binary = skmetrics.recall_score(y_test, predict_test, average = 'binary')
    recall_macro = skmetrics.recall_score(y_test, predict_test, average = 'macro')
    recall_micro = skmetrics.recall_score(y_test, predict_test, average = 'micro')
    jaccard_binary = skmetrics.jaccard_score(y_test, predict_test,average = 'binary')
    jaccard_macro = skmetrics.jaccard_score(y_test, predict_test, average = 'macro')
    jaccard_micro = skmetrics.jaccard_score(y_test, predict_test, average = 'micro')
    roc_auc_macro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'macro')
    roc_auc_micro = skmetrics.roc_auc_score(y_test, predict_risk, average = 'micro')
    fpr, tpr, thresholds = skmetrics.roc_curve(y_test, predict_risk, drop_intermediate = False)
    brier_actual = skmetrics.brier_score_loss(y_test, predict_risk)
    print("Data length (y_test) = ", len(y_test))
    print("Data length (predict_risk)) = ", len(predict_risk))
    print("fpr_raw_iter = ", len(fpr))
    print("tpr_raw_iter = ", len(fpr))
    print(suffix)
    print()




    mean_risk = [y_test.mean() for _ in range(len(y_test))]
    brier_baseline = skmetrics.brier_score_loss(y_test, mean_risk)
    
    all_0 = [0 for _ in range(len(y_test))]
    brier_0 = skmetrics.brier_score_loss(y_test,all_0)

    all_1 = [1 for _ in range(len(y_test))]
    brier_1 = skmetrics.brier_score_loss(y_test, all_1)

    output_dict = {"r2_"+ suffix:r2,
                   "BalancedAccuracy_"+ suffix: BalancedAccuracy,
                   "f1_binary_"+ suffix: f1_binary, 
                   "f1_macro_"+ suffix: f1_macro, "f1_micro_"+ suffix: f1_micro,
                   "log_loss_"+ suffix: log_loss, 
                   "recall_binary_"+ suffix: recall_binary, "recall_macro_"+ suffix: recall_macro, "recall_micro_"+ suffix: recall_micro,
                   "jaccard_binary_"+ suffix:jaccard_binary, "jaccard_macro_"+ suffix: jaccard_macro, "jaccard_micro_"+ suffix: jaccard_micro,
                   "roc_auc_macro_"+ suffix: roc_auc_macro, "roc_auc_micro_"+ suffix: roc_auc_micro, 
                    "fpr_"+ suffix: copy.deepcopy(fpr), "tpr_"+ suffix: copy.deepcopy(tpr), "thresh_"+ suffix: copy.deepcopy(thresholds), 
                   'brier_actual_' + suffix:brier_actual, 'brier_baseline_' + suffix : brier_baseline, 'brier_1_' + suffix : brier_1, 'brier_0_' + suffix : brier_0}
    


    return output_dict

def R_GAM(X_train, y_train, X_test_a, y_test_a, 
          X_test_b, y_test_b, 
          X_test_a_b, y_test_a_b, 
          formula, familyType, threshold = 0.5,
         groupVars = ['a', 'b']):
    '''
    Conduct elastic net regression on training data and test predictive power against test data

    :param X_train: pandas dataframe containing training data features
    :param y_train: pandas dataframe containing training data responses
    :param X_test: pandas dataframe containing training data features for testing
    :param y_test: pandas dataframe containing training data responses for testing
    :param iterFormula: formula to be entered into r model
    :param threshold:  threshold above which prediction will be set to 1, below which it will be set to 0
         :return: dictionary of model object and diagnostic parameters
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    
    
   
    
    #combine value into X train and test data, convert to r dataFrames
    X_train['value'] = y_train['value']
    r_train = dataFrame_to_r(X_train)
    
    
    
    X_test_a['value'] = y_test_a['value']
    #r_test_pixels = dataFrame_to_r(X_test_pixels)
    
    X_test_b['value'] = y_test_b['value']
    #r_test_years = dataFrame_to_r(X_test_years)
    
    X_test_a_b['value'] = y_test_a_b['value']
    #r_test_pixels_years = dataFrame_to_r(X_test_pixels_years)
    


    model = mgcv.gam(formula = base.eval(base.parse(text=formula)),family = base.eval(base.parse(text=familyType)), data = r_train)

    blockDict = {groupVars[0]: X_test_a, groupVars[1]: X_test_b, groupVars[0] +'_' +  groupVars[1]: X_test_a_b}
    
    out_stats = {'blank': ''}
    for i in blockDict:
        iterData = blockDict[i]
        suffix = "_" + i
        iterDict = predict_Test(model, iterData, threshold, suffix = i)
        out_stats.update(iterDict)
    
    
    output_dict = {"model": model, 
                   "out_stats": out_stats}
    
    return output_dict



def R_logGAM_2dimTest(combined_Data, target_Data, varsToGroupBy, groupVars, testGroups, 
                        DataFields, outPath,
                        preset_GroupVar = None,
                        splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                        familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                        k = -1):
    '''Conduct logistic regressions on the data, with k-fold cross-validation conducted independently 
        across both years and pixels. 
        Returns a variety of diagnostics of model performance (including f1 scores, recall, and average precision) 
        when predicting fire risk at 
        A) locations outside of the training dataset
        B) years outside of the training dataset
        C) locations and years outside of the training dataset

      Returns a list of objects, consisting of:
        0: Combined_Data file with testing/training groups labeled
        1: Target Data file with testing/training groups labeled
        2: summary dataFrame of MSE and R2 for each model run
            (against holdout data representing either novel locations, novel years, or both)
        3: list of elastic net models for use in predicting Fires in further locations/years
        4: list of list of years not used in model training for each run
    :param combined_Data: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param varsToGroupBy: list of (2) column names from combined_Data & target_Data to be used in creating randomized groups
    :param groupVars: list of (2) desired column names for the resulting randomized groups
    :param testGroups: number of distinct groups into which data sets should be divided (for each of two variables) 
    :param Datafields: list of explanatory factors to be intered into model
    :param preset_GroupVar: if using region variable, or other grouping variable for which groups are already set, should be string of column name.  Default = None.  If str, replaces pixel_id
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param max_smoothing: maximum amount of smoothing to conduct in GAMs
    :param penalty_space: space to scan for amount of L2 penalization terms
        -list consisting of min penalization, max penalization, number of points within that range to test
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    
    combined_Data, target_Data, varsToGroupBy, groupVars, testGroups = random.TestTrain_GroupMaker(combined_Data, target_Data, 
                                                             varsToGroupBy, 
                                                             groupVars, 
                                                             testGroups, 
                                                             preset_GroupVar = preset_GroupVar)
    print("varsToGroupBy:", varsToGroupBy)
    print('groupVars: ', groupVars)
    print('testGroups: ', testGroups)
    combined_Data.to_csv(outPath + 'regionalCombined_DataTest.csv')
    target_Data.to_csv(outPath + 'regionaltarget_DataTest.csv')

    #get list of group ids, since in cases where group # <10, may not begin at zero
    a_testVals = list(set(combined_Data[groupVars[0]].tolist()))
    b_testVals = list(set(combined_Data[groupVars[1]].tolist()))

    Models_Summary = pd.DataFrame([], columns = [])

    #used to create list of model runs
    models = []
    
    
  
  #used to create data for entry as columns into summary DataFrame
    
    #blank list to be filled with the a & b groups used to test in each iteration
    a_test= []
    b_test = []
    
    
    a_b_r2List = []
    a_r2List = []
    b_r2List = []
    
    a_b_BalancedAccuracyList = []
    a_BalancedAccuracyList = []
    b_BalancedAccuracyList = []
    
    a_b_F1_binaryList = []
    a_F1_binaryList = []
    b_F1_binaryList = []
    
    a_b_F1_MacroList = []
    a_F1_MacroList = []
    b_F1_MacroList = []
    
    a_b_F1_MicroList = []
    a_F1_MicroList = []
    b_F1_MicroList = []
    
    a_b_logLossList = []
    a_logLossList = []
    b_logLossList = []
    
    a_b_recall_binaryList = []
    a_recall_binaryList = []
    b_recall_binaryList = []
    
    a_b_recall_MacroList = []
    a_recall_MacroList = []
    b_recall_MacroList = []
    
    a_b_recall_MicroList = []
    a_recall_MicroList = []
    b_recall_MicroList = []
    
    a_b_jaccard_binaryList = []
    a_jaccard_binaryList = []
    b_jaccard_binaryList = []
    
    a_b_jaccard_MacroList = []
    a_jaccard_MacroList = []
    b_jaccard_MacroList = []
    
    a_b_jaccard_MicroList = []
    a_jaccard_MicroList = []
    b_jaccard_MicroList = []
    
    a_b_roc_auc_MacroList = []
    a_roc_auc_MacroList = []
    b_roc_auc_MacroList = []
    
    a_b_roc_auc_MicroList = []
    a_roc_auc_MicroList = []
    b_roc_auc_MicroList = []
    
    a_b_average_precisionList = []
    a_average_precisionList = []
    b_average_precisionList = []

    a_b_brier_baseline_List = []
    a_b_brier_actual_List = []
    a_b_brier_0_List = []
    a_b_brier_1_List = []

    a_brier_baseline_List = []
    a_brier_actual_List = []
    a_brier_0_List = []
    a_brier_1_List = []

    b_brier_baseline_List = []
    b_brier_actual_List = []
    b_brier_0_List = []
    b_brier_1_List = []

    roc_fpr_a = []
    roc_tpr_a = []
    roc_thresh_a = []

    roc_fpr_b = []
    roc_tpr_b = []
    roc_thresh_b = []

    roc_fpr_a_b = []
    roc_tpr_a_b = []
    roc_thresh_a_b = []

    iterCounter = 0
    
  #used to create a list of lists of years that are excluded within each model run
    excluded_b = []
    excluded_a = []

    #set up formula for use in model
    formula = 'value~'
    for iterparam in DataFields:
        formula += " + s(" + iterparam + ', k = ' + str(k) + ',  bs = \"' + splineType + '"'  ')'
    
    
    for x in a_testVals:


        for y in b_testVals:

            iterCounter +=1


            trainData_X = combined_Data[combined_Data[groupVars[0]] != x]
            trainData_X = trainData_X[trainData_X[groupVars[1]] != y]
            trainData_X = trainData_X.loc[:, DataFields]


            trainData_y = target_Data[target_Data[groupVars[0]] != x]
            trainData_y = trainData_y[trainData_y[groupVars[1]] != y]


            testData_X_a_b = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_a_b = testData_X_a_b[testData_X_a_b[groupVars[1]] == y]
            testData_X_a_b = testData_X_a_b.loc[:, DataFields]

            testData_X_a = combined_Data[combined_Data[groupVars[0]] == x]
            testData_X_a = testData_X_a[testData_X_a[groupVars[1]] != y]
            testData_X_a = testData_X_a.loc[:, DataFields]

            testData_X_b = combined_Data[combined_Data[groupVars[0]] != x]
            testData_X_b = testData_X_b[testData_X_b[groupVars[1]] == y]
            testData_X_b = testData_X_b.loc[:, DataFields]



            testData_y_a_b = target_Data[target_Data[groupVars[0]] == x]
            testData_y_a_b = testData_y_a_b[testData_y_a_b[groupVars[1]] == y]


            testData_y_a = target_Data[target_Data[groupVars[0]] == x]
            testData_y_a = testData_y_a[testData_y_a[groupVars[1]] != y]
            excluded_a.append(list(set(testData_y_a[varsToGroupBy[0]].tolist())))


            testData_y_b = target_Data[target_Data[groupVars[0]] != x]
            testData_y_b = testData_y_b[testData_y_b[groupVars[1]] == y]
            excluded_b.append(list(set(testData_y_b[varsToGroupBy[1]].tolist())))
            
            
           
        
            
            
            iterOutput = R_GAM(X_train = trainData_X, y_train = trainData_y, 
                      X_test_a = testData_X_a, y_test_a = testData_y_a, 
                      X_test_b = testData_X_b, y_test_b = testData_y_b, 
                      X_test_a_b = testData_X_a_b, y_test_a_b = testData_y_a_b, 
                      formula = formula,
                      familyType = familyType, threshold = 0.5, groupVars = groupVars)
            
            iter_stats = iterOutput['out_stats'] #get dictionary of stats summaries

            
            a_test.append(str(x))
            b_test.append(str(y))
            
            a_b_r2List.append(iter_stats['r2_' + groupVars[0] +  '_' + groupVars[1]])
            a_r2List.append(iter_stats['r2_' + groupVars[0]])
            b_r2List.append(iter_stats['r2_' + groupVars[1]])

            a_b_BalancedAccuracyList.append(iter_stats['BalancedAccuracy_' + groupVars[0] +  '_' + groupVars[1]])
            a_BalancedAccuracyList.append(iter_stats['BalancedAccuracy_' + groupVars[0]])
            b_BalancedAccuracyList.append(iter_stats['BalancedAccuracy_'+ groupVars[1]])
            
            a_b_F1_binaryList.append(iter_stats['f1_binary_' + groupVars[0] +  '_' + groupVars[1]])
            a_F1_binaryList.append(iter_stats['f1_binary_' + groupVars[0]])
            b_F1_binaryList.append(iter_stats['f1_binary_'+ groupVars[1]])
            
            a_b_F1_MacroList.append(iter_stats['f1_macro_' + groupVars[0] +  '_' + groupVars[1]])
            a_F1_MacroList.append(iter_stats['f1_macro_' + groupVars[0]])
            b_F1_MacroList.append(iter_stats['f1_macro_'+ groupVars[1]])
            
            a_b_F1_MicroList.append(iter_stats['f1_micro_' + groupVars[0] +  '_' + groupVars[1]])
            a_F1_MicroList.append(iter_stats['f1_micro_' + groupVars[0]])
            b_F1_MicroList.append(iter_stats['f1_micro_'+ groupVars[1]])
            
            a_b_logLossList.append(iter_stats['log_loss_' + groupVars[0] +  '_' + groupVars[1]])
            a_logLossList.append(iter_stats['log_loss_' + groupVars[0]])
            b_logLossList.append(iter_stats['log_loss_'+ groupVars[1]])
            
            a_b_recall_binaryList.append(iter_stats['recall_binary_' + groupVars[0] +  '_' + groupVars[1]])
            a_recall_binaryList.append(iter_stats['recall_binary_' + groupVars[0]])
            b_recall_binaryList.append(iter_stats['recall_binary_'+ groupVars[1]])
            
            a_b_recall_MacroList.append(iter_stats['recall_macro_' + groupVars[0] +  '_' + groupVars[1]])
            a_recall_MacroList.append(iter_stats['recall_macro_' + groupVars[0]])
            b_recall_MacroList.append(iter_stats['recall_macro_'+ groupVars[1]])
            
            a_b_recall_MicroList.append(iter_stats['recall_micro_' + groupVars[0] +  '_' + groupVars[1]])
            a_recall_MicroList.append(iter_stats['recall_micro_' + groupVars[0]])
            b_recall_MicroList.append(iter_stats['recall_micro_'+ groupVars[1]])
            
            a_b_jaccard_binaryList.append(iter_stats['jaccard_binary_' + groupVars[0] +  '_' + groupVars[1]])
            a_jaccard_binaryList.append(iter_stats['jaccard_binary_' + groupVars[0]])
            b_jaccard_binaryList.append(iter_stats['jaccard_binary_'+ groupVars[1]])
            
            a_b_jaccard_MacroList.append(iter_stats['jaccard_macro_' + groupVars[0] +  '_' + groupVars[1]])
            a_jaccard_MacroList.append(iter_stats['jaccard_macro_' + groupVars[0]])
            b_jaccard_MacroList.append(iter_stats['jaccard_macro_'+ groupVars[1]])
            
            a_b_jaccard_MicroList.append(iter_stats['jaccard_micro_' + groupVars[0] +  '_' + groupVars[1]])
            a_jaccard_MicroList.append(iter_stats['jaccard_micro_' + groupVars[0]])
            b_jaccard_MicroList.append(iter_stats['jaccard_micro_'+ groupVars[1]])
            
            a_b_roc_auc_MacroList.append(iter_stats['roc_auc_macro_' + groupVars[0] +  '_' + groupVars[1]])
            a_roc_auc_MacroList.append(iter_stats['roc_auc_macro_' + groupVars[0]])
            b_roc_auc_MacroList.append(iter_stats['roc_auc_macro_'+ groupVars[1]])
            
            a_b_roc_auc_MicroList.append(iter_stats['roc_auc_micro_' + groupVars[0] +  '_' + groupVars[1]])
            a_roc_auc_MicroList.append(iter_stats['roc_auc_micro_' + groupVars[0]])
            b_roc_auc_MicroList.append(iter_stats['roc_auc_micro_'+ groupVars[1]])
            
            #a_b_average_precisionList.append(iter_stats['average_precision_' + groupVars[0] +  '_' + groupVars[1]])
            #a_average_precisionList.append(iter_stats['average_precision_' + groupVars[0]])
            #b_average_precisionList.append(iter_stats['average_precision_'+ groupVars[1]])
            
            a_b_brier_actual_List.append(iter_stats['brier_actual_' + groupVars[0] +  '_' + groupVars[1]])
            a_brier_actual_List.append(iter_stats['brier_actual_' + groupVars[1]])
            b_brier_actual_List.append(iter_stats['brier_actual_' + groupVars[0]])

            a_b_brier_baseline_List.append(iter_stats['brier_baseline_' + groupVars[0] +  '_' + groupVars[1]])
            a_brier_baseline_List.append(iter_stats['brier_baseline_' + groupVars[1]])
            b_brier_baseline_List.append(iter_stats['brier_baseline_' + groupVars[0]])

            a_b_brier_0_List.append(iter_stats['brier_0_' + groupVars[0] +  '_' + groupVars[1]])
            a_brier_0_List.append(iter_stats['brier_0_' + groupVars[1]])
            b_brier_0_List.append(iter_stats['brier_0_' + groupVars[0]])

            a_b_brier_1_List.append(iter_stats['brier_1_' + groupVars[0] +  '_' + groupVars[1]])
            a_brier_1_List.append(iter_stats['brier_1_' + groupVars[1]])
            b_brier_1_List.append(iter_stats['brier_1_' + groupVars[0]])


            models.append(iterOutput['model'])

            print("iterCounter: ", iterCounter)
            print("fpr_a_iter_len = ", len(iter_stats['fpr_' + groupVars[0]].tolist()))
            print("tpr_a_iter_len = ", len(iter_stats['tpr_' + groupVars[0]].tolist()))
            print("fpr_b_iter_len = ", len(iter_stats['fpr_' + groupVars[1]].tolist()))
            print("tpr_b_iter_len = ", len(iter_stats['tpr_' + groupVars[1]].tolist()))
            print("fpr_a_b_iter_len = ", len(iter_stats['fpr_'  + groupVars[0] +  '_' + groupVars[1]]))
            print("tpr_a_b_iter_len = ", len(iter_stats['tpr_'  + groupVars[0] +  '_' + groupVars[1]]))

           
            roc_fpr_a.append(iter_stats['fpr_' + groupVars[0]].tolist())
            roc_tpr_a.append(copy.deepcopy(iter_stats['tpr_' + groupVars[0]].tolist()))
            roc_thresh_a.append(copy.deepcopy(iter_stats['thresh_' + groupVars[0]].tolist()))

            roc_fpr_b.append(copy.deepcopy(iter_stats['fpr_' + groupVars[1]].tolist()))
            roc_tpr_b.append(copy.deepcopy(iter_stats['tpr_' + groupVars[1]].tolist()))
            roc_thresh_b.append(copy.deepcopy(iter_stats['thresh_' + groupVars[1]].tolist()))

            roc_fpr_a_b.append(copy.deepcopy(iter_stats['fpr_' + groupVars[0]+  '_' + groupVars[1]].tolist()))
            roc_tpr_a_b.append(copy.deepcopy(iter_stats['tpr_' + groupVars[0]+  '_' + groupVars[1]].tolist()))
            roc_thresh_a_b.append(copy.deepcopy(iter_stats['thresh_' + groupVars[0]+  '_' + groupVars[1]].tolist()))
            


          
    '''            
    #create marginal response graphs - uses entire dataset
    
    penalty_space_forGAM = np.random.uniform(penalty_space[0], penalty_space[1], (penalty_space[2],len(DataFields)))
    
    gam = LogisticGAM(n_splines = max_smoothing).gridsearch(combined_Data[DataFields].values, 
                                                            target_Data['value'], 
                                                            lam =  penalty_space_forGAM)
    
    #save model summary to text file
    with open(outPath +'modelDetails.txt', 'w') as f:
        with redirect_stdout(f):
            gam.summary()
    
    for j in range(len(DataFields)):
        plt.clf()
        
        XX = gam.generate_X_grid(term=j)#create grid of uniform terms for response variable
        plt.plot(XX[:, j], gam.partial_dependence(term = j, X=XX))
        plt.plot(XX[:, j], gam.partial_dependence(term=j, X=XX, width=.95)[1], c='r', ls='--')
        plt.title(DataFields[j])
        plt.tight_layout()
        plt.savefig(outPath + "Marginal_Response_"+ DataFields[j] + ".tif")
        
    '''
        
    #combine MSE and R2 Lists into single DataFrame
    
    #combine MSE and R2 Lists into single DataFrame
    
    
    Models_Summary[groupVars[0] + 'excluded'] = a_test
    Models_Summary[groupVars[1] + 'excluded'] = b_test
    
    Models_Summary[groupVars[0] + groupVars[1] + '_r2'] = a_b_r2List
    Models_Summary[groupVars[0] +  '_r2'] = a_r2List
    Models_Summary[groupVars[1] +  '_r2'] = b_r2List
    
    Models_Summary[groupVars[0] + groupVars[1] + '_BalancedAccuracy'] = a_b_BalancedAccuracyList
    Models_Summary[groupVars[0] +  '_BalancedAccuracy'] = a_BalancedAccuracyList
    Models_Summary[groupVars[1] +  '_BalancedAccuracy'] = b_BalancedAccuracyList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_F1_binaryList'] = a_b_F1_binaryList
    Models_Summary[groupVars[0] +  '_F1_binaryList'] = a_F1_binaryList
    Models_Summary[groupVars[1] +  '_F1_binaryList'] = b_F1_binaryList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_F1_MacroList'] = a_b_F1_MacroList
    Models_Summary[groupVars[0] +  '_F1_MacroList'] = a_F1_MacroList
    Models_Summary[groupVars[1] +  '_F1_MacroList'] = b_F1_MacroList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_F1_MicroList'] = a_b_F1_MicroList
    Models_Summary[groupVars[0] +  '_F1_MicroList'] = a_F1_MicroList
    Models_Summary[groupVars[1] +  '_F1_MicroList'] = b_F1_MicroList

    Models_Summary[groupVars[0] + groupVars[1] + '_log_loss'] = a_b_logLossList
    Models_Summary[groupVars[0] +  '_log_loss'] = a_logLossList
    Models_Summary[groupVars[1] +  '_log_loss'] = b_logLossList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_recall_binaryList'] = a_b_recall_binaryList
    Models_Summary[groupVars[0] +  '_recall_binaryList'] = a_recall_binaryList
    Models_Summary[groupVars[1] +  '_recall_binaryList'] = b_recall_binaryList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_recall_MacroList'] = a_b_recall_MacroList
    Models_Summary[groupVars[0] +  '_recall_MacroList'] = a_recall_MacroList
    Models_Summary[groupVars[1] +  '_recall_MacroList'] = b_recall_MacroList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_recall_MicroList'] = a_b_recall_MicroList
    Models_Summary[groupVars[0] +  '_recall_MicroList'] = a_recall_MicroList
    Models_Summary[groupVars[1] +  '_recall_MicroList'] = b_recall_MicroList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_jaccard_binaryList'] = a_b_jaccard_binaryList
    Models_Summary[groupVars[0] +  '_jaccard_binaryList'] = a_jaccard_binaryList
    Models_Summary[groupVars[1] +  '_jaccard_binaryList'] = b_jaccard_binaryList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_jaccard_MacroList'] = a_b_jaccard_MacroList
    Models_Summary[groupVars[0] +  '_jaccard_MacroList'] = a_jaccard_MacroList
    Models_Summary[groupVars[1] +  '_jaccard_MacroList'] = b_jaccard_MacroList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_jaccard_MicroList'] = a_b_jaccard_MicroList
    Models_Summary[groupVars[0] +  '_jaccard_MicroList'] = a_jaccard_MicroList
    Models_Summary[groupVars[1] +  '_jaccard_MicroList'] = b_jaccard_MicroList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_roc_auc_MacroList'] = a_b_roc_auc_MacroList
    Models_Summary[groupVars[0] +  '_roc_auc_MacroList'] = a_roc_auc_MacroList
    Models_Summary[groupVars[1] +  '_roc_auc_MacroList'] = b_roc_auc_MacroList
    
    Models_Summary[groupVars[0] + groupVars[1] + '_roc_auc_MicroList'] = a_b_roc_auc_MicroList
    Models_Summary[groupVars[0] +  '_roc_auc_MicroList'] = a_roc_auc_MicroList
    Models_Summary[groupVars[1] +  '_roc_auc_MicroList'] = b_roc_auc_MicroList
    
    #Models_Summary[groupVars[0] + groupVars[1] + '_average_precision'] = a_b_average_precisionList
    #Models_Summary[groupVars[0] +  '_average_precision'] = a_average_precisionList
    #Models_Summary[groupVars[1] +  '_average_precision'] = b_average_precisionList


    Models_Summary[groupVars[0] + groupVars[1] + '_brier_baseline'] = a_b_brier_baseline_List
    Models_Summary[groupVars[0] +  '_brier_baseline'] = a_brier_baseline_List
    Models_Summary[groupVars[1] +  '_brier_baseline'] = b_brier_baseline_List

    Models_Summary[groupVars[0] + groupVars[1] + '_brier_actual'] = a_b_brier_actual_List
    Models_Summary[groupVars[0] +  '_brier_actual'] = a_brier_actual_List
    Models_Summary[groupVars[1] +  '_brier_actual'] = b_brier_actual_List


    Models_Summary[groupVars[0] + groupVars[1] + '_brier_0'] = a_b_brier_0_List
    Models_Summary[groupVars[0] +  '_brier_0'] = a_brier_0_List
    Models_Summary[groupVars[1] +  '_brier_0'] = b_brier_0_List


    Models_Summary[groupVars[0] + groupVars[1] + '_brier_1'] = a_b_brier_1_List
    Models_Summary[groupVars[0] +  '_brier_1'] = a_brier_1_List
    Models_Summary[groupVars[1] +  '_brier_1'] = b_brier_1_List

    
    
    

    plt.clf()

    for roc_iter in range(0, len(roc_fpr_a)):
      plt.plot(roc_fpr_a[roc_iter], roc_tpr_a[roc_iter], color = 'blue', alpha = 0.3, linewidth = 2, label = 'ROC')
      plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          label='Null', alpha=.8)
      plt.xlabel("False Positive Rate", fontsize = '16', fontweight = 'bold')
      plt.ylabel("True Positive Rate", fontsize = '16', fontweight = 'bold')
      plt.yticks(fontsize = '12', fontweight = 'bold')
      plt.xticks(fontsize = '12', fontweight = 'bold')
      plt.tight_layout()
      plt.savefig(outPath + "ROC_Curve_"+ groupVars[0] + ".tif")


    plt.clf()

    for roc_iter in range(0, len(roc_fpr_b)):
      plt.plot(roc_fpr_b[roc_iter], roc_tpr_b[roc_iter], color = 'blue', alpha = 0.3, linewidth = 2, label = 'ROC')
      plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          label='Null', alpha=.8)
      plt.xlabel("False Positive Rate", fontsize = '16', fontweight = 'bold')
      plt.ylabel("True Positive Rate", fontsize = '16', fontweight = 'bold')
      plt.yticks(fontsize = '12', fontweight = 'bold')
      plt.xticks(fontsize = '12', fontweight = 'bold')
      plt.tight_layout()
      plt.savefig(outPath + "ROC_Curve_"+ groupVars[1] + ".tif")

    plt.clf()

    for roc_iter in range(0, len(roc_fpr_a_b)):
      plt.plot(roc_fpr_a_b[roc_iter], roc_tpr_a_b[roc_iter], color = 'blue', alpha = 0.3, linewidth = 2, label = 'ROC')
      plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          label='Null', alpha=.8)
      plt.xlabel("False Positive Rate", fontsize = '16', fontweight = 'bold')
      plt.ylabel("True Positive Rate", fontsize = '16', fontweight = 'bold')
      plt.yticks(fontsize = '12', fontweight = 'bold')
      plt.xticks(fontsize = '12', fontweight = 'bold')
      plt.tight_layout()
      plt.savefig(outPath + "ROC_Curve_"+ groupVars[0] + groupVars[1] + ".tif")

    pickling_on = open(outPath + "logGAM_2dim.pickle", "wb")
    #pickle.dump([combined_Data, target_Data, Models_Summary, Models, excluded_Years], pickling_on)
    
    pickle.dump([combined_Data.loc[:, [groupVars[0], groupVars[1]]],  models, excluded_a, excluded_b, excluded_b], pickling_on)
    pickling_on.close

    Models_Summary.to_csv(outPath + "Model_Summary_logGAM.csv")

    #return combined_Data, target_Data, Models_Summary, Models, excluded_Years

    return combined_Data, target_Data, Models_Summary, models, excluded_a, excluded_b




'''
def R_Gam_Summary(combined_Data, target_Data,
                        DataFields, outPath,
                        fullDataPath = None,
                        exampleRasterPath = None,
                        splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                        familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                        k = -1):
    Conduct logistic regressions on the data, with k-fold cross-validation conducted independently 
        across both years and pixels. 
        Returns a variety of diagnostics of model performance (including f1 scores, recall, and average precision) 
        when predicting fire risk at 
        A) locations outside of the training dataset
        B) years outside of the training dataset
        C) locations and years outside of the training dataset

      Returns a list of objects, consisting of:
        0: Combined_Data file with testing/training groups labeled
        1: Target Data file with testing/training groups labeled
        2: summary dataFrame of MSE and R2 for each model run
            (against holdout data representing either novel locations, novel years, or both)
        3: list of elastic net models for use in predicting Fires in further locations/years
        4: list of list of years not used in model training for each run
    :param combined_Data: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param Datafields: list of explanatory factors to be intered into model
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    formula = 'value~'
    for iterparam in DataFields:
        formula += " + s(" + iterparam + ', k = ' + str(k) + ',  bs = \"' + splineType + '"'  ')'
    
    
    combined_Data['value'] = target_Data['value']
    r_train = dataFrame_to_r(combined_Data)
    
    model = mgcv.gam(formula = base.eval(base.parse(text=formula)),family = base.eval(base.parse(text=familyType)), data = r_train)
    
    
    #print model summary as text file
    summary =str(base.summary(model))
    #print(type(summary))
    text_file = open(outPath + "summary.txt", "w")
    n = text_file.write(summary)
    text_file.close()
    
    
    #get plot output and convert it into graphs
    rplot = robjects.r('plot.gam')
    q = rplot(model)
    
    each set will be one item in the list - 
         within each set, the items will then be ordered as follows: 
    0: x values corresponding to fit & se
    1: True/False value pertaining to scale (ignore)
    2: standard error (2* se)
    3: raw (for all pixels)
    4: string of X label
    5: string of y label
    6: ignore
    7: multiplier of SE (ignore, defaults to 2)
    8: x limits (ignore?)
    9: fit (actual values to use)
    10: True/False value indicating whether it should be plotted (ignore)

    
    for i in range(0,len(q)):
        #marginal response graph
        x_Data=np.asarray(q[i][0])
        y_Data=np.asarray(q[i][9]).flatten()
        se_Data = np.asarray(q[i][2])
        upper_Data = np.add(y_Data, se_Data)
        lower_Data = np.subtract(y_Data, se_Data)
        print(x_Data.shape)
        print(y_Data.shape)
        print(se_Data.shape)
        print(upper_Data.shape)
        x_label = str(q[i][4])
        x_label= x_label[5:-2]
        y_label = str(q[i][5])
        y_label= y_label[5:-2]
        plt.clf()

        plt.plot(x_Data, y_Data, color = 'red', linewidth = 2)
        plt.plot(x_Data, upper_Data, linewidth = 2, color = 'blue', alpha = 0.5, linestyle = "--")
        plt.plot(x_Data, lower_Data, linewidth = 2, color = 'blue', alpha = 0.5, linestyle = "--")
        plt.xlabel(x_label, fontsize = '16', fontweight = 'bold')
        plt.ylabel(y_label, fontsize = '16', fontweight = 'bold')
        plt.yticks(fontsize = '12', fontweight = 'bold')
        plt.xticks(fontsize = '12', fontweight = 'bold')
        plt.tight_layout()
        plt.savefig(outPath + "Marginal_Response_"+ DataFields[i] + ".tif")
    
    #Marginal Response Map
    
    fullData = pd.read_csv(fullDataPath)
    #fullData = fullData.loc[:, DataFields]
    
    for j in DataFields:

        iter_fullData = copy.deepcopy(fullData.loc[:, [j]])
        other_fields = copy.deepcopy(DataFields)
        other_fields.remove(j)
        for k in other_fields:
            print(k)
            iter_fullData[k] = fullData[k].mean()
        r_full = dataFrame_to_r(iter_fullData)
        fullTest = stats.predict(model,r_full, type = 'response')
        #print(type(fullTest))
        #print(fullTest)
        #fullTest = pandas2ri.ri2py_dataframe(fullTest)
        #fullTest.to_csv(outPath + "testArray_" + j + ".csv", fullTest, delimiter = ",")
        fullTest = np.asarray(fullTest)
        
        arrayToRaster(fullTest, templateRasterPath = exampleRasterPath, outPath = outPath+ "Marginal_Map_"+ j + ".tif")
'''

def R_Gam_Summary(combined_Data, target_Data,
                        DataFields, outPath,
                        fullDataPath = None,
                        exampleRasterPath = None,
                        splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                        familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                        parameterNames = None):
    '''Conduct logistic regressions on the data, with k-fold cross-validation conducted independently 
        across both years and pixels. 
        Returns a variety of diagnostics of model performance (including f1 scores, recall, and average precision) 
        when predicting fire risk at 
        A) locations outside of the training dataset
        B) years outside of the training dataset
        C) locations and years outside of the training dataset

      Returns a list of objects, consisting of:
        0: Combined_Data file with testing/training groups labeled
        1: Target Data file with testing/training groups labeled
        2: summary dataFrame of MSE and R2 for each model run
            (against holdout data representing either novel locations, novel years, or both)
        3: list of elastic net models for use in predicting Fires in further locations/years
        4: list of list of years not used in model training for each run
    :param combined_Data: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param Datafields: list of explanatory factors to be intered into model
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    formula = 'value~'
    for iterparam in DataFields:
        formula += " + s(" + iterparam + ', k = 5,  bs = \"' + splineType + '"'  ')'
    
    
    combined_Data['value'] = target_Data['value']
    r_train = dataFrame_to_r(combined_Data)
    
    model = mgcv.gam(formula = base.eval(base.parse(text=formula)), family = base.eval(base.parse(text=familyType)), data = r_train)
    
    
    #print model summary as text file
    summary =str(base.summary(model))
    print(summary)
    text_file = open(outPath + "summary.txt", "w")
    n = text_file.write(summary)
    text_file.close()
    
    
    #get plot output and convert it into graphs
    rplot = robjects.r('plot.gam')
    q = rplot(model)
    
    '''each set will be one item in the list - 
         within each set, the items will then be ordered as follows: 
    0: x values corresponding to fit & se
    1: True/False value pertaining to scale (ignore)
    2: standard error (2* se)
    3: raw (for all pixels)
    4: string of X label
    5: string of y label
    6: ignore
    7: multiplier of SE (ignore, defaults to 2)
    8: x limits (ignore?)
    9: fit (actual values to use)
    10: True/False value indicating whether it should be plotted (ignore)
'''
    
    for i in range(0,len(q)):
        #marginal response graph
        x_Data=np.asarray(q[i][0])
        y_Data=np.asarray(q[i][9]).flatten()
        se_Data = np.asarray(q[i][2])
        upper_Data = np.add(y_Data, se_Data)
        lower_Data = np.subtract(y_Data, se_Data)
        print(x_Data.shape)
        print(y_Data.shape)
        print(se_Data.shape)
        print(upper_Data.shape)


        x_label = str(q[i][4])
        x_label = x_label[5:-2]
        #y_label = str(q[i][5])
        #y_label= y_label[5:-2]
        
        if type(parameterNames) == dict:
          
          x_label = parameterNames[x_label] 
          
        plt.clf()

        plt.plot(x_Data, y_Data, color = 'red', linewidth = 2)
        plt.plot(x_Data, upper_Data, linewidth = 2, color = 'blue', alpha = 0.5, linestyle = "--")
        plt.plot(x_Data, lower_Data, linewidth = 2, color = 'blue', alpha = 0.5, linestyle = "--")
        plt.xlabel(x_label, fontsize = '18', fontweight = 'bold')
        plt.ylabel('Smoothed Coefficient', fontsize = '18', fontweight = 'bold')
        plt.yticks(fontsize = '15', fontweight = 'bold')
        plt.xticks(fontsize = '15', fontweight = 'bold')
        plt.tight_layout()
        plt.savefig(outPath + "Marginal_Response_"+ DataFields[i] + "_terms2.tif")
    
    
    #Marginal Response Map
    
    fullData = pd.read_csv(fullDataPath)
    fullData = fullData.loc[:, DataFields]
    
    
        
    predict_proba = stats.predict(model,fullData, type = 'terms')
    print(type(predict_proba))
    predict_proba = np.array(predict_proba)
    


    print(predict_proba.shape)


    data = pd.DataFrame(predict_proba, columns = DataFields)  
    for j in DataFields:
        ###########
        #data['mask'] = index_mask
        #data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredRisk_" + j + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data[j], exampleRasterPath, outPath + "Marginal_Map_"+ j + "_terms2.tif")

        
def R_Gam_Summary_regional(combined_Data, target_Data,
                        DataFields, outPath,
                        fullDataPath = None,
                        exampleRasterPath = None,
                        splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                        familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                        region = None, # if not none, iterate across all values of region, to calculate regional models
                        k = -1,
                        parameterNames):
  if region != None:
    regionList = pd.unique(combined_Data[region])

    for x in regionList:
      combined_Data_Regional = combined_Data[combined_Data[region] == x]
      target_Data_Regional = target_Data[combined_Data[region] == x]

      R_Gam_Summary(combined_Data_Regional, target_Data_Regional,
                        DataFields, outPath = outPath + "Region_" + str(x) + '_',
                        fullDataPath = fullDataPath,
                        exampleRasterPath = exampleRasterPath,
                        splineType = splineType, # list for creating space for identifing optimal wifggliness penalization:
                        familyType = familyType, #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                        k = k,
                        parameterNames =parameterNames)    
        
    
def R_logGAM_2dimTest_regional(combined_Data_Training, target_Data_Training, 
                              varsToGroupBy, groupVars, testGroups, DataFields,
                              outPath, preset_GroupVar = None,  
                              splineType = 'cs', 
                              familyType = "binomial",
                              region = 'Region',
                              null_regions = []):
    
    regionList = pd.unique(combined_Data_Training[region]).tolist()
    regionList = list(set(regionList) - set(null_regions))

    
    for x in regionList:
        try:
            combined_Data_Regional = combined_Data_Training[combined_Data_Training[region] == x]
            target_Data_Regional = target_Data_Training[combined_Data_Training[region] == x]
            print('Region = ', x)
            print('length = ', len(combined_Data_Regional))

            R_logGAM_2dimTest(combined_Data_Regional,
                                target_Data_Regional,
                                varsToGroupBy = varsToGroupBy,
                                groupVars =groupVars,
                                testGroups = testGroups,
                                DataFields = DataFields,
                                outPath = outPath + "Region_" + str(x) + '_',
                                preset_GroupVar = preset_GroupVar,
                                splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                                familyType = "binomial" #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                            )
        except:
            print("error: Region ", x)

def R_GAM_YearPredictor_Class_holdouts(combined_Data_Training, target_Data_Training, 
                                preMasked_Data_Path, outPath,
                                outPath_Holdouts, year_List, periodLen, 
                                DataFields, mask,
                               splineType = 'cs', # list for creating space for identifing optimal wifggliness penalization:
                               familyType = "binomial", #where first value indicates minimum penalty, second indicates max penalty, and 3rd value indicates number of values
                                k = -1,
                                holdOut_Subsets = {},
                                elim_type = None,
                                suffix = ''): # number of smooths to allow per parameter - -1 for no maximum)
    '''annually predict fire risk- train model on combined_Data across all available years except year of interest
    save resulting predictions as csv and as tif to location 'outPath'
    
    :param combined_Data_Training: dataFrame including all desired explanatory factors 
            across all locations & years to be used in training model
    :param target_Data_Training: dataFrame including observed fire occurrences 
            across all locations & years to be used in training model
    :param preMasked_Data_Path: file path to location of files to use in predicting fire risk 
                    (note - these files should not have undergone Poisson disk masking)
    :param outPath: desired output location for predicted fire risk files (csv, pickle, and tif)
    :param year_List: list of years for which predictions are desired
    :param Datafields: list of explanatory factors to be intered into model
    :param mask: filepath of raster mask to be used in masking output predictions, 
            and as an example raster for choosing array shape and projections for .tif output files
    :param splineType: type of splines to use - default is cs, which is cubic with shrinkage
    :param familyType: type of GAM to use: default binomial
    :return:  returns a list of all models, accompanied by a list of years being predicted 
            - note - return output is equivalent to data exported as models.pickle
    '''
    
    mgcv = importr('mgcv') # import mgcv library from r
    stats = importr('stats') # import stats library from r
    base = importr('base') # import base library from r
    
    
    model_List = []
    
    
    
    #set up formla for model
    iter_formula = 'value~'
    for iterparam in DataFields:
        iter_formula += " + s(" + iterparam + ', k = ' + str(k) + ',  bs = \"' + splineType + '"'  ')'
    
    
    for iterYear in year_List:
        combined_Data_iter_train = combined_Data_Training[combined_Data_Training['year'] != iterYear]
        combined_Data_iter_train = combined_Data_iter_train.loc[:, DataFields]
        
        target_Data_iter_train = target_Data_Training[target_Data_Training['year'] != iterYear]
        combined_Data_iter_train['value'] = target_Data_iter_train['value']
        
        
        r_combined_Data_iter_train_r = dataFrame_to_r(combined_Data_iter_train)
        
        
       
        
        #run model 
        model = mgcv.gam(formula = base.eval(base.parse(text=iter_formula)),family = base.eval(base.parse(text=familyType)), data = r_combined_Data_iter_train_r)
        
        
        
        #seriesToRaster(predict_iter, templateRasterPath, outPath + "Pred_FireRisk_" + str(iterYear) + ".tif")

        full_X = pd.read_csv(preMasked_Data_Path + "CD_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + ".csv")
        full_X_raw = full_X.loc[:, DataFields]
        full_X = dataFrame_to_r(full_X_raw)

        
        predict_proba = stats.predict(model,full_X, type = 'response')
        predict_proba = np.array(predict_proba)
        


        print(predict_proba.shape)
        # data_risk[1]represents predictedprobability  risk of fire, 
        #data_risk[2] represents probability of no fire
        data = pd.DataFrame(predict_proba, columns = ['PredRisk'])  
        index_mask = image_to_series_simple(mask)             ###########
        data['mask'] = index_mask
        data['PredRisk_Masked'] = data.apply(zeroMasker, axis =1)
        
        data.to_csv(outPath + "PredRisk_" + str(iterYear) + ".csv")
        
        #output predicted risk as tiff
        seriesToRaster(data['PredRisk_Masked'], mask, outPath + "PredRisk_" + str(iterYear) + "_" + str(iterYear + periodLen - 1) + "LogGam_Class.tif")

        for holdOutName in holdOut_Subsets: # create a dictionary of strings (holdoutNames) and lists(lists of data to hold out) to iterate on as a parameter
            print(holdOutName)
            holdoutFields = holdOut_Subsets[holdOutName]

            full_holdout = copy.deepcopy(full_X_raw)  
            if elim_type != dict:
                for iterField in holdoutFields:
                    print(iterField)
                    print(full_holdout[iterField].mean())
                    full_holdout[iterField] = combined_Data_iter_train[iterField].mean()
            elif elim_type == dict:
                for iterField in holdoutFields:
                    print(iterField)
                    if elim_field[iterField] == 'mean':
                        print('Mean: ', full_holdout[iterField].mean())
                        full_holdout[iterField] = combined_Data_iter_train[iterField].mean()
                    elif elim_field[iterField] == 'min':
                        print('Min: ', full_holdout[iterField].min())
                        full_holdout[iterField] = combined_Data_iter_train[iterField].min()
                    elif elim_field[iterField] == 'max':
                        print('Min: ', full_holdout[iterField].max())
                        full_holdout[iterField] = combined_Data_iter_train[iterField].max()
                    elif elim_field[iterField] == 'zero':
                        print('Zero: ')
                        full_holdout[iterField] = 0.0

            full_holdout = dataFrame_to_r(full_holdout)

            holdout_data = stats.predict(model,full_holdout, type = 'response')
            holdout_data = np.array(holdout_data)

            holdout_data = pd.DataFrame(holdout_data, columns = ['PredRisk'])  

            index_mask = image_to_series_simple(mask)

            holdout_data['mask'] = index_mask
            holdout_data['PredRisk_Masked'] = holdout_data.apply(zeroMasker, axis =1)

            holdout_data.to_csv(outPath + "PredRisk_" + str(iterYear) + ".csv")

            #output predicted risk as tiff
            seriesToRaster(holdout_data['PredRisk_Masked'], mask, outPath_Holdouts + "PredRisk_" + holdOutName + '_' + str(iterYear) + "_" + str(iterYear + periodLen - 1) + suffix + ".tif")
