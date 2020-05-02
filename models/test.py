import pandas as pd
import numpy as np
from enum import Enum

from sklearn.svm import OneClassSVM
from sklearn import metrics

class NormalizationType(Enum):
    ZSCORE = 'zscore'
    MINMAX = 'minmax'

norm_type = NormalizationType.ZSCORE


def normalize_rows( df, norm_type ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]
    
    rows, cols = X.shape
    if( norm_type == NormalizationType.MINMAX ):
        for i in range(0, rows):
            row = X[i,:]
            maxr = max(row)
            minr = min( row)
            if( maxr != minr ):
                X[i,:] = (X[i,:]- minr) /(maxr - minr)
            else:
                X[i,:] = 1
    if( norm_type == NormalizationType.ZSCORE ):
        for i in range(0, rows):
            row = X[i,:]
            mu = np.mean( row )
            sigma = np.std( row )
            if( sigma == 0 ):
                sigma = 0.0001
            X[i,:] = (X[i,:] - mu) / sigma
            
    df = pd.DataFrame( X )
    df['user'] = y 
    return df



# returns the list of unique classids from the last column
def create_userids( df ):
    array = df.values
    y = array[:, -1]
    return unique( y )

# function to get unique values of a list
def unique(list1):      
    # insert the list to the set 
    list_set = set(list1)
    # convert the set to the list 
    unique_list = (list(list_set)) 
    return unique_list


def evaluate_authentication( df ):
    print(df.shape)
    userids = create_userids( df )
    NUM_USERS = len(userids)

    print("Number of classes: ")
    auc_list = list()
    eer_list = list()
    global_positive_scores = list()
    global_negative_scores = list()
    for i in range(0,NUM_USERS):
        userid = userids[i]
        user_train_data = df.loc[df.iloc[:, -1].isin([userid])]
        # Select data for training
        user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
        user_array = user_train_data.values
 
        num_samples = user_array.shape[0]
        train_samples = (int)(num_samples * 0.66)
        test_samples = num_samples - train_samples
        # print("#train_samples: "+str(train_samples)+"\t#test_samples: "+ str(test_samples))
        user_train = user_array[0:train_samples,:]
        user_test = user_array[train_samples:num_samples,:]
     
        other_users_data = df.loc[~df.iloc[:, -1].isin([userid])]
        other_users_data = other_users_data.drop(other_users_data.columns[-1], axis=1)
        other_users_array = other_users_data.values   
        
        clf = OneClassSVM(gamma='scale')
        clf.fit(user_train)
 
        positive_scores = clf.score_samples(user_test)
        negative_scores =  clf.score_samples(other_users_array)   
        auc, eer = compute_AUC_EER(positive_scores, negative_scores )

        global_positive_scores.extend(positive_scores)
        global_negative_scores.extend(negative_scores)

        print(str(userid)+", "+ str(auc)+", "+str(eer) )
        auc_list.append(auc)
        eer_list.append(eer)
    print('AUC  mean : %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('EER  mean:  %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )
    
    global_auc, global_eer = compute_AUC_EER(global_positive_scores, global_negative_scores)
    print("Global AUC: "+str(global_auc))
    print("Global EER: "+str(global_eer))


def compute_AUC_EER(positive_scores, negative_scores):  
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))

    fpr, tpr, threshold = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # Calculating EER
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    return roc_auc, EER


training_filename = "generatedCSVData/balabit_vx_vy.csv"
df = pd.read_csv(training_filename)
df = normalize_rows( df, NormalizationType.ZSCORE)
print("Authentication raw data")
evaluate_authentication( df )