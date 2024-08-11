

#%%
import pandas as pd
import matplotlib.pyplot as plt
from cosfire_workflow_utils import *
#%%
dt = pd.read_excel("I:\My Drive\PhD Steven\MNRAS Paper\COSFIRE_Paper_Data\COSFIRE - Final Results with tuples.xlsx", sheet_name='Selected sets of Parameters')
dt.columns = ['sigma', 'maxrhovalue', 't1', 'sigma0', 'alpha']
dt['descriptor_identity'] = ['descriptor_set_' + str(col) for col in range(1,27) ]
dt
# %%
dtw = pd.read_csv(r"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\result_09052024.csv")
dtw['maxrhovalue'] = dtw['Rho List'].map(lambda x: x.split('..')[2])
dtw.drop(['Rho List'], axis=1, inplace=True)
dtw.columns = dtw.columns.str.lower()
#dtw.rename(columns={'maxrhovalue': 'maxrhovalue'}, inplace=True)
# %%
df1 = dt[['sigma', 'maxrhovalue', 't1', 'sigma0', 'alpha']]
df2 = dtw[['sigma', 'maxrhovalue', 't1', 'sigma0', 'alpha']]

#df2.loc[:, 'maxrhovalue'] = df2['maxrhovalue'].astype('int64')
df2 = df2.astype({"maxrhovalue": int})

df1.sort_values(by=['sigma', 'maxrhovalue', 't1', 'sigma0', 'alpha'], inplace=True)
df2.sort_values(by=['sigma', 'maxrhovalue', 't1', 'sigma0', 'alpha'], inplace=True)


# %%
#check if the two data frames are the same#
print(pd.concat([df1,df2]).drop_duplicates(keep=False))
print(pd.concat([df1, df2]).loc[df1.index.symmetric_difference(df2.index)])
# %%
import numpy as np
def find_max(row):
    return row.max()

max_accuracy = dtw[[col for col in dtw.columns if col.startswith('filter_')]].apply(find_max, axis=1)
avg_accuracy = dtw[[col for col in dtw.columns if col.startswith('filter_')]].apply(lambda x: np.mean(x), axis=1)

dtw['max_accuracy'] = max_accuracy
dtw['avg_accuracy'] = avg_accuracy
dtw.sort_values(by=['max_accuracy' ], ascending = False, inplace = True)
dtw
# %%

y = list(dtw[[col for col in dtw.columns if col.startswith('filter_')]].iloc[0,:])
y2 = list(dtw[[col for col in dtw.columns if col.startswith('filter_')]].iloc[1,:])
x = list(range(1,101))


df = pd.DataFrame(
    {
        "x": list(range(1,101)),
        "y": y,
        "y2": y2,
    }
)


import matplotlib.pyplot as plt 

# plot lines 
plt.plot(x, y, label = "Best 1") 
plt.plot(x, y2, label = "Best 2") 
plt.legend() 
plt.show()
# %%
# Best parameters:
dtw[['sigma', 'maxrhovalue', 't1', 'sigma0', 'alpha']].iloc[0,:]
# %%
'''%ls /scratch/p307791/radio/results/version1/width\=150/noperatorspergender\=100/DoG/sigma\=5/0-5-30/t1\=0.05/sigma0\=0.50/alpha\=0.10/
%sigma             5
%maxrholistvalue      30
%t1             0.05
%sigma0          0.5
%alpha           0.10

'''





#%%
from scipy.io import loadmat

def get_data(path):
       
   # Load the MATLAB file
   data = loadmat(path)
   df0 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['training'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_train = pd.concat([df0, df1, df2, df3], ignore_index=True)

   df0 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_test = pd.concat([df0, df1, df2, df3], ignore_index=True)
  
  
   # Rename the columns:
   column_names = ["descrip_" + str(i) for i in range(1, 401)] + ["label_code"]
   df_train.columns = column_names
   df_test.columns = column_names

   dic_labels = { 'Bent':2,
                  'Compact':3,
                     'FRI':0,
                     'FRII':1
               }

  
   df_train['label_code'] = df_train['label_code'].map(dic_labels)
   df_test['label_code'] = df_test['label_code'].map(dic_labels)


   return df_train, df_test

# %%
path = r"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptor_set_2_train_valid.mat"

df_train, df_test = get_data(path)

print('Shape of df_train: ',df_train.shape)
print('Shape of df_test: ',df_test.shape)


# %%
path = r"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptor_set_1_train_test.mat"

df_train, df_test = get_data(path)

print('Shape of df_train: ',df_train.shape)
print('Shape of df_test: ',df_test.shape)
# %%

num = 3


df_base = pd.read_csv(r"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\model_selection_train_valid_and_test_separate_15052024_v1.csv")

df_base = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','mAP_valid', 'mAP_test']]

df_temp = pd.read_csv(f"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\model_selection_train_valid_and_test_separate_15052024_v{num}.csv")
df_temp = df_temp[['learning_rate', 'batch_size',  'alpha', 'margin','mAP_valid', 'mAP_test']]


cols =  ['learning_rate', 'batch_size','margin','alpha','mAP_valid', 'mAP_test']
df_temp = df_temp[cols]
df_temp.columns = ['learning_rate', 'batch_size','margin','alpha',f'mAP_valid_{num}', f'mAP_test_{num}']

df_base = pd.merge(df_base,df_temp, on=['learning_rate', 'batch_size','margin','alpha'])

df_base_valid = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha','margin'] + list(df_base.columns[df_base.columns.str.startswith('mAP_valid')])]

df_base_test = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha','margin'] + list(df_base.columns[df_base.columns.str.startswith('mAP_test')])]
# %%

dt_m = pd.read_csv(r"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptors\merged_validation_runs.csv")

dt_m.sort_values('mAP_test',ascending = False).head(20)
# %%
path = r"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\model_selection_valid_and_test_13052024_model7.csv"

df_hold_out = pd.read_csv(path)

seed = 2
df_base = df_hold_out.query('seed==@seed')
df_base = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','mAP_valid', 'mAP_test']]
df_base.columns = ['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin',f'mAP_valid_seed_{seed}', f'mAP_test_seed_{seed}']

for seed in list(range(12,100,10)) :

    df_temp = df_hold_out.query('seed==@seed')
    df_temp = df_temp[[ 'learning_rate', 'batch_size',  'alpha', 'margin','mAP_valid', 'mAP_test']]
    df_temp.columns = ['learning_rate', 'batch_size',  'alpha', 'margin',f'mAP_valid_seed_{seed}', f'mAP_test_seed_{seed}']
    df_base = pd.merge(df_base,df_temp, on=['learning_rate', 'batch_size','margin','alpha'])

df_base_valid = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha','margin'] + list(df_base.columns[df_base.columns.str.startswith('mAP_valid')])]
df_base_valid.to_csv("I:/My Drive/deep_learning/deep_hashing/deep_hashing_github/COSFIRE_26_valid_hyperparameters_descriptors/model_selection_valid_and_13052024_model7_wide_format.csv", index = False)

df_base_test = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha','margin'] + list(df_base.columns[df_base.columns.str.startswith('mAP_test')])]
df_base_test.to_csv("I:/My Drive/deep_learning/deep_hashing/deep_hashing_github/COSFIRE_26_valid_hyperparameters_descriptors/model_selection_test_and_13052024_model7_wide_format.csv", index = False)

df_base.to_csv("I:/My Drive/deep_learning/deep_hashing/deep_hashing_github/COSFIRE_26_valid_hyperparameters_descriptors/descriptors/model_selection_valid_and_test_13052024_model7_wide_format.csv",index=False)

# %%

###################################################
# Experiment 1
###################################################
from scipy import stats
def find_max(row):
    return row.max()

################
df_base_valid_st = df_base_valid[list(df_base.columns[df_base.columns.str.startswith('mAP_valid')])]

# Apply the function row-wise
df_base_valid_st['max_map'] = df_base_valid_st.apply(find_max, axis=1)

df_base_valid['max_map'] = df_base_valid.apply(find_max, axis=1)
df_base_valid_st.sort_values(by='max_map', ascending=False, inplace=True)

#maximum Value
df_base_valid_st['max_map'].max()
#########


# extract max value row.
max_index = df_base_valid_st['max_map'].idxmax()
max_row = df_base_valid_st.loc[max_index]
print(max_row)


pvalues = []

for _,index in enumerate(df_base_valid_st.index):
  _, p_value = stats.ttest_ind(max_row[:-1], df_base_valid_st.loc[index][:-1], alternative='greater')
  pvalues.append((p_value>=0.05)*1)
  
  
df_base_valid_st['sig'] = pvalues
df_base_valid_st1 = df_base_valid_st.query('sig == 1')


max_indexx = df_base_valid_st1.iloc[:, :-3].mean().idxmax()

print(sum(pvalues),max_indexx)

#%%
####################################################
# Use the hyperparameters to the test data.
####################################################
optimal_params = df_base_valid.iloc[df_base_valid_st1.index].iloc[:, :6].reset_index()
optimal_params.drop(['index'], axis=1, inplace = True)

#rotation invariance test data with optimal parameters
test_data_1 = pd.merge(optimal_params, df_base_test, how='left')
test_data_1_sub = test_data_1[list(test_data_1.columns[test_data_1.columns.str.startswith('mAP_test_seed')])]
test_data_1['max_map'] = test_data_1_sub.apply(find_max, axis=1)
test_data_1['average_map'] = test_data_1_sub.apply(np.mean, axis=1)

test_data_1.sort_values(by=['average_map'], ascending= False,inplace=True)

test_data_1.describe().iloc[1:3][list(test_data_1.columns[test_data_1.columns.str.startswith('mAP_test_seed')]) + ['average_map']]
#%%


###################################################
# Experiment 2
###################################################

#%%
dt_valid  = pd.read_csv("I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptors\merged_validation_runs_wide_format.csv")

dt_test  = pd.read_csv("I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptors\merged_test_runs_wide_format.csv")
# %%

################
dt_valid_sub = dt_valid[list(dt_valid.columns[dt_valid.columns.str.startswith('mAP_valid')])]

# Apply the function row-wise
dt_valid_sub['max_map'] = dt_valid_sub.apply(find_max, axis=1)

dt_valid_sub['max_map'] = dt_valid_sub.apply(find_max, axis=1)
dt_valid_sub.sort_values(by='max_map', ascending=False, inplace=True)

#maximum Value
dt_valid_sub['max_map'].max()
#########


# extract max value row.
max_index = dt_valid_sub['max_map'].idxmax()
max_row = dt_valid_sub.loc[max_index]
print(max_row)


pvalues = []

for _,index in enumerate(dt_valid_sub.index):
  _, p_value = stats.ttest_ind(max_row[:-1], dt_valid_sub.loc[index][:-1], alternative='greater')
  pvalues.append((p_value>=0.05)*1)
  
  
dt_valid_sub['sig'] = pvalues
dt_valid_sub1 = dt_valid_sub.query('sig == 1')


max_indexx = dt_valid_sub1.iloc[:, :-3].mean().idxmax()

print(sum(pvalues),max_indexx)

dt_valid_sub1.describe().iloc[1:3]

#%%
####################################################
# Use the hyperparameters to the test data.
####################################################
optimal_params = dt_valid.loc[dt_valid_sub1.index].iloc[:, :6].reset_index()
optimal_params.drop(['index'], axis=1, inplace = True)

#rotation invariance test data with optimal parameters
test_data_1 = pd.merge(optimal_params, dt_test, how='left')
test_data_1_sub = test_data_1[list(test_data_1.columns[test_data_1.columns.str.startswith('mAP_test')])]
test_data_1['max_map'] = test_data_1_sub.apply(find_max, axis=1)
test_data_1['average_map'] = test_data_1_sub.apply(np.mean, axis=1)

test_data_1.sort_values(by=['average_map'], ascending= False,inplace=True)

test_data_1.describe().iloc[1:3][list(test_data_1.columns[test_data_1.columns.str.startswith('mAP_test')]) + ['average_map']]
# %%




# %%

# import pandas as pd

# df_base = pd.read_csv('data_results/model_selection_valid_and_test_20052024_1.csv')
# print('Original size: ',df_base)
# df_base = df_base.drop_duplicates()
# print('Deduplicated size: ',df_base)
# df_base = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','mAP_valid', 'mAP_test']]

# for i in list(range(2,4)):
   
#     df_temp = pd.read_csv(f'data_results/model_selection_valid_and_test_20052024_{i}.csv')
#     print('Original size: ',df_temp)
#     df_temp = df_temp.drop_duplicates()
#     print('Deduplicated size: ',df_temp)
#     df_temp = df_temp[['learning_rate', 'batch_size','margin','alpha','mAP_valid', 'mAP_test']]
#     df_temp.columns = ['learning_rate', 'batch_size','margin','alpha',f'mAP_valid_{i}', f'mAP_test_{i}']
#     df_base = pd.merge(df_base,df_temp, on=['learning_rate', 'batch_size','margin','alpha'])
# #%%

# df_base.to_csv('./descriptors/merged_validation_runs__20052024.csv', index = False)

# df_base_valid = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha','margin'] + list(df_base.columns[df_base.columns.str.startswith('mAP_valid')])]
# df_base_valid.to_csv('./descriptors/merged_validation_runs_wide_format_20052024.csv', index = False)

# df_base_test = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha','margin'] + list(df_base.columns[df_base.columns.str.startswith('mAP_test')])]
# df_base_test.to_csv('./descriptors/merged_test_runs_wide_format_20052024.csv', index = False)
# %%




#%%

import re

def extract_numbers(data_list):
    data_list = data_list.split(' ')
    numbers = []
    pattern = r'-?\d+\.\d+'  # Regular expression pattern to match floating-point numbers
    
    for item in data_list:
        if item:  # Check if the item is not an empty string
            match = re.match(pattern, item)
            if match:
                numbers.append(float(match.group()))
    
    return np.array(numbers)



df_testing = pd.read_csv("overall_noise_results/test_df.csv")
df_training = pd.read_csv("overall_noise_results/train_df.csv")
valid_df = pd.read_csv("overall_noise_results/valid_df.csv")

# Example usage
input_list = df_testing['predictions'][0]
output_list = extract_numbers(input_list)
print(df_testing['predictions'][0],output_list)

df_testing['predictions'] = df_testing.predictions.map(extract_numbers)
df_training['predictions'] = df_training.predictions.map(extract_numbers)
valid_df['predictions'] = valid_df.predictions.map(extract_numbers)

#%%
from sklearn.preprocessing import label_binarize
def SimplifiedTopMap(rB, qB, retrievalL, queryL, topk):
  '''
    rB - binary codes of the training set - reference set,
    qB - binary codes of the query set,
    retrievalL - labels of the training set - reference set, 
    queryL - labels of the query set, and 
    topk - the number of top retrieved results to consider.

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = 100
  '''
  num_query = queryL.shape[0]
  mAP = [0] * num_query
  for i, query in enumerate(queryL):
    rel = (np.dot(query, retrievalL.transpose()) > 0)*1 # relevant train label refs.
    hamm = np.count_nonzero(qB[i] != rB, axis=1) #hamming distance
    ind = np.argsort(hamm) #  Hamming distances in ascending order.
    rel = rel[ind] #rel is reordered based on the sorted indices ind, so that it corresponds to the sorted Hamming distances.

    top_rel = rel[:topk] #contains the relevance values for the top-k retrieved results
    tsum = np.sum(top_rel) 

    #skips the iteration if there are no relevant results.
    if tsum == 0:
        continue

    pr_count = np.linspace(1, tsum, tsum) 
    tindex = np.asarray(np.where(top_rel == 1)) + 1.0 #is the indices where top_rel is equal to 1 (i.e., the positions of relevant images)
    pr = pr_count / tindex # precision
    mAP_sub = np.mean(pr) # AP
    mAP[i] = mAP_sub 
      


  return np.round(np.mean(mAP),4) *100 #mAP


def mAP_values(r_database,q_database, thresh = 0.5, percentile = True, topk = 100):
    if percentile:
        r_binary = np.array([((out >= np.percentile(out,thresh))*1)  for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([((out >= np.percentile(out,thresh))*1)  for _, out in enumerate(q_database.predictions)])
    else:
        r_binary = np.array([np.array((out >= thresh) * 1) for _, out in enumerate(r_database.predictions)])
        q_binary = np.array([np.array((out >= thresh) * 1) for _, out in enumerate(q_database.predictions)])

    train_label = label_binarize(r_database.label_code, classes=[0, 1, 2,3])
    valid_label = label_binarize(q_database.label_code, classes=[0,1, 2,3])

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = topk
    mAP = SimplifiedTopMap(rB, qB, retrievalL, queryL, topk)
  
    return np.round(mAP,4), r_binary, train_label, q_binary, valid_label


# %%
threshold_max_map = 0
maP_valid,_, _, _, _ = mAP_values(df_training,valid_df,thresh = threshold_max_map, percentile = False)

mAP_test,_, _, _, _ = mAP_values(df_training, df_testing,thresh = threshold_max_map, percentile = False)
# %%





#%%
from sklearn.manifold import TSNE
from cosfire_workflow_utils import *

num = 2

df_training, df_testing = get_data(rf"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptors\descriptor_set_{num}_train_test.mat")
df_train = preprocessing.normalize(df_training.iloc[:, :-1].values)
y_train = df_training.iloc[:, -1].values

df_test = preprocessing.normalize(df_testing.iloc[:, :-1].values)
y_test = df_testing.iloc[:, -1].values

_, valid_df = get_data(rf"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptors\descriptor_set_{num}_train_valid.mat")
df_valid = preprocessing.normalize(valid_df.iloc[:, :-1].values)
y_valid = valid_df.iloc[:, -1].values

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Plot 1: Training data
tsne = TSNE(n_components=2, random_state=42)
z = tsne.fit_transform(df_train)
df = pd.DataFrame()
df["y"] = y_train
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 4), data=df, ax=axes[0]).set(title="Train: T-SNE projection")

# Plot 2: Validation data
tsne = TSNE(n_components=2, random_state=42)
z = tsne.fit_transform(df_valid)
df = pd.DataFrame()
df["y"] = y_valid
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 4), data=df, ax=axes[1]).set(title="Valid: T-SNE projection")

# Plot 3: Testing data
tsne = TSNE(n_components=2, random_state=42)
z = tsne.fit_transform(df_test)
df = pd.DataFrame()
df["y"] = y_test
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 4), data=df, ax=axes[2]).set(title="Test: T-SNE projection")

plt.tight_layout()
plt.show()




# %%
from scipy.io import loadmat
import pandas as pd
import numpy as np

test_res = []
valid_res = []
for num in range(1,27):
    path_test = f"descriptors/descriptor_set_{num}_test_result.mat"
    dt_test = loadmat(path_test)
    test_res.append(np.round(dt_test['result'][0][93][0][3][15][0][0][0]*100,2))

    path_valid = f"descriptors/descriptor_set_{num}_valid_result.mat"
    dt_valid = loadmat(path_valid)
    valid_res.append(np.round(dt_valid['result'][0][93][0][3][15][0][0][0]*100,2))

data = {'acc_valid': valid_res,
        'acc_test': test_res
        }

df = pd.DataFrame(data)
df['dff'] = df.acc_test - df.acc_valid
df








# %%
# %%

data_path_f1 = "test/descriptor_set_1_train_valid_test.mat"
data_path_valid_f1 = "test/descriptor_set_1_train_valid.mat"
train_df,valid_df,test_df_f1 = get_and_check_data(data_path_f1,data_path_valid_f1,dic_labels)


#%%
train_df, valid_test_df = get_data(data_path_f1,dic_labels)
_, valid_prev = get_data(data_path_valid_f1,dic_labels)

cols = list(train_df.columns[:10])
valid_test_df['id'] = range(valid_test_df.shape[0]) 
valid_df = pd.merge(valid_test_df,valid_prev[cols], on=cols)

diff_set = set(np.array(valid_test_df.id)) - set(np.array(valid_df.id))
test_df = valid_test_df[valid_test_df['id'].isin(diff_set)]

diff_set_valid = set(np.array(valid_test_df.id)) - set(np.array(test_df.id))
valid_df = valid_test_df[valid_test_df['id'].isin(diff_set_valid)]

# diff = valid_df[cols]-valid_prev[cols]
# valid_df = valid_df[~np.isnan(np.array(diff.descrip_1))]
print(valid_df.label_code.value_counts())
# %%
