
#%%

#####################
#
############
from cosfire_workflow_utils_12082024 import *
import json
from tqdm import tqdm
#%%%
label = 'mean_std'
train_df_dn = pd.read_csv('paper images/df_training_densenet.csv')
test_df_dn = pd.read_csv('paper images/df_testing_densenet.csv')
valid_df_dn = pd.read_csv('paper images/df_valid_densenet.csv')

predictions_test = []
predictions_train = []
predictions_valid = []

for i in range(train_df_dn.shape[0]):
    predictions_train.append(np.array(ast.literal_eval(train_df_dn.predictions[i])))

for i in range(test_df_dn.shape[0]):
    predictions_test.append(np.array(ast.literal_eval(test_df_dn.predictions[i])))

for i in range(valid_df_dn.shape[0]):
    predictions_valid.append(np.array(ast.literal_eval(valid_df_dn.predictions[i])))

train_df_dn['predictions'] = predictions_train
test_df_dn['predictions'] = predictions_test
valid_df_dn['predictions'] = predictions_valid

thresholds_abs_values = np.arange(-1, 1.2, 0.1)
mAP_results_valid = []
mAP_results_test = []

for _,thresh in enumerate(thresholds_abs_values):
    
    mAP_valid_thresh,_, _, _, _,_,_ = mAP_values(train_df_dn,valid_df_dn,thresh = thresh, percentile = False)  
    mAP_results_valid.append(mAP_valid_thresh)

data_abs_values = {'mAP_valid': mAP_results_valid,        
        'threshold': thresholds_abs_values}

df_thresh_abs_values = pd.DataFrame(data_abs_values)
# Find the index of the maximum mAP value
max_map_index = df_thresh_abs_values['mAP_valid'].idxmax()
threshold_max_map_abs_values_dn = df_thresh_abs_values.loc[max_map_index, 'threshold']


topk=100
mAP_dn,mAP_std_dn,mAP_values_dn, r_binary_dn, train_label_dn, q_binary_dn, valid_label_dn = mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
print('mAP top 100  for DenseNet is: ',np.round(mAP_dn,2))

topk_number_images_dn = list(range(10,205,5))
mAP_topk_dn = []
mAP_topk_std_dn = []
map_values_list = []
for _, topk in enumerate(topk_number_images_dn):
    maP_dn,mAP_std_dn,map_values, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
    mAP_topk_dn.append(maP_dn)
    mAP_topk_std_dn.append(mAP_std_dn)
    map_values_list.append(map_values)
    

data_densenet = {'topk_number_images': topk_number_images_dn,
        'mAP': mAP_topk_dn,
        'mAP_std': mAP_topk_std_dn}
df_densenet = pd.DataFrame(data_densenet)
df_densenet.to_csv(f'mAP_vs_{topk}_images_72bit_densenet_{label}.csv', index = False)

plt.figure(figsize=(10/3, 3))
# Plot the line curve
sns.lineplot(data=df_densenet, x='topk_number_images', y='mAP',label='DenseNet',  color='blue')
# plt.fill_between(df_densenet['topk_number_images'], df_densenet['mAP'] - df_densenet['mAP_std'], df_densenet['mAP'] + df_densenet['mAP_std'], color='#439CEF')
plt.xlabel(f'Top {topk} retrieved images')
plt.ylabel('mAP')
plt.ylim(0, 103)
plt.grid(False)
plt.savefig( f'map_vs_{topk}_number_images_densenet_{label}.png')
plt.savefig( f'map_vs_{topk}_number_images_densenet_{label}.svg',format='svg', dpi=1200)
plt.show()

#%%

predictions_valid = np.loadtxt('predictions_valid.out', delimiter=',')
predictions_train = np.loadtxt('predictions_train.out', delimiter=',')
predictions_test = np.loadtxt('predictions_test.out', delimiter=',')


train_df_cosfire = train_df_dn.drop(['predictions'], axis=1)
train_df_cosfire['predictions'] = list(predictions_train)

test_df_cosfire = test_df_dn.drop(['predictions'], axis=1)
test_df_cosfire['predictions'] = list(predictions_test)


threshold_max_map_abs_values = 0.9038373156470542
# topk=100
# mAP,mAP_std,mAP_values1, r_binary, train_label, q_binary, valid_label= mAP_values(train_df_cosfire, test_df_cosfire,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
# print('mAP top 100  for COSFIRE is: ',np.round(mAP,2))

topk_number_images = list(range(10,205,5))
mAP_topk = []
mAP_topk_std = []
map_values_list = []
for _, topk in enumerate(topk_number_images):
    maP,mAP_std,map_values, _, _, _,_= mAP_values(train_df_cosfire, test_df_cosfire,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    mAP_topk.append(maP)
    mAP_topk_std.append(mAP_std)
    map_values_list.append(map_values)

data = {'topk_number_images': topk_number_images,
        'mAP': mAP_topk,
        'mAP_std': mAP_topk_std}
df_cosfire = pd.DataFrame(data)
df_cosfire.to_csv('mAP_vs_topk_images_72bit_cosfire_{label}.csv', index = False)

df_cosfire = pd.read_csv('mAP_vs_topk_images_72bit_cosfire.csv')

sns.lineplot(data=df_cosfire, x='topk_number_images', y='mAP',label='Cosfire',  color='blue')
# plt.fill_between(df_cosfire['topk_number_images'], df_cosfire['mAP'] - df_cosfire['mAP_std'], df_cosfire['mAP'] + df_cosfire['mAP_std'], color='#439CEF')
plt.xlabel('The number of retrieved samples')
plt.ylabel('mAP')
#plt.ylim(0, 110)
plt.savefig( f'map_vs_topk_number_images_cosfire_{label}.png')
plt.savefig( f'map_vs_topk_number_images_cosfire_{label}.svg',format='svg', dpi=1200)
plt.show()


# %%

#fig, ax = plt.subplots(figsize=(10/3, 3))
sns.lineplot(data=df_cosfire, x='topk_number_images', y='mAP',label='Cosfire',  color='red')
# plt.fill_between(df_cosfire['topk_number_images'], df_cosfire['mAP'] - df_cosfire['mAP_std'], df_cosfire['mAP'] + df_cosfire['mAP_std'], color='#439CEF')
sns.lineplot(data=df_densenet, x='topk_number_images', y='mAP',label='DenseNet',  color='blue')
# plt.fill_between(df_densenet['topk_number_images'], df_densenet['mAP'] - df_densenet['mAP_std'], df_densenet['mAP'] + df_densenet['mAP_std'], color='#ffbaba')

plt.xlabel('The number of retrieved samples')
plt.ylabel('mAP')
plt.ylim(80, 92)
plt.savefig( f'map_vs_topk_number_images_densenet_{label}_cbd.png')
plt.savefig( f'map_vs_topk_number_images_densenet_{label}_cbd.svg',format='svg', dpi=1200)
plt.show()
# %%


#%%
# # When R=k Special case:
# if (query == np.array([0, 0, 1, 0])).sum()==4:#Bent
#         mAP_sub = np.sum(pr) /np.min(np.array([305,topk]))
# elif (query == np.array([0, 1, 0, 0])).sum()==4:#FRII
#         mAP_sub = np.sum(pr) / np.min(np.array([434,topk]))
# elif (query == np.array([1, 0, 0, 0])).sum()==4:#FRI
#         mAP_sub = np.sum(pr) /  np.min(np.array([215,topk]))
# else:# (query == np.array([0, 0, 0, 1])).sum()==4:#Compact
#         mAP_sub = np.sum(pr) / np.min(np.array([226,topk]))
threshold_max_map_abs_values = -0.9
#Bent
topk=305
mAP_bent,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#FRII
topk=434
mAP_frii,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#FRI
topk=215
mAP_fri,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
#Compact
topk=226
mAP_comp,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)

print('====== COSFIRE ======')
print('mAP (305) Bent :',round(mAP_bent,2))
print('mAP (434) FRII :',round(mAP_frii,2))
print('mAP (215) FRI :',round(mAP_fri,2))
print('mAP (226) Compact :',round(mAP_comp,2))
print('Average mAP when R=K: ',round(np.mean(np.array([mAP_bent,mAP_frii,mAP_fri,mAP_comp])),2))
print('====== COSFIRE ======\n')

def mAP_at_k_equals_R(train_df, test_df,threshold_max_map_abs_values,topk):        
    #Bent
    topk=305
    mAP_bent,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    #FRII
    topk=434
    mAP_frii,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    #FRI
    topk=215
    mAP_fri,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    #Compact
    topk=226
    mAP_comp,_,_, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
    mean_average = round(np.mean(np.array([mAP_bent,mAP_frii,mAP_fri,mAP_comp])),2)
    
    return mean_average



#Bent
topk=305
mAP_bent_dn,_,_, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
#FRII
topk=434
mAP_frii_dn,_,_, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
#FRI
topk=215
mAP_fri_dn,_,_, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
#Compact
topk=226
mAP_comp_dn,_,_, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)

print('====== DenseNet ======')
print('mAP (305) Bent :',round(mAP_bent_dn,2))
print('mAP (434) FRII :',round(mAP_frii_dn,2))
print('mAP (215) FRI :',round(mAP_fri_dn,2))
print('mAP (226) Compact :',round(mAP_comp_dn,2))
print('Average mAP when R=K: ',round(np.mean(np.array([mAP_bent_dn,mAP_frii_dn,mAP_fri_dn,mAP_comp_dn])),2))
print('====== DenseNet ======\n')
# %%



##############################################################
##############################################################
###########    Distances                          ###########
##############################################################
##############################################################

#%%

distances_cosine_densenet_dict_list = json.load(open('distances_cosine_densenet_dict.json'))
distances_cosine_cosfire_dict_list = json.load(open('distances_cosine_cosfire_dict_list.json'))

#################################################
#################################################
#######         Bent class              #########
#################################################
#################################################
# print('Class: ',test_df_cosfire.lable_name[0])
# print('Class: ',test_df_cosfire.lable_name[103])


topk=100
start = 100
end = 203



dat_cos_densenet = np.array(distances_cosine_densenet_dict_list[str(start)])
dat_cos_cosfire = np.array(distances_cosine_cosfire_dict_list[str(start)])
for i in range(start,end):#len(distances_cosine_densenet_dict_list.keys())):
     dat_cos_densenet = dat_cos_densenet + np.array(distances_cosine_densenet_dict_list[str(i)])
     dat_cos_cosfire = dat_cos_cosfire + np.array(distances_cosine_cosfire_dict_list[str(i)])
     

dat_cos_densenet_average = dat_cos_densenet/(end-start)#len(distances_cosine_cosfire_dict_list.keys())
dat_cos_cosfire_average = dat_cos_cosfire/(end-start)#len(distances_cosine_densenet_dict_list.keys())



mAP_dn,mAP_std,mAP_values1, pr_denom_dn, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
print('mAP top 100  for DenseNet is: ',np.round(mAP_dn,2))

performance_dn = []
for _,index in enumerate(list(pr_denom_dn.keys())):
   performance_dn.append(len(pr_denom_dn[index][0]))
performance_dn = np.array(performance_dn)
print('DenseNet: ',(performance_dn.min(),performance_dn.max()))

# Extract values greater than 90 and their respective indices
select_ploting_dn = [(i, value) for i, value in enumerate(performance_dn) if value >= 89]


df_plot_dn = pd.DataFrame(select_ploting_dn, columns=['query_number', 'value'])
#df_plot_dn.sort_values(['value'])
#query_number_dn = df_plot_dn.loc[df_plot_dn['value'].idxmin(), 'query_number']
# Find the first position where the value is less than 99
df_plot_dn.query('query_number >= @start and query_number <= @end')
# data_k = np.array((df_plot_dn.sort_values(['value']).query_number))
# query_number_dn = np.where(data_k < 99)[0][0]
query_number_dn = np.array(df_plot_dn.query_number)[0]


mAP_cosf,mAP_std,mAP_values_cosf, pr_denom_cosf, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_cosfire, test_df_cosfire,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
print('mAP top 100  for COSFIRE is: ',np.round(mAP_cosf,2))

performance_cosf = []
for _,index in enumerate(list(pr_denom_cosf.keys())):
   performance_cosf.append(len(pr_denom_cosf[index][0]))
performance_cosf = np.array(performance_cosf)
print('COSFIRE: ',(performance_cosf.min(),performance_cosf.max()))

# Extract values greater than 90 and their respective indices
select_ploting_cosf = [(i, value) for i, value in enumerate(performance_cosf) if value >= 89]


df_plot_cosf = pd.DataFrame(select_ploting_cosf, columns=['query_number', 'value'])
#df_plot_cosf = df_plot_cosf.sort_values(['value'])
# Find the first position where the value is less than 99
data_v = np.array((df_plot_cosf.sort_values(['value']).query_number))
query_number_cosf = np.where(data_v < 99)[0][0]





data1 = normalize_distance(dat_cos_cosfire_average[0:topk])
data2 = normalize_distance(dat_cos_densenet_average[0:topk])



distances_cosine_cosfire_dict_list_subset = {}
distances_cosine_densenet_dict_list_subset = {}
for i in range(start,end):
   distances_cosine_cosfire_dict_list_subset[i] = np.array(distances_cosine_cosfire_dict_list[str(i)])
   distances_cosine_densenet_dict_list_subset[i] = np.array(distances_cosine_densenet_dict_list[str(i)])

# query_number_upd_dn = list(distances_cosine_densenet_dict_list_subset.keys())[query_number_dn]
# query_number_upd_cosf = list(distances_cosine_cosfire_dict_list_subset.keys())[query_number_cosf]
data3 = normalize_distance(distances_cosine_densenet_dict_list_subset[query_number_dn][0:topk])
data4 = normalize_distance(distances_cosine_cosfire_dict_list_subset[query_number_cosf][0:topk])

# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, data1, label='COSFIRE*')#, marker='o', markersize=3)
plt.plot(indices, data2, label='DenseNet*')#, marker='s', markersize=3)
plt.plot(indices, data3, label='DenseNet')#), marker='s', markersize=3)
plt.plot(indices, data4, label='COSFIRE')#, marker='s', markersize=3)

mix_values_cosf = np.array(range(1, topk + 1))
only_correct_cosf = pr_denom_cosf[query_number_cosf]
comb_predictions_cosf = np.array([1 if value in only_correct_cosf else 0 for value in mix_values_cosf])
irrelevant_images_cosf = [(i, value) for i, value in enumerate(comb_predictions_cosf) if  value == 0]

mix_values_dn = np.array(range(1, topk + 1))
only_correct_dn = pr_denom_dn[query_number_dn]
comb_predictions_dn = np.array([1 if value in only_correct_dn else 0 for value in mix_values_dn])
irrelevant_images_dn = [(i, value) for i, value in enumerate(comb_predictions_dn) if  value == 0]

for xx in range(len(irrelevant_images_dn)):
    index = irrelevant_images_dn[xx][0]
    plt.scatter(index, data3[index], color='red', marker='o', s=30)

for yy in range(len(irrelevant_images_cosf)):
    index = irrelevant_images_cosf[yy][0]
    plt.scatter(index, data4[index], color='black', marker='o', s=30)
# Customize the plot
plt.xlabel(f'Top {topk} images')
plt.ylabel('Normalised distance')
plt.legend()
plt.grid(False)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.savefig('paper images/distance_Bent.svg',format='svg', dpi=1200)
plt.show()





#%%
########################################################################

############################################################################

def plot_distances(topk, start, end, label):

  
    # Initialize arrays for DenseNet and COSFIRE
    dat_cos_densenet = np.array(distances_cosine_densenet_dict_list[str(start)])
    dat_cos_cosfire = np.array(distances_cosine_cosfire_dict_list[str(start)])

    # Sum up the distances for the range from start to end
    for i in range(start, end):
        dat_cos_densenet += np.array(distances_cosine_densenet_dict_list[str(i)])
        dat_cos_cosfire += np.array(distances_cosine_cosfire_dict_list[str(i)])

    # Compute average distances
    dat_cos_densenet_average = dat_cos_densenet / (end - start)
    dat_cos_cosfire_average = dat_cos_cosfire / (end - start)

    # DenseNet mAP and performance
    mAP_dn, mAP_std, mAP_values1, pr_denom_dn, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_dn, test_df_dn, thresh=threshold_max_map_abs_values_dn, percentile=False, topk=topk)
    print(f'mAP top {topk} for DenseNet is: {np.round(mAP_dn, 2)}')

    performance_dn = [len(pr_denom_dn[index][0]) for index in pr_denom_dn.keys()]
    performance_dn = np.array(performance_dn)
    print('DenseNet: ', (performance_dn.min(), performance_dn.max()))

    # Extract values greater than 90 and their respective indices for DenseNet
    # select_ploting_dn = [(i, value) for i, value in enumerate(performance_dn) if value >= 89]
    # df_plot_dn = pd.DataFrame(select_ploting_dn, columns=['query_number', 'value'])
    # df_plot_dn = df_plot_dn.query('query_number >= @start and query_number <= @end')
    # df_plot_dn = df_plot_dn.sort_values(['value'])
    # #query_number_dn = df_plot_dn.loc[df_plot_dn['value'].idxmin(), 'query_number']
    # # Find the first position where the value is less than 99

    for qn in range(10):

        df_plot_dn = pd.DataFrame({'query_number': list(pr_denom_dn.keys()),
                    'performance': performance_dn})
        df_plot_dn = df_plot_dn.query('performance <= 95')
        df_plot_dn = df_plot_dn.query('query_number >= @start and query_number <= @end')
        df_plot_dn = df_plot_dn.sort_values(['performance'],ascending=False)

        threshold_dn = 95
        # Use a while loop to adjust the threshold until df_plot_dn is not empty
        while df_plot_dn.empty:
            # Update the select_ploting_dn list based on the current threshold
            df_plot_dn = pd.DataFrame({'query_number': list(pr_denom_dn.keys()),
                    'performance': performance_dn})
            df_plot_dn = df_plot_dn.query('performance <= @threshold_dn')
            df_plot_dn = df_plot_dn.query('query_number >= @start and query_number <= @end')
            df_plot_dn = df_plot_dn.sort_values(['performance'],ascending=False)
            
            # Reduce the threshold by 1
            threshold_dn += 1

        # data_k = np.array((df_plot_dn.sort_values(['value']).value))
        # query_number_dn = np.where(data_k <= 100)[0][0]
        query_number_dn = np.array(df_plot_dn.query_number)[qn]


        # COSFIRE mAP and performance
        mAP_cosf, mAP_std, mAP_values_cosf, pr_denom_cosf, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_cosfire, test_df_cosfire, thresh=threshold_max_map_abs_values, percentile=False, topk=topk)
        print(f'mAP top {topk} for COSFIRE is: {np.round(mAP_cosf, 2)}')

        performance_cosf = [len(pr_denom_cosf[index][0]) for index in pr_denom_cosf.keys()]
        performance_cosf = np.array(performance_cosf)
        print('COSFIRE: ', (performance_cosf.min(), performance_cosf.max()))

        # Extract values greater than 90 and their respective indices for COSFIRE
        #select_ploting_cosf = [(i, value) for i, value in enumerate(performance_cosf) if value >= 0]
        #df_plot_cosf = pd.DataFrame(select_ploting_cosf, columns=['query_number', 'value'])

        df_plot_cosf = pd.DataFrame({'query_number': list(pr_denom_cosf.keys()),
                    'performance': performance_cosf})
        df_plot_cosf = df_plot_cosf.query('performance <= 95')
        df_plot_cosf = df_plot_cosf.query('query_number >= @start and query_number <= @end')
        df_plot_cosf = df_plot_cosf.sort_values(['performance'],ascending=False)

        threshold_cosf = 95
        # Use a while loop to adjust the threshold until df_plot_cosf is not empty
        while df_plot_cosf.empty:
            # Update the select_ploting_dn list based on the current threshold
            df_plot_cosf = pd.DataFrame({'query_number': list(pr_denom_cosf.keys()),
                    'performance': performance_cosf})
            df_plot_cosf = df_plot_cosf.query('performance <= @threshold_cosf')
            df_plot_cosf = df_plot_cosf.query('query_number >= @start and query_number <= @end')
            df_plot_cosf = df_plot_cosf.sort_values(['performance'],ascending=False)
                      
            # Reduce the threshold by 1
            threshold_cosf += 1


        #data_v = np.array((df_plot_cosf.sort_values(['value']).value))
        #query_number_cosf = np.where(data_v <= 100)[0][0]
        query_number_cosf = np.array(df_plot_cosf.query_number)[qn]


        # Normalize distances for plotting
        data1 = normalize_distance(dat_cos_cosfire_average[0:topk])
        data2 = normalize_distance(dat_cos_densenet_average[0:topk])

        distances_cosine_cosfire_dict_list_subset = {}
        distances_cosine_densenet_dict_list_subset = {}
        for i in range(start, end):
            distances_cosine_cosfire_dict_list_subset[i] = np.array(distances_cosine_cosfire_dict_list[str(i)])
            distances_cosine_densenet_dict_list_subset[i] = np.array(distances_cosine_densenet_dict_list[str(i)])


        data3 = normalize_distance(distances_cosine_densenet_dict_list_subset[query_number_dn][0:topk])
        data4 = normalize_distance(distances_cosine_cosfire_dict_list_subset[query_number_cosf][0:topk])

        # Create indices for plotting
        indices = np.arange(len(data1))

        # Create the plot
        plt.figure(figsize=(10/3, 3))

        # Plot both curves
        plt.plot(indices, data1, label='COSFIRE*')
        plt.plot(indices, data2, label='DenseNet*')
        plt.plot(indices, data3, label='DenseNet')
        plt.plot(indices, data4, label='COSFIRE')

        mix_values_cosf = np.array(range(1, topk + 1))
        only_correct_cosf = pr_denom_cosf[query_number_cosf]
        comb_predictions_cosf = np.array([1 if value in only_correct_cosf else 0 for value in mix_values_cosf])

        mix_values_dn = np.array(range(1, topk + 1))
        only_correct_dn = pr_denom_dn[query_number_dn]
        comb_predictions_dn = np.array([1 if value in only_correct_dn else 0 for value in mix_values_dn])

        # Mark irrelevant images on the plot
        irrelevant_images_dn = [(i, value) for i, value in enumerate(comb_predictions_dn) if value == 0]
        irrelevant_images_cosf = [(i, value) for i, value in enumerate(comb_predictions_cosf) if value == 0]
        print(f'{label}: ', irrelevant_images_cosf)
        for xx in range(len(irrelevant_images_dn)):
            index = irrelevant_images_dn[xx][0]
            plt.scatter(index, data3[index], color='red', marker='o', s=30)
        for yy in range(len(irrelevant_images_cosf)):
            index = irrelevant_images_cosf[yy][0]
            plt.scatter(index, data4[index], color='black', marker='o', s=30)

        # Customize the plot
        plt.xlabel(f'Top {topk} images')
        plt.ylabel('Normalized distance')
        plt.legend()
        plt.grid(False)

        # Use scientific notation on y-axis
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # Show the plot
        plt.tight_layout()
        plt.savefig(f'paper images/distance_{label}_{qn}.png', format='svg', dpi=1200)
        plt.show()

        #stoping criteria
        
        if qn == np.array((df_plot_dn.shape[0],df_plot_cosf.shape[0])).min()-1:
            break
        else:
            print('===================================')
            print(f'============= {qn} ==========+====')
            print('===================================')


plot_distances(topk=100, start=0, end=102, label='Bent')
plot_distances(topk=100, start=103, end=202, label='Compact')
plot_distances(topk=100, start=203, end=302, label='FRI')
plot_distances(topk=100, start=303, end=403, label='FRII')
topk=100
start=303
end=403
label='FRII'

# %%
# %%

plt.figure(figsize=(8, 6))
sns.kdeplot(mAP_values_cosf, fill=True, color="blue", alpha=0.5)

# Add titles and labels
plt.title("Cosfire Plot", fontsize=16)
plt.xlabel("Values", fontsize=14)
plt.ylabel("Density", fontsize=14)

# Show the plot
plt.show()


plt.figure(figsize=(8, 6))
sns.kdeplot(mAP_values_dn, fill=True, color="blue", alpha=0.5)

# Add titles and labels
plt.title("DenseNet Plot", fontsize=16)
plt.xlabel("Values", fontsize=14)
plt.ylabel("Density", fontsize=14)

# Show the plot
plt.show()




# %%
label = 'mean_std'
train_df_dn = pd.read_csv('paper images/df_training_densenet.csv')
test_df_dn = pd.read_csv('paper images/df_testing_densenet.csv')
valid_df_dn = pd.read_csv('paper images/df_valid_densenet.csv')

predictions_test = []
predictions_train = []

for i in range(train_df_dn.shape[0]):
    predictions_train.append(np.array(ast.literal_eval(train_df_dn.predictions[i])))

for i in range(test_df_dn.shape[0]):
    predictions_test.append(np.array(ast.literal_eval(test_df_dn.predictions[i])))

train_df_dn['predictions'] = predictions_train
test_df_dn['predictions'] = predictions_test


topk=100
mAP_dn,mAP_std,mAP_values1, pr_denom_dn, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
print('mAP top 100  for DenseNet is: ',np.round(mAP_dn,2))


train_df_cosfire = pd.read_csv('paper images/train_df.csv')
test_df_cosfire = pd.read_csv('paper images/test_df.csv')

predictions_valid = np.loadtxt('paper images/predictions_valid.out', delimiter=',')
predictions_train = np.loadtxt('paper images/predictions_train.out', delimiter=',')
predictions_test = np.loadtxt('paper images/predictions_test.out', delimiter=',')


train_df_cosfire = train_df_dn.drop(['predictions'], axis=1)
train_df_cosfire['predictions'] = list(predictions_train)

test_df_cosfire = test_df_dn.drop(['predictions'], axis=1)
test_df_cosfire['predictions'] = list(predictions_test)


threshold_max_map_abs_values = -0.20000000000000018
topk=100

mAP_test_abs_values,mAP_std,mAP_values1, train_binary, train_label, test_binary, test_label = mAP_values(train_df_cosfire, test_df_cosfire,thresh = threshold_max_map_abs_values, percentile = False)

print('mAP top 100  for COSFIRE is: ',np.round(mAP_test_abs_values,2))


#%%
distances_norm_densenet_dict_list = json.load(open(r"C:\Users\P307791\Documents\deep_hashing_github\paper images\distances_norm_densenet_dict_list.json"))
distances_norm_cosfire_dict_list = json.load(open(r"C:\Users\P307791\Documents\deep_hashing_github\paper images\distances_norm_cosfire_dict_list.json"))

start = 300
topk = 400
query_number = 1

dat_cos_densenet = np.array(distances_norm_densenet_dict_list[str(start)])
dat_cos_cosfire = np.array(distances_norm_cosfire_dict_list[str(start)])
for i in range(start,topk):#len(distances_norm_densenet_dict_list.keys())):
     dat_cos_densenet = dat_cos_densenet + np.array(distances_norm_densenet_dict_list[str(i)])
     dat_cos_cosfire = dat_cos_cosfire + np.array(distances_norm_cosfire_dict_list[str(i)])
     
dat_cos_densenet_average = dat_cos_densenet/(topk-start)#len(distances_norm_densenet_dict_list.keys())
dat_cos_cosfire_average = dat_cos_cosfire/(topk-start)#len(distances_norm_cosfire_dict_list.keys())


mix_values_dn = np.array(range(1,101))
only_correct_dn = pr_denom_dn[query_number]
comb_predictions_dn = np.array([1 if value in only_correct_dn else 0 for value in mix_values_dn])

irrelevant_images_dn = [(i, value) for i, value in enumerate(comb_predictions_dn) if  value == 0]
print(irrelevant_images_dn)


mAP_cosf,mAP_std,mAP_values_cosf, pr_denom_cosf, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_cosfire, test_df_cosfire,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
print('mAP top 100  for COSFIRE is: ',np.round(mAP_cosf,2))


mix_values_cosf = np.array(range(1,101))
only_correct_cosf = pr_denom_cosf[query_number]
comb_predictions_cosf = np.array([1 if value in only_correct_cosf else 0 for value in mix_values_cosf])

irrelevant_images_cosf = [(i, value) for i, value in enumerate(comb_predictions_cosf) if  value == 0]
print(irrelevant_images_cosf)



topk=100
data1_cosf = normalize_distance(dat_cos_cosfire_average[0:topk])
data2_dn = normalize_distance(dat_cos_densenet_average[0:topk])
data3_dn = normalize_distance(distances_norm_densenet_dict_list[str(query_number)][0:topk])
data4_cosf = normalize_distance(distances_norm_cosfire_dict_list[str(query_number)][0:topk])

# Create indices
indices = np.arange(len(data1_cosf))

SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# Create the plot
plt.figure(figsize=(10/3, 3))

# Plot both curves
plt.plot(indices, data1_cosf, label='COSFIRE*')#, marker='o', markersize=3)
plt.plot(indices, data2_dn, label='DenseNet*')#, marker='s', markersize=3)
plt.plot(indices, data3_dn, label='DenseNet')#), marker='s', markersize=3)
plt.plot(indices, data4_cosf, label='COSFIRE')#, marker='s', markersize=3)
for xx in range(len(irrelevant_images_dn)):
    index = irrelevant_images_dn[xx][0]
    plt.scatter(index, data3_dn[index], color='red', marker='o', s=30)

for yy in range(len(irrelevant_images_cosf)):
    index = irrelevant_images_cosf[yy][0]
    plt.scatter(index, data4_cosf[index], color='black', marker='o', s=30)
# Customize the plot
plt.xlabel(f'Top {topk} images')
plt.ylabel('Normalised distance')
plt.legend()
plt.grid(False)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.savefig('paper images/distance_Bent.svg',format='svg', dpi=1200)
plt.show()

# %%

def plot_distances(topk, start, end, label):

    # Initialize arrays for DenseNet and COSFIRE
    dat_cos_densenet = np.array(distances_cosine_densenet_dict_list[str(start)])
    dat_cos_cosfire = np.array(distances_cosine_cosfire_dict_list[str(start)])

    # Sum up the distances for the range from start to end
    for i in range(start, end):
        dat_cos_densenet += np.array(distances_cosine_densenet_dict_list[str(i)])
        dat_cos_cosfire += np.array(distances_cosine_cosfire_dict_list[str(i)])

    # Compute average distances
    dat_cos_densenet_average = dat_cos_densenet / (end - start)
    dat_cos_cosfire_average = dat_cos_cosfire / (end - start)

    # DenseNet mAP and performance
    mAP_dn, mAP_std, mAP_values1, pr_denom_dn, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_dn, test_df_dn, thresh=threshold_max_map_abs_values_dn, percentile=False, topk=topk)
    print(f'mAP top {topk} for DenseNet is: {np.round(mAP_dn, 2)}')

    performance_dn = [len(pr_denom_dn[index][0]) for index in pr_denom_dn.keys()]
    performance_dn = np.array(performance_dn)
    print('DenseNet: ', (performance_dn.min(), performance_dn.max()))

    # Extract values greater than 90 and their respective indices for DenseNet
    select_ploting_dn = [(i, value) for i, value in enumerate(performance_dn) if value >= 89]
    df_plot_dn = pd.DataFrame(select_ploting_dn, columns=['query_number', 'value'])
    df_plot_dn = df_plot_dn.query('query_number >= @start and query_number <= @end')
    #df_plot_dn.sort_values(['value'])
    #query_number_dn = df_plot_dn.loc[df_plot_dn['value'].idxmin(), 'query_number']
    # Find the first position where the value is less than 99

    threshold_dn = 89
    # Use a while loop to adjust the threshold until df_plot_dn is not empty
    while df_plot_dn.empty:
        # Update the select_ploting_dn list based on the current threshold
        select_ploting_dn = [(i, value) for i, value in enumerate(performance_dn) if value >= threshold]
        
        # Create a DataFrame from the selected values
        df_plot_dn = pd.DataFrame(select_ploting_dn, columns=['query_number', 'value'])
        
        # Apply the query filter
        df_plot_dn = df_plot_dn.query('query_number >= @start and query_number <= @end')
        print(df_plot_dn)
        
        # Reduce the threshold by 5
        threshold_dn -= 5

    data_k = np.array((df_plot_dn.sort_values(['value']).value))
    qn = 10
    query_number_dn = np.where(data_k <= 100)[0][qn]

    # COSFIRE mAP and performance
    mAP_cosf, mAP_std, mAP_values_cosf, pr_denom_cosf, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df, test_df, thresh=threshold_max_map_abs_values, percentile=False, topk=topk)
    print(f'mAP top {topk} for COSFIRE is: {np.round(mAP_cosf, 2)}')

    performance_cosf = [len(pr_denom_cosf[index][0]) for index in pr_denom_cosf.keys()]
    performance_cosf = np.array(performance_cosf)
    print('COSFIRE: ', (performance_cosf.min(), performance_cosf.max()))

    # Extract values greater than 90 and their respective indices for COSFIRE
    select_ploting_cosf = [(i, value) for i, value in enumerate(performance_cosf) if value >= 89]
    df_plot_cosf = pd.DataFrame(select_ploting_cosf, columns=['query_number', 'value'])
    df_plot_cosf = df_plot_cosf.query('query_number >= @start and query_number <= @end')
    #df_plot_cosf = df_plot_cosf.sort_values(['value'])
    # Find the first position where the value is less than 99
    threshold = 89

    # Use a while loop to adjust the threshold until df_plot_cosf is not empty
    while df_plot_cosf.empty:
        # Update the select_ploting_cosf list based on the current threshold
        select_ploting_cosf = [(i, value) for i, value in enumerate(performance_cosf) if value >= threshold]
        
        # Create a DataFrame from the selected values
        df_plot_cosf = pd.DataFrame(select_ploting_cosf, columns=['query_number', 'value'])
        
        # Apply the query filter
        df_plot_cosf = df_plot_cosf.query('query_number >= @start and query_number <= @end')
        print(df_plot_cosf)
        
        # Reduce the threshold by 5
        threshold -= 5

    

    data_v = np.array((df_plot_cosf.sort_values(['value']).value))
    query_number_cosf = np.where(data_v <= 100)[0][qn]


    # Normalize distances for plotting
    data1 = normalize_distance(dat_cos_cosfire_average[0:topk])
    data2 = normalize_distance(dat_cos_densenet_average[0:topk])

    distances_cosine_cosfire_dict_list_subset = {}
    distances_cosine_densenet_dict_list_subset = {}
    for i in range(start, end):
        distances_cosine_cosfire_dict_list_subset[i] = np.array(distances_cosine_cosfire_dict_list[str(i)])
        distances_cosine_densenet_dict_list_subset[i] = np.array(distances_cosine_densenet_dict_list[str(i)])

    query_number_upd_dn = list(distances_cosine_densenet_dict_list_subset.keys())[query_number_dn]
    query_number_upd_cosf = list(distances_cosine_cosfire_dict_list_subset.keys())[query_number_cosf]
    data3 = normalize_distance(distances_cosine_densenet_dict_list_subset[query_number_upd_dn][0:topk])
    data4 = normalize_distance(distances_cosine_cosfire_dict_list_subset[query_number_upd_cosf][0:topk])

    # Create indices for plotting
    indices = np.arange(len(data1))

    # Create the plot
    plt.figure(figsize=(10/3, 3))

    # Plot both curves
    plt.plot(indices, data1, label='COSFIRE*')
    plt.plot(indices, data2, label='DenseNet*')
    plt.plot(indices, data3, label='DenseNet')
    plt.plot(indices, data4, label='COSFIRE')

    mix_values_cosf = np.array(range(1, topk + 1))
    only_correct_cosf = pr_denom_cosf[query_number_upd_cosf]
    comb_predictions_cosf = np.array([1 if value in only_correct_cosf else 0 for value in mix_values_cosf])

    mix_values_dn = np.array(range(1, topk + 1))
    only_correct_dn = pr_denom_dn[query_number_upd_dn]
    comb_predictions_dn = np.array([1 if value in only_correct_dn else 0 for value in mix_values_dn])

    # Mark irrelevant images on the plot
    irrelevant_images_dn = [(i, value) for i, value in enumerate(comb_predictions_dn) if value == 0]
    irrelevant_images_cosf = [(i, value) for i, value in enumerate(comb_predictions_cosf) if value == 0]
    print(f'{label}: ', irrelevant_images_cosf)
    for xx in range(len(irrelevant_images_dn)):
        index = irrelevant_images_dn[xx][0]
        plt.scatter(index, data3[index], color='red', marker='o', s=30)
    for yy in range(len(irrelevant_images_cosf)):
        index = irrelevant_images_cosf[yy][0]
        plt.scatter(index, data4[index], color='black', marker='o', s=30)

    # Customize the plot
    plt.xlabel(f'Top {topk} images')
    plt.ylabel('Normalized distance')
    plt.legend()
    plt.grid(False)

    # Use scientific notation on y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Show the plot
    plt.tight_layout()
    plt.savefig(output_dir + f'/distance_{label}_{qn}.svg', format='svg', dpi=1200)
    plt.savefig(output_dir + f'/distance_{label}_{qn}.png')
    plt.show()



plot_distances(topk=100, start=0, end=102, label='Bent')
plot_distances(topk=100, start=103, end=202, label='Compact')
plot_distances(topk=100, start=203, end=302, label='FRI')
plot_distances(topk=100, start=303, end=403, label='FRII')
