#!/usr/bin/env python
# coding: utf-8

# Image retrieval
#%%

from python_modules import *
from cosfire_workflow_utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
hyper_params = {"num_epochs": 200, "lr": 1e-3, 
                 'bits': 72,
                'dataset':'data_complete'}



dic_labels = { 'Bent':2,
  'Compact':3,
    'FRI':0,
    'FRII':1
}


dic_labels_rev = { 2:'Bent',
                3:'Compact',
                  0:'FRI',
                  1: 'FRII'
              }

output_dir = './' + hyper_params["dataset"] + '_' +str(time.ctime()).replace(' ', '_').replace(':','_') + '/'

os.mkdir(output_dir)

print('output directory:',output_dir)

#%%

        
#For Reproducibility
def reproducibility_requirements(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #print("Set seed of", str(seed),"is done for Reproducibility")

reproducibility_requirements()#For Reproducibility


#Data loading
path = Path(hyper_params["dataset"])
bs = 32

# Gets to the parent folders and reads the train and valid folders. 
#The test folder is not imported...
#verify this by dls.train.items, and check the size (len(dls.train.items))
#also for valid ===> dls.valid.items, and check the size (len(dls.valid.items))
# NOT for test since it is not imported.
# Therefore the train contains all the 1180 images and valid 398 images
dls = DataBlock(blocks = (ImageBlock,CategoryBlock),
                get_items = get_image_files,
                splitter = GrandparentSplitter(),
                item_tfms=Resize(224),
                get_y = parent_label)

#Here we split the train to train and valid data by hold out.
# dls = DataBlock(blocks = (ImageBlock,CategoryBlock),
#                 get_items = get_image_files,
#                 splitter = RandomSplitter(valid_pct=0.2, seed=42),
#                 item_tfms=Resize(224),
#                 get_y = parent_label)


dls = dls.dataloaders(path)


## train and valid images as per above split.
train_images = dls.train.items
valid_images = dls.valid.items


class CustomHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hd = nn.Sequential(AdaptiveConcatPool2d(),
                        Flatten(),
                        nn.BatchNorm1d(in_features),
                        nn.Dropout(p = 0.5),
                        nn.Linear(in_features, 512),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm1d(512),
                        nn.Dropout(p = 0.2),
                        nn.Linear(512, out_features),
                        nn.Tanh()
                        )
        
    def forward(self, x):
        return self.hd(x)

#%%
'''
y==> target
u==> output
'''
def DSHLoss(output, target, alpha = 0.00001, margin=36):
    # Move the input tensors to GPU
    output = output.to(device)
    target = target.to(device)
        
    #alpha = 1e-5  # m > 0 is a margin. The margin defines a radius around X1,X2. Dissimilar pairs contribute to the loss
    # function only if their distance is within this radius
    m = 2 * margin  # Initialize U and Y with the current batch's embeddings and labels
    target = target.int()
    

    # Create a duplicate y_label
    target = target.unsqueeze(1).expand(len(target), len(target))
    #y_label = torch.ones_like(torch.empty(len(target), len(target)))
    y_label = torch.ones_like(torch.empty(len(target), len(target))).to(device)
    y_label[target == target.t()] = 0

    #dist = torch.cdist(output, output, p=2).pow(2)
    dist = torch.cdist(output, output, p=2).pow(2).to(device)
    loss1 = 0.5 * (1 - y_label) * dist
    loss2 = 0.5 * y_label * torch.clamp(m - dist, min=0)

    #B1 = torch.norm(torch.abs(output) - 1, p=1, dim=1)
    B1 = torch.norm(torch.abs(output) - 1, p=1, dim=1).to(device)
    B2 = B1.unsqueeze(1).expand(len(target), len(target))

    loss3 = (B2 + B2.t()) * alpha
    minibatch_loss = torch.mean(loss1 + loss2 + loss3)
    return minibatch_loss

#%%
'''
The partial function is a higher-order function provided by the functools module in Python. It is used to create a new function with some of the arguments of the original function already fixed or partially applied.
Here's how partial works:
 - When you call partial(func, *args, **kwargs), it returns a new function that behaves like func but with some of its arguments already set to the values provided in args and kwargs.
- The new function created by partial has the same behavior as the original function, but you can call it with fewer arguments because some of them are already fixed.
- When you call the new function, the fixed arguments are prepended to the arguments you provide, and then the original function is called with the combined set of arguments.
/// Using partial allows you to create specialized versions of functions with some arguments already set, which can be convenient when you want to use the same function with different fixed arguments in different parts of your code.
'''

loss_function = partial(DSHLoss, alpha=0.00001, margin=36)
#loss_function = DSHLoss(alpha=0.00001, margin=36)

# Define model and metrics
densenet161_model = vision_learner(dls, models.densenet161,
                                    custom_head = CustomHead(4416,hyper_params["bits"]),
                                    loss_func = DSHLoss,
                                    metrics = error_rate)
    
#%%
# Freeze the earlier layers and keep only the last layer trainable
densenet161_model.freeze()


# densenet161_model.lr_find()
# plt.savefig(output_dir +'lr_finder.png')
# plt.close()


# Train the model using the lr from the plot for 1 epoch and check
densenet161_model.fit_one_cycle(1)


# Train the model using the lr 
densenet161_model.unfreeze()
densenet161_model.fit_one_cycle(n_epoch = hyper_params["num_epochs"], lr_max=slice(hyper_params["lr"]),wd=1e-3)
densenet161_model.recorder.plot_loss()
plt.savefig(output_dir + 'Densenet_Train_and_valid_Loss_curve_at_freeze.png')
plt.close()
    
    
#Save the model
densenet161_model.save("IMR_DNT161_model_" + hyper_params["dataset"] + '_' + str(time.ctime()).replace(' ', '_').replace(':','_'))
print('Model name: ',"IMR_DNT161_model_" + hyper_params["dataset"] + '_' + str(time.ctime()).replace(' ', '_').replace(':','_'))

 #%%   
#If you have to import it...however..here we do not need since the saved model is currently in the memory.
#densenet161_model.load(model_name)


# %%
#Get the predictions and binarize the data


def binarize_data(file_path,dic_labels,split_train = True):
  if split_train:
    files = file_path
  else:
    files = get_image_files(file_path)
  train_dl = densenet161_model.dls.test_dl(files)
  preds,y = densenet161_model.get_preds(dl=train_dl)
  db_label = np.array([str(files[pth]).split('/')[2] for pth in range(len(files))])
  db_label = np.array([dic_labels[lbl] for x, lbl in enumerate(db_label)])
  
  return preds, db_label


#Save/load the the binaries 

# if os.path.exists(output_dir +'/df_testing.csv') and os.path.exists(output_dir +'/df_training.csv') and os.path.exists(output_dir +'/df_valid.csv'):
    
#     df_training = pd.read_csv(output_dir +'/df_training.csv')
#     df_training['predictions'] = [ast.literal_eval(pred) for pred in df_training.predictions]

#     df_valid = pd.read_csv(output_dir +'/df_valid.csv')
#     df_valid['predictions'] = [ast.literal_eval(pred) for pred in df_valid.predictions]

#     df_testing = pd.read_csv(output_dir +'/df_testing.csv')
#     df_testing['predictions'] = [ast.literal_eval(pred) for pred in df_testing.predictions]

######################################################################
preds_train,  train_label = binarize_data(file_path =  'radio_complete/train/',
                                                dic_labels = dic_labels,
                                                split_train = False
                                                )

preds_valid, valid_label = binarize_data(file_path =  'radio_complete/valid/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )
preds_test, test_label = binarize_data(file_path = 'radio_complete/test/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )

######################################################################
preds_train_f05,  train_label_f05 = binarize_data(file_path =  'data_complete_gnoise_f05/train/',
                                                dic_labels = dic_labels,
                                                split_train = False
                                                )

preds_valid_f05, valid_label_f05 = binarize_data(file_path =  'data_complete_gnoise_f05/valid/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )
preds_test_f05, test_label_f05 = binarize_data(file_path = 'data_complete_gnoise_f05/test/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )
                                              
######################################################################
preds_train_f1,  train_label_f1 = binarize_data(file_path =  'data_complete_gnoise_f1/train/',
                                                dic_labels = dic_labels,
                                                split_train = False
                                                )

preds_valid_f1, valid_label_f1 = binarize_data(file_path =  'data_complete_gnoise_f1/valid/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )
preds_test_f1, test_label_f1 = binarize_data(file_path = 'data_complete_gnoise_f1/test/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )

######################################################################
preds_train_f3,  train_label_f3 = binarize_data(file_path =  'data_complete_gnoise_f3/train/',
                                                dic_labels = dic_labels,
                                                split_train = False
                                                )

preds_valid_f3, valid_label_f3 = binarize_data(file_path =  'data_complete_gnoise_f3/valid/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )
preds_test_f3, test_label_f3 = binarize_data(file_path = 'data_complete_gnoise_f3/test/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )


######################################################################
preds_train_f5,  train_label_f5 = binarize_data(file_path =  'data_complete_gnoise_f5/train/',
                                                dic_labels = dic_labels,
                                                split_train = False
                                                )

preds_valid_f5, valid_label_f5 = binarize_data(file_path =  'data_complete_gnoise_f5/valid/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )
preds_test_f5, test_label_f5 = binarize_data(file_path = 'data_complete_gnoise_f5/test/',
                                            dic_labels = dic_labels,
                                            split_train = False
                                                )
######################################################################

######################################################################
#### save the Test predictions
######################################################################
df_testing = pd.DataFrame()
flat_predictions_test = []
for i in range(len(test_label)):
  flat_predictions_test.append(list(np.array(preds_test)[i]))

df_testing['predictions'] = flat_predictions_test
df_testing['label_code'] = test_label
df_testing['lable_name'] = df_testing['label_code'].map(dic_labels_rev)
df_testing.to_csv(output_dir +'/df_testing.csv',index = False)

######################################################################
#### save the Valid predictions
######################################################################
df_valid = pd.DataFrame()
flat_predictions_valid = []
for i in range(len(valid_label)):
  flat_predictions_valid.append(list(np.array(preds_valid)[i]))

df_valid['predictions'] = flat_predictions_valid
df_valid['label_code'] = valid_label
df_valid['lable_name'] = df_valid['label_code'].map(dic_labels_rev)
df_valid.to_csv(output_dir +'/df_valid.csv',index = False)

######################################################################
df_testing_f05 = pd.DataFrame()
flat_predictions_test_f05 = []
for i in range(len(test_label_f05)):
  flat_predictions_test_f05.append(list(np.array(preds_test_f05)[i]))

df_testing_f05['predictions'] = flat_predictions_test_f05
df_testing_f05['label_code'] = test_label_f05
df_testing_f05['lable_name'] = df_testing_f05['label_code'].map(dic_labels_rev)
df_testing_f05.to_csv(output_dir +'/df_testing_f05.csv',index = False)

######################################################################
df_testing_f1 = pd.DataFrame()
flat_predictions_test_f1 = []
for i in range(len(test_label_f1)):
  flat_predictions_test_f1.append(list(np.array(preds_test_f1)[i]))

df_testing_f1['predictions'] = flat_predictions_test_f1
df_testing_f1['label_code'] = test_label_f1
df_testing_f1['lable_name'] = df_testing_f1['label_code'].map(dic_labels_rev)
df_testing_f1.to_csv(output_dir +'/df_testing_f1.csv',index = False)

######################################################################
df_testing_f3 = pd.DataFrame()
flat_predictions_test_f3 = []
for i in range(len(test_label_f3)):
  flat_predictions_test_f3.append(list(np.array(preds_test_f3)[i]))

df_testing_f3['predictions'] = flat_predictions_test_f3
df_testing_f3['label_code'] = test_label_f3
df_testing_f3['lable_name'] = df_testing_f3['label_code'].map(dic_labels_rev)
df_testing_f3.to_csv(output_dir +'/df_testing_f3.csv',index = False)

######################################################################
df_testing_f5 = pd.DataFrame()
flat_predictions_test_f5 = []
for i in range(len(test_label_f5)):
  flat_predictions_test_f5.append(list(np.array(preds_test_f5)[i]))

df_testing_f5['predictions'] = flat_predictions_test_f5
df_testing_f5['label_code'] = test_label_f5
df_testing_f5['lable_name'] = df_testing_f5['label_code'].map(dic_labels_rev)
df_testing_f5.to_csv(output_dir +'/df_testing_f5.csv',index = False)

######################################################################
#### save the Train predictions
######################################################################
df_training = pd.DataFrame()
flat_predictions_train = []
for i in range(len(train_label)):
  flat_predictions_train.append(list(np.array(preds_train)[i]))

df_training['predictions'] = flat_predictions_train
df_training['label_code'] = train_label
df_training['lable_name'] = df_training['label_code'].map(dic_labels_rev)
df_training.to_csv(output_dir +'/df_training.csv',index = False)

######################################################################
df_training_f05 = pd.DataFrame()
flat_predictions_train_f05 = []
for i in range(len(train_label_f05)):
  flat_predictions_train_f05.append(list(np.array(preds_train_f05)[i]))

df_training_f05['predictions'] = flat_predictions_train_f05
df_training_f05['label_code'] = train_label_f05
df_training_f05['lable_name'] = df_training_f05['label_code'].map(dic_labels_rev)
df_training_f05.to_csv(output_dir +'/df_training_f05.csv',index = False)

######################################################################
df_training_f1 = pd.DataFrame()
flat_predictions_train_f1 = []
for i in range(len(train_label_f1)):
  flat_predictions_train_f1.append(list(np.array(preds_train_f1)[i]))

df_training_f1['predictions'] = flat_predictions_train_f1
df_training_f1['label_code'] = train_label_f1
df_training_f1['lable_name'] = df_training_f1['label_code'].map(dic_labels_rev)
df_training_f1.to_csv(output_dir +'/df_training_f1.csv',index = False)

######################################################################
df_training_f3 = pd.DataFrame()
flat_predictions_train_f3 = []
for i in range(len(train_label_f3)):
  flat_predictions_train_f3.append(list(np.array(preds_train_f3)[i]))

df_training_f3['predictions'] = flat_predictions_train_f3
df_training_f3['label_code'] = train_label_f3
df_training_f3['lable_name'] = df_training_f3['label_code'].map(dic_labels_rev)
df_training_f3.to_csv(output_dir +'/df_training_f3.csv',index = False)

######################################################################
df_training_f5 = pd.DataFrame()
flat_predictions_train_f5 = []
for i in range(len(train_label_f5)):
  flat_predictions_train_f5.append(list(np.array(preds_train_f5)[i]))

df_training_f5['predictions'] = flat_predictions_train_f5
df_training_f5['label_code'] = train_label_f5
df_training_f5['lable_name'] = df_training_f5['label_code'].map(dic_labels_rev)
df_training_f5.to_csv(output_dir +'/df_training_f5.csv',index = False)

######################################################################
#### Model Validation
######################################################################

thresholds_abs_values = np.arange(-1, 1.2, 0.1)
mAP_results_valid = []
mAP_results_test = []
df_training1 = shuffle(df_training)
for _,thresh in enumerate(thresholds_abs_values):
    
    # mAP_valid_thresh,_, _, _, _ = mAP_values(df_training,df_valid,thresh = thresh, percentile = False,topk=100)
    # mAP_test_thresh,_, _, _, _ = mAP_values(df_training, df_testing,thresh = thresh, percentile = False,topk=100)
    mAP_valid_thresh, _, _ = MapWithPR_values(df_training1,df_valid,thresh = thresh, percentile = False,topk=100)
    mAP_test_thresh, _, _ = MapWithPR_values(df_training1, df_testing,thresh = thresh, percentile = False,topk=100)
    
    
    

    mAP_results_valid.append(mAP_valid_thresh)
    mAP_results_test.append(mAP_test_thresh)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(thresholds_abs_values, mAP_results_valid, label='mAP_valid')
plt.plot(thresholds_abs_values, mAP_results_test, label='mAP_test')
plt.xlabel('Threshold')
plt.ylabel('mAP')
plt.legend()
plt.savefig(output_dir + '/Maps_curves_abs_values.png')
plt.close()

data = {'mAP': mAP_results_valid,
        'thresholds_abs_values': thresholds_abs_values}

df = pd.DataFrame(data)
df.to_csv(output_dir +'/mAP_vs_threshold_validation_abs_values.csv',index = False)

# Find the index of the maximum mAP value
max_map_index = df['mAP'].idxmax()

# Retrieve the threshold corresponding to the maximum mAP
threshold_max_map = df.loc[max_map_index, 'thresholds_abs_values']

#maP_valid,mAP_std,mAP_values1, r_binary, train_label, q_binary, valid_label = mAP_values(df_training,df_valid,thresh = threshold_max_map, percentile = False,topk=100)
maP_tesst,mAP_std,mAP_values1, train_binary, train_label, test_binary, test_label = mAP_values(df_training1,df_testing,thresh = threshold_max_map, percentile = False,topk=100)

# maP_test,_, _, _, _ = mAP_values(df_training,df_testing,thresh = threshold_max_map, percentile = False, topk=100)

# maP_test_f05,_, _, _, _ = mAP_values(df_training,df_testing_f05,thresh = threshold_max_map, percentile = False, topk=100)
# maP_test_f1,_, _, _, _ = mAP_values(df_training,df_testing_f1,thresh = threshold_max_map, percentile = False, topk=100)
# maP_test_f3,_, _, _, _ = mAP_values(df_training,df_testing_f3,thresh = threshold_max_map, percentile = False, topk=100)
# maP_test_f5,_, _, _, _ = mAP_values(df_training,df_testing_f5,thresh = threshold_max_map, percentile = False, topk=100)

maP_valid, _, _ = MapWithPR_values(df_training1,df_valid,thresh = threshold_max_map, percentile = False,topk=100)

maP_test, _, _ = MapWithPR_values(df_training1,df_testing,thresh = threshold_max_map, percentile = False, topk=100)

maP_test_f05, _, _ = MapWithPR_values(df_training,df_testing_f05,thresh = threshold_max_map, percentile = False, topk=100)
maP_test_f1, _, _ = MapWithPR_values(df_training,df_testing_f1,thresh = threshold_max_map, percentile = False, topk=100)
maP_test_f3, _, _ = MapWithPR_values(df_training,df_testing_f3,thresh = threshold_max_map, percentile = False, topk=100)
maP_test_f5, _, _ = MapWithPR_values(df_training,df_testing_f5,thresh = threshold_max_map, percentile = False, topk=100)




# Create a dictionary with the variable names and their values
data = {'optimal threshold': [threshold_max_map],
        'Best Validation mAP': [maP_valid],
        'Test mAP': [maP_test],
        'Test mAP f05': [maP_test_f05],
        'Test mAP f1': [maP_test_f1],
        'Test mAP f3': [maP_test_f3],
        'Test mAP f5': [maP_test_f5]
        }

# Create the DataFrame
df_results = pd.DataFrame(data)
df_results.to_csv(output_dir +'/final_results.csv',index = False)


################################################################
## Train
################################################################
# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Iterate over label_code values
for label_code in range(4):
    dff = df_training.query(f'label_code == {label_code}')
    out_array_train = []
    dd_train = np.array([out_array_train.extend(np.array(out)) for _, out in enumerate(dff.predictions)])
    out_array_train = np.array(out_array_train)
    
    # Plot the KDE curve with a hue
    sns.kdeplot(out_array_train, label=f'{dic_labels_rev[label_code]}', ax=ax)

# Set the x-axis limits
ax.set_xlim(-1, 1)

# Customize the plot
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(True)
ax.legend()
plt.savefig(output_dir + 'Density_plot_train.png')
plt.close()

################################################################

array_dat = []
for i in range(df_training['predictions'].shape[0]):
  array_dat.append(list(df_training['predictions'].iloc[i]))

array_dat = np.array(array_dat)
array_dat.shape

from sklearn.manifold import TSNE

y = df_training.lable_name
tsne = TSNE(n_components=2, random_state=42)
z = tsne.fit_transform(array_dat)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 4),
                data=df).set(title="Model last layer T-SNE projection")

plt.savefig(output_dir + 'T-SNE_projection_train.png')
plt.close()



################################################################
## Valid
################################################################

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Iterate over label_code values
for label_code in range(4):
    dff = df_valid.query(f'label_code == {label_code}')
    out_array_valid = []
    dd_valid = np.array([out_array_valid.extend(np.array(out)) for _, out in enumerate(dff.predictions)])
    out_array_valid = np.array(out_array_valid)
    
    # Plot the KDE curve with a hue
    sns.kdeplot(out_array_valid, label=f'{dic_labels_rev[label_code]}', ax=ax)

# Set the x-axis limits
ax.set_xlim(-1, 1)

# Customize the plot
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(True)
ax.legend()
plt.savefig(output_dir + 'Density_plot_valid.png')
plt.close()

################################################################

array_dat = []
for i in range(df_valid['predictions'].shape[0]):
  array_dat.append(list(df_valid['predictions'].iloc[i]))

array_dat = np.array(array_dat)
array_dat.shape

from sklearn.manifold import TSNE

y = df_valid.lable_name
tsne = TSNE(n_components=2, random_state=42)
z = tsne.fit_transform(array_dat)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 4),
                data=df).set(title="Model last layer T-SNE projection")

plt.savefig(output_dir + 'T-SNE_projection_valid.png')
plt.close()


################################################################
## Test
################################################################
# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Iterate over label_code values
for label_code in range(4):
    dff = df_testing.query(f'label_code == {label_code}')
    out_array_test = []
    dd_test = np.array([out_array_test.extend(np.array(out)) for _, out in enumerate(dff.predictions)])
    out_array_test = np.array(out_array_test)
    
    # Plot the KDE curve with a hue
    sns.kdeplot(out_array_test, label=f'{dic_labels_rev[label_code]}', ax=ax)

# Set the x-axis limits
ax.set_xlim(-1, 1)

# Customize the plot
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(True)
ax.legend()
plt.savefig(output_dir + 'Density_plot_test.png')
plt.close()

################################################################

array_dat = []
for i in range(df_testing['predictions'].shape[0]):
  array_dat.append(list(df_testing['predictions'].iloc[i]))

array_dat = np.array(array_dat)
array_dat.shape

from sklearn.manifold import TSNE

y = df_testing.lable_name
tsne = TSNE(n_components=2, random_state=42)
z = tsne.fit_transform(array_dat)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 4),
                data=df).set(title="Model last layer T-SNE projection")

plt.savefig(output_dir + 'T-SNE_projection_test.png')
plt.close()

################################################################

print('The optimal threshold is: ', threshold_max_map)
print('The Best Validation mAP is: ',maP_valid)

print('At the optimal threshold: ', threshold_max_map)
print('The Test  mAP is: ',maP_test)

# def add_data_paths(train_paths = '/scratch/p307791/data_complete/train/*/*', dic_labels = dic_labels):
#     df_labels_train_paths = pd.DataFrame()
#     df_labels_train_paths['paths'] = glob.glob(train_paths)
#     df_labels_train_paths['label'] = df_labels_train_paths['paths'].apply(lambda x: x.split(os.path.sep)[5] )
#     df_labels_train_paths['label_code'] = df_labels_train_paths['label'].map(dic_labels)
#     df_labels_train_paths = df_labels_train_paths.sort_values('label_code')
#     df_labels_train_paths = df_labels_train_paths.reset_index()[['paths', 'label', 'label_code']]
#     return df_labels_train_paths

# df_labels_train_paths = add_data_paths(train_paths = '/scratch/p307791/data_complete/train/*/*', dic_labels = dic_labels)
# df_labels_test_paths = add_data_paths(train_paths = '/scratch/p307791/data_complete/test/*/*', dic_labels = dic_labels)

# def perf_percentages(input_data):
#     unique, counts = np.unique(input_data, return_counts=True)
#     df = pd.DataFrame()
#     df['unique'] = unique
#     df['counts'] = counts
#     df['Percentage'] = np.round(counts / counts.sum() * 100)
#     return df

        
# def query_image(test_image_index = 190, 
#             test_images_paths = df_labels_test_paths,
#             train_images_db_paths = df_labels_train_paths,
#             train_images_db = train_binary,
#             test_binary = test_binary):

        
#     print('Test Image is: ', test_images_paths.label[test_image_index])
#     fig = plt.figure(figsize=(3, 3))
#     image_test = Image.open(test_images_paths.paths[test_image_index])
#     image_test = torch.from_numpy(np.array(image_test))
#     plt.imshow(image_test[:, :, 1], cmap='viridis')
#     plt.axis('off')
#     plt.savefig(output_dir +f'/imr_image_query_{test_image_index}.png')
#     plt.savefig(output_dir +f'/imr_image_query_{test_image_index}.svg',format='svg', dpi=1200)

#     test_image = test_binary[test_image_index]  
#     #np.count_nonzero(np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1,
#     # 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])==np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
#     # 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]))
#     similarity_distance = np.count_nonzero(test_image != train_images_db, axis=1)
#     sort_indices = np.argsort(similarity_distance)
#     top_indices = sort_indices[:100]
#     #print(top_indices)
#     paths_to_imgs = [train_images_db_paths.paths[index] for _,index in enumerate(top_indices)]
#     df = perf_percentages([train_images_db_paths.label[index] for index in top_indices])
#     print(df)
#     cols = 7
#     rows = 4

#     fig = plt.figure(figsize=(2 * cols, 2 * rows))
#     for col in range(cols):
#         for i, img_path in enumerate(paths_to_imgs[:cols*rows]):
#             ax = fig.add_subplot(rows, cols, i + 1)
#             ax.grid(visible=False)
#             ax.axis("off")
#             image = Image.open(img_path)
#             image = torch.from_numpy(np.array(image))
#             ax.imshow(image[:, :, 1], cmap='viridis')
#             ax.set_title(img_path.split(os.path.sep)[5])

#     plt.savefig(output_dir +f'/imr_images_rb_{test_image_index}.png')
#     plt.savefig(output_dir +f'/imr_images_rb_{test_image_index}.svg',format='svg', dpi=1200)
#     #plt.show()

# query_image(test_image_index = 15)
# query_image(test_image_index = 150)
# query_image(test_image_index = 350)
# query_image(test_image_index = 270)
# query_image(test_image_index = 251)
# query_image(test_image_index = 271)
# query_image(test_image_index = 200)

mAP, cum_prec, cum_recall, cum_map = MapWithPR_values_withCumMAP(df_training1,df_testing,thresh = threshold_max_map, percentile = False)

pr_data = {
"mAP": cum_map.tolist(),
"P": cum_recall.tolist(),
"R": cum_prec.tolist()
}

with open(output_dir + '/DenseNet_raw.json', 'w') as f:
            f.write(json.dumps(pr_data))

num_dataset=df_training1.shape[0]
index_range = num_dataset // 100
index = [i * 100 - 1 for i in range(1, index_range + 1)]
max_index = max(index)
overflow = num_dataset - index_range * 100
index = index + [max_index + i for i in range(1, overflow + 1)]
cum_map = cum_map[index]
c_prec = cum_prec[index]
c_recall = cum_recall[index]

pr_data = {
"index": index,
"mAP": cum_map.tolist(),
"P": c_prec.tolist(),
"R": c_recall.tolist()
}

with open(output_dir + '/DenseNet.json', 'w') as f:
            f.write(json.dumps(pr_data))

pr_data = {
    "COSFIRE": 'FINAL_Best1_2/COSFIRE.json',
    "DenseNet": output_dir + '/DenseNet.json'
}
N = 200
# N = -1
for key in pr_data:
    path = pr_data[key]
    pr_data[key] = json.load(open(path))


# markers = "DdsPvo*xH1234h"
markers = ".........................."
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

plt.figure(figsize=(15, 5))
plt.subplot(131)

for method in pr_data:
    P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    print(len(P))
    print(len(R))
    plt.plot(R, P, linestyle="-",  label=method)
plt.grid(False)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.subplot(132)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, R, linestyle="-",  label=method)
plt.xlim(0, max(draw_range))
plt.grid(False)
plt.xlabel('The number of retrieved samples')
plt.ylabel('recall')
plt.legend()

plt.subplot(133)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, P, linestyle="-",  label=method)
plt.xlim(0, max(draw_range))
plt.grid(False)
plt.xlabel('The number of retrieved samples')
plt.ylabel('precision')
plt.legend()
plt.savefig(output_dir +"/pr.png")


topk_number_images = list(range(90, 1180, 10)) + [1180]
mAP_topk = []

# Loop with progress bar
for _, topk in tqdm(enumerate(topk_number_images), total=len(topk_number_images), desc="Calculating mAP"):
    maP, _, _, _, _,_,_ = mAP_values(df_training1, df_testing, thresh=threshold_max_map, percentile=False, topk=topk)
    mAP_topk.append(maP)

data = {'topk_number_images': topk_number_images, 'mAP': mAP_topk}
df = pd.DataFrame(data)
df.to_csv(output_dir + '/mAP_vs_topk_images_72bit_densenet.csv', index=False)

# Plot the line curve
sns.lineplot(x='topk_number_images', y='mAP', data=df, color='r')
plt.xlabel('Top k number of images')
plt.ylabel('mAP')
plt.savefig(output_dir + '/map_vs_topk_number_images.png')


# Second loop with progress bar
mAP_topk = []
for _, topk in tqdm(enumerate(topk_number_images), total=len(topk_number_images), desc="Calculating mAP with PR"):
    mAP, cum_prec, cum_recall = MapWithPR_values(df_training1, df_testing, thresh=threshold_max_map, percentile=False, topk=topk)
    mAP_topk.append(mAP)

data = {'topk_number_images': topk_number_images, 'mAP': mAP_topk}
df = pd.DataFrame(data)
df.to_csv(output_dir + '/mAP_vs_topk_images_72bit_densenet2.csv', index=False)

# Plot the line curve
sns.lineplot(x='topk_number_images', y='mAP', data=df, color='r')
plt.xlabel('Top k number of images')
plt.ylabel('mAP')
plt.savefig(output_dir + '/map_vs_topk_number_images2.png')




#distances_hamm_densenet_dict = {}
distances_hamm_densenet_dict = {}

# Use tqdm to create a progress bar for the loop
for i in tqdm(range(test_binary.shape[0]), desc="Processing Test Samples"):

  distances_hamm_densenet = []
  vector_a = test_binary[i]
  for j in range(train_binary.shape[0]):
    vector_b = train_binary[j]
    distances_hamm_densenet.append(similarity_distance_count(vector_a, vector_b))

  distances_hamm_densenet = np.array(distances_hamm_densenet)

  ind = np.argsort(distances_hamm_densenet)
  distances_hamm_densenet = distances_hamm_densenet[ind]
  distances_hamm_densenet_dict[i] = distances_hamm_densenet


distances_hamm_densenet_dict_list = {}
for i in range(len(distances_hamm_densenet_dict.keys())):
  distances_hamm_densenet_dict_list[i] = list(distances_hamm_densenet_dict[i])


# Apply the conversion function to the entire dictionary
#converted_dict = convert_numpy(distances_hamm_densenet_dict_list)

# # Now save the converted dictionary as JSON
# with open(output_dir + '/distances_hamm_densenet_dict_list.json', 'w') as json_file:
#   json.dump(distances_hamm_densenet_dict_list, json_file)

with open(output_dir + '/distances_hamm_densenet_dict_list.txt', 'w') as txt_file:
  txt_file.write(str(distances_hamm_densenet_dict_list))


dat_hamm_densenet = np.array(distances_hamm_densenet_dict[0])

for i in range(1,len(distances_hamm_densenet_dict.keys())):
  dat_hamm_densenet = dat_hamm_densenet + np.array(distances_hamm_densenet_dict[i])
    
    
dat_densenet_hamm_average = dat_hamm_densenet/len(distances_hamm_densenet_dict.keys()) 




topk = 100
data1 = dat_densenet_hamm_average[0:topk]

#data2 = dat_densenet_hamm_average[0:topk]
# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(indices, data1, label='densenet hamm', marker='o')
#plt.plot(indices, data2, label='DenseNet hamm', marker='s')

# Customize the plot
plt.xlabel(f'Top {topk} images', fontsize=12)
plt.ylabel('Hamming distance', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.savefig(output_dir + "/distance_measure_hamm_comparison.png")
plt.show()
plt.close()


dat_densenet = distances_hamm_densenet_dict[0]
#dat_densenet = distances_densenet_dict[0]
for i in range(1,len(distances_hamm_densenet_dict.keys())):
  dat_densenet =+ distances_hamm_densenet_dict[i]
  #dat_densenet =+ distances_densenet_dict[i]
    
dat_densenet_average = dat_densenet/len(distances_hamm_densenet_dict.keys())
#dat_densenet_average = dat_densenet/len(distances_densenet_dict.keys())

# Data vectors

data1 = dat_densenet_average[0:topk]
#data2 = dat_densenet_average[0:topk]

# Create indices
indices = np.arange(len(data1))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both curves
plt.plot(indices, data1, label='densenet hamm', marker='o')
#plt.plot(indices, data2, label='DenseNet cosine', marker='s')

# Customize the plot
plt.xlabel(f'Top {topk} images', fontsize=12)
plt.ylabel('Hamming distance', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Use scientific notation on y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig(output_dir + "/distance_measure_hamm_comparison.png")
plt.show()
plt.close()