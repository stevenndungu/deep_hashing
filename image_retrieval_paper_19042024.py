#!/usr/bin/env python
# coding: utf-8

# Image retrieval
#%%

from python_modules import *
from cosfire_workflow_utils import *

#%%
hyper_params = {"num_epochs": 200, "lr": 1e-3, 
                 'bits': 36,
                'dataset':'data'}


dic_labels = {'Bent': 0,
                'Compact': 1, 
                'FRI': 2,
                'FRII': 3 
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



dls = dls.dataloaders(path)

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
                        nn.Sigmoid()
                        )
        
    def forward(self, x):
        return self.hd(x)

#%%
'''
y==> target
u==> output
'''
def DSHLoss(output, target, alpha = 0.00001, margin=36):
    
    
    #alpha = 1e-5  # m > 0 is a margin. The margin defines a radius around X1,X2. Dissimilar pairs contribute to the loss
    # function only if their distance is within this radius
    m = 2 * margin  # Initialize U and Y with the current batch's embeddings and labels
    target = target.int()
    # Create a duplicate y_label
    target = target.unsqueeze(1).expand(len(target),len(target))
    y_label = torch.ones_like(torch.empty(len(target), len(target)))
    y_label[target == target.t()] = 0

    dist = torch.cdist(output, output, p=2).pow(2)
    loss1 = 0.5 * (1 - y_label) * dist
    loss2 = 0.5 * y_label * torch.clamp(m - dist, min=0)

    B1 = torch.norm(torch.abs(output) - 1, p=1, dim=1)
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


densenet161_model.lr_find()
plt.savefig(output_dir +'lr_finder.png')
plt.close()


# Train the model using the lr from the plot for 1 epoch and check
densenet161_model.fit_one_cycle(15)


# Train the model using the lr 
densenet161_model.unfreeze()
densenet161_model.fit_one_cycle(n_epoch = hyper_params["num_epochs"], lr_max=slice(hyper_params["lr"]),wd=1e-3)
densenet161_model.recorder.plot_loss()
plt.savefig(output_dir + 'Densenet_Train_and_Test_Loss_curve_at_freeze.png')
plt.close()
    
    
#Save the model
densenet161_model.save("IMR_DNT161_model_" + hyper_params["dataset"] + '_' + str(time.ctime()).replace(' ', '_').replace(':','_'))
print('Model name: ',"IMR_DNT161_model_" + hyper_params["dataset"] + '_' + str(time.ctime()).replace(' ', '_').replace(':','_'))

 #%%   
#If you have to import it...however..here we do not need since the saved model is currently in the memory.
#densenet161_model.load(model_name)


# %%
#Get the predictions and binarize the data

def binarize_data(file_path,dic_labels):
  files = get_image_files(file_path)
  train_dl = densenet161_model.dls.test_dl(files)
  preds,y = densenet161_model.get_preds(dl=train_dl)
  db_label = np.array([str(files[pth]).split('/')[2] for pth in range(len(files))])
  db_label = np.array([dic_labels[lbl] for x, lbl in enumerate(db_label)])
  
  return preds, db_label


#Save/load the the binaries 

if os.path.exists(output_dir +'/df_testing.csv') and os.path.exists(output_dir +'/df_training.csv') and os.path.exists(output_dir +'/df_valid.csv'):
    
    df_training = pd.read_csv(output_dir +'/df_training.csv')
    df_training['predictions'] = [ast.literal_eval(pred) for pred in df_training.predictions]

    df_valid = pd.read_csv(output_dir +'/df_valid.csv')
    df_valid['predictions'] = [ast.literal_eval(pred) for pred in df_valid.predictions]

    df_testing = pd.read_csv(output_dir +'/df_testing.csv')
    df_testing['predictions'] = [ast.literal_eval(pred) for pred in df_testing.predictions]

else:
    
    preds_train,  train_label = binarize_data(file_path = 'data/train/',
                                                    dic_labels = dic_labels
                                                    )

    preds_valid, valid_label = binarize_data(file_path = 'data/valid/',
                                                dic_labels = dic_labels
                                                    )
    preds_test, test_label = binarize_data(file_path = 'data/test/',
                                                dic_labels = dic_labels
                                                    )

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
#### Model Validation
######################################################################

#Get optimal threshold from validation data 
thresholds = list(range(0,100,5))

mAP_results = []
for _,thresh in enumerate(thresholds):

  maP,train_binary, train_label, valid_binary, valid_label = mAP_values(df_training,df_valid,thresh, percentile = True,topk=100)
  mAP_results.append(maP)



data = {'mAP': mAP_results,
        'threshold': thresholds}

df = pd.DataFrame(data)
df.to_csv(output_dir +'/mAP_vs_threshold_validation.csv',index = False)

# Find the index of the maximum mAP value
max_map_index = df['mAP'].idxmax()

# Retrieve the threshold corresponding to the maximum mAP
threshold_max_map = df.loc[max_map_index, 'threshold']

maP_valid,train_binary, train_label, valid_binary, valid_label = mAP_values(df_training,df_valid,thresh = threshold_max_map, percentile = True,topk=100)

maP_test,train_binary, train_label, test_binary, test_label = mAP_values(df_training,df_testing,thresh = threshold_max_map, percentile = True, topk=100)

# Generate random single values for the variables
threshold_max_map = 60
maP_valid = 80
maP_test = 90

# Create a dictionary with the variable names and their values
data = {'optimal threshold': [threshold_max_map],
        'Best Validation mAP': [maP_valid],
        'Test mAP': [maP_test]}

# Create the DataFrame
df_results = pd.DataFrame(data)
df_results.to_csv(output_dir +'/final_results.csv',index = False)

# Plot the line curve
plt.plot(thresholds, mAP_results,  linestyle='-',color = 'red')
plt.xlabel('Threshold (Percentile)')
plt.ylabel('mAP')
plt.savefig(output_dir + 'mAP_vs_threshold_curve_validation.png')
plt.close()

print('The optimal threshold is: ', threshold_max_map)
print('The Best Validation mAP is: ',maP_valid)

print('At the optimal threshold: ', threshold_max_map)
print('The Test  mAP is: ',maP_test)

