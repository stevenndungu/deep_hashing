#%%
import glob
import os, random
import torch
import torch.nn as nn
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
os.environ['PYTHONHASHSEED'] = 'python'
from IPython.display import Markdown, display
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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

reproducibility_requirements()


# Hyperparameters
parser = argparse.ArgumentParser(description='COSFIRENet Training and Evaluation')
parser.add_argument('--data_path', type=str, default= "G:/My Drive/cosfire/COSFIREdescriptor.mat", help='Path to the COSFIREdescriptor.mat file')
parser.add_argument('--input_size', type=int, default=200, help='Input size of the Descriptors')
parser.add_argument('--output_size', type=int, default=36, help='Output size of the COSFIRENet')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=800, help='Number of epochs')
parser.add_argument('--alpha', type=int, default= 1e-5)
parser.add_argument('--props', type=int, default=0.1)


args = parser.parse_args()
input_size = args.input_size
output_size = args.output_size
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
data_path = args.data_path
alpha = args.alpha
props = args.props


def SimplifiedTopMap(rB, qB, retrievalL, queryL, topk):
  '''
    rB - binary codes of the training set - reference set,
    qB - binary codes of the query set,
    retrievalL - labels of the training set - reference set, 
    queryL - labels of the query set, and 
    topk - the number of top retrieved results to consider.

    rB = train_binary
    qB = valid_binary
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



def mAP_values(df_training,df_testing, thresh = 0.5, percentile = True):
    if percentile:
        train_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(df_training.predictions)])
        valid_binary = np.array([((out >= np.percentile(out,thresh))*1).tolist()  for _, out in enumerate(df_testing.predictions)])
    else:
        train_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(df_training.predictions)])
        valid_binary = np.array([((out >= thresh) * 1).tolist() for _, out in enumerate(df_testing.predictions)])

    train_label = label_binarize(df_training.label_code, classes=[0, 1, 2,3])
    valid_label = label_binarize(df_testing.label_code, classes=[0,1, 2,3])

    rB = train_binary
    qB = valid_binary
    retrievalL = train_label
    queryL = valid_label
    topk = 100
    mAP = SimplifiedTopMap(rB, qB, retrievalL, queryL, topk)
  
    return np.round(mAP,4), train_binary, train_label, valid_binary, valid_label


def load_data(path):
       
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
   df_training = pd.concat([df0, df1, df2, df3], ignore_index=True)

   df0 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][0])
   df0['label'] = 'FRI'
   df1 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][1])
   df1['label'] = 'FRII'
   df2 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][2])
   df2['label'] = 'Bent'
   df3 = pd.DataFrame(data['COSFIREdescriptor']['testing'][0][0][0][0][3])
   df3['label'] = 'Compact'
   df_testing = pd.concat([df0, df1, df2, df3], ignore_index=True)
  
   # Rename the columns:
   column_names = ["descrip_" + str(i) for i in range(1, 201)] + ["label_code"]
   df_training.columns = column_names
   df_testing.columns = column_names

   dic_labels = { 'Bent':2,
                  'Compact':3,
                     'FRI':0,
                     'FRII':1
               }
   df_training['label_name'] = df_training['label_code'].map(dic_labels)
   df_testing['label_name'] = df_testing['label_code'].map(dic_labels)


   df_training_new = pd.concat([df_training,df_testing], ignore_index=True)

   train_label_code = df_training['label_name']
   valid_label_code = df_testing['label_name']

   df_training.drop('label_code', axis=1, inplace=True)
   df_testing.drop('label_code', axis=1, inplace=True)

   return df_training, df_testing, train_label_code, valid_label_code, df_training_new

class CosfireNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(CosfireNet, self).__init__()
        self.hd = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.hd(x)

class CosfireDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def DSHLoss(u, y, alpha):
    #alpha = 1e-5  # m > 0 is a margin. The margin defines a radius around X1,X2. Dissimilar pairs contribute to the loss
    # function only if their distance is within this radius
    m = 2 * output_size  # Initialize U and Y with the current batch's embeddings and labels
    y = y.int()
    # Create a duplicate y_label
    y = y.unsqueeze(1).expand(len(y),len(y))
    #y = torch.Tensor(np.tile(np.array([np.array(Y)]).transpose(), (1, len(y))))
    y_label = torch.ones_like(torch.empty(len(y), len(y)))
    y_label[y == y.t()] = 0

    dist = torch.cdist(u, u, p=2).pow(2)
    loss1 = 0.5 * (1 - y_label) * dist
    loss2 = 0.5 * y_label * torch.clamp(m - dist, min=0)

    B1 = torch.norm(torch.abs(u) - 1, p=1, dim=1)
    B2 = B1.unsqueeze(1).expand(len(y), len(y))

    loss3 = (B2 + B2.t()) * alpha
    minibatch_loss = torch.mean(loss1 + loss2 + loss3)
    return minibatch_loss



# Grid search parameters
learning_rate_values = [0.1, 0.01, 0.001, 0.0001]
alphas = [1e-3,1e-4,1e-5]


def run():
    for learning_rate in learning_rate_values:
        for alpha in alphas:

            path = args.data_path
            df_training, df_testing, train_label_code, valid_label_code, _ = load_data(path)
            
            train_df, valid_df = train_test_split(df_training, test_size=props, random_state=42)
            train_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
            valid_df.rename(columns = {'label_name': 'label_code'}, inplace = True)

            # Create DataLoader for training set
            train_dataset = CosfireDataset(train_df)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Create DataLoader for validation set
            val_dataset = CosfireDataset(valid_df)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Instantiate CosfireNet
            model = CosfireNet(input_size, output_size)
            
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

            # Lists to store training and validation losses
            train_losses = []
            val_losses = []

            # Variables to keep track of the best model
            # best_val_loss = float('inf')
            # best_model_path = 'best_model1.pth'

            # Training loop
            for _ in tqdm(range(epochs), desc='Training Progress', leave=True):
                model.train()
                total_train_loss = 0.0
                for _, (inputs, labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    u = model(inputs)
                    loss = DSHLoss(u = u, y=labels, alpha = alpha)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                scheduler.step()

                # Calculate average training loss
                average_train_loss = total_train_loss / len(train_dataloader)
                train_losses.append(average_train_loss)

                # Validation loop
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for val_inputs, val_labels in val_dataloader:
                        val_outputs = model(val_inputs)
                        val_loss = DSHLoss(u = val_outputs, y=val_labels, alpha = alpha)
                        total_val_loss += val_loss.item()

                # Calculate average validation loss
                average_val_loss = total_val_loss / len(val_dataloader)
                val_losses.append(average_val_loss)

                # Save the model if it is the best so far
               #  if average_val_loss < best_val_loss:
               #      best_val_loss = average_val_loss
               #      torch.save(model.state_dict(), best_model_path)



            ###################################
            ###       Evaluate              ###
            ###################################

            #model = CosfireNet(input_size, output_size)

            # Load the best model
            # best_model_path = 'best_model1.pth'
            # model.load_state_dict(torch.load(best_model_path))
            model.eval()

            valid_dataset = CosfireDataset(valid_df)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            train_dataset = CosfireDataset(train_df)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            # Lists to store predictions
            predictions = []

            # Perform predictions on the train set
            with torch.no_grad():
                for train_inputs, _ in tqdm(train_dataloader, desc='Predicting', leave=True):
                    train_outputs = model(train_inputs)
                    predictions.append(train_outputs.numpy())

            # Flatten the predictions
            flat_predictions_train = [item for sublist in predictions for item in sublist]

            # Append predictions to the df_train DataFrame
            train_df['predictions'] = flat_predictions_train
            
            #################################################################

            # Lists to store predictions
            predictions = []

            # Perform predictions on the valid set
            with torch.no_grad():
                for valid_inputs, _ in tqdm(valid_dataloader, desc='Predicting', leave=True):
                    valid_outputs = model(valid_inputs)
                    predictions.append(valid_outputs.numpy())

            # Flatten the predictions
            flat_predictions_valid = [item for sublist in predictions for item in sublist]

            # Append predictions to the valid_df DataFrame
            valid_df['predictions'] = flat_predictions_valid
            
            ################################################################


            thresholds = list(range(0,105,5))#[30, 50, 55, 65, 70, 85, 90]#)
            #thresholds = np.linspace(0.1, 1, 10).tolist()
            mAP_results = []
            for _,thresh in enumerate(thresholds):
                maP,train_binary, train_label, valid_binary, valid_label = mAP_values(train_df,valid_df,thresh, percentile = True)
                mAP_results.append(maP)


            data = {'mAP': mAP_results,
                    'threshold': thresholds}

            df = pd.DataFrame(data)
            
            # Find the index of the maximum mAP value
            max_map_index = df['mAP'].idxmax()

            # Retrieve the threshold corresponding to the maximum mAP
            threshold_max_map = df.loc[max_map_index, 'threshold']
            maP,train_binary, train_label, valid_binary, valid_label = mAP_values(train_df,valid_df,thresh = threshold_max_map, percentile = True)

            print('The mAP is: ',maP)

          

            results = {
                'input_size': input_size,
                'output_size': output_size,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'mAP': maP,
                'alpha': alpha,
                'model_type' : 'model_v5',
                "props": props
               
                
            }
            results_df = pd.DataFrame([results])
            
            df = pd.read_csv("results.csv")
            results_df = pd.concat([df, results_df], ignore_index=True)
            results_df.to_csv('results.csv', index=False)


if __name__ == '__main__' :
    run()
    
    
    


# %%

