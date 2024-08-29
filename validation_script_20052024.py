#%%
from cosfire_workflow_utils import *

# Hyperparameters
parser = argparse.ArgumentParser(description='COSFIRENet Training and Evaluation')
# parser.add_argument('--data_path_valid', type=str, default= f"./descriptors/descriptor_set_{num}_train_valid.mat", help='Path to the Train_valid.mat file')
# parser.add_argument('--data_path', type=str, default= f"./descriptors/descriptor_set_{num}_train_test.mat", help='Path to the Train_test.mat file')
parser.add_argument('--input_size', type=int, default=372, help='Input size of the Descriptors')
parser.add_argument('--output_size', type=int, default=36, help='Output size of the COSFIRENet')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
parser.add_argument('--alpha', type=int, default= 0.001)
parser.add_argument('--margin', type=int, default=36)
parser.add_argument('--num', type=int, default=1)
args = parser.parse_args()
input_size = args.input_size
output_size = args.output_size
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
# data_path = args.data_path
# data_path_valid = args.data_path_valid
alpha = args.alpha
margin = args.margin
num = args.num
print('num: ', num)

data_path_valid = f"./descriptors/descriptor_set_{num}_train_valid.mat" # Path to the Train_valid.mat file
data_path = f"./descriptors/descriptor_set_{num}_train_test.mat" # Path to the Train_test.mat file

output_dir = f'final_model_selection_test_v{num}' 

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    print(f"The directory {output_dir} already exists.")

print(f'final_model_selection_test_v{num}')

#model 8
class CosfireNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(CosfireNet, self).__init__()
        self.hd = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.BatchNorm1d(300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.Tanh(),
            nn.Linear(100, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size),
            nn.Tanh()
        )
    def forward(self, x):
        return self.hd(x)


# Data
class CosfireDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(preprocessing.normalize(dataframe.iloc[:, :-1].values), dtype=torch.float32)
        self.labels = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# loss function
def DSHLoss(u, y, alpha, margin):
    #alpha = 1e-5  # m > 0 is a margin. The margin defines a radius around X1,X2. Dissimilar pairs contribute to the loss
    # function only if their distance is within this radius
    m = 2 * margin  # Initialize U and Y with the current batch's embeddings and labels
    y = y.int()
    # Create a duplicate y_label
    y = y.unsqueeze(1).expand(len(y),len(y))
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

epoch_values = [650]
learning_rate_values = [0.1]
alphas = [0.00001]
margin_values = [36]
props_values = [0]
batch_size_values = [64]

#List of Grid search parameters to iterate over

# learning_rate_values = [0.1, 0.01, 0.001, 0.0001]
# alphas = [1e-03, 1e-4, 1e-5]
# margin_values = [24, 36, 48]
# batch_size_values = [16, 32, 64]


def run():
    for epochs in epoch_values:
        for learning_rate in learning_rate_values:
            for alpha in alphas:
                for margin in margin_values:
                    for batch_size in batch_size_values:

                         #select the optimal number of columns from the classification paper.
                        column_list = [f'descrip_{i}' for i in range(1, 372)] + ['label_code']

                        # Train & Test data
                        #df_training, valid_df = train_test_split(df_training, test_size=props, random_state=42)
                        df_training, df_testing = get_data(data_path)
                        #Get the optimal 372 descriptors only
                        # df_training = df_training[column_list]
                        # df_testing = df_testing[column_list]
                        
                        # Validation data
                        _, valid_df = get_data(data_path_valid)
                        #valid_df = valid_df[column_list]

                        # Verify the data set sizes based on Table 1 of the paper. 
                        print('Train data shape: ', df_training.shape)
                        print('Valid data shape: ', valid_df.shape)                        
                        print('Test data shape: ', df_testing.shape)

                        # Rename label_name column:   
                        df_training.rename(columns = {'label_name': 'label_code'}, inplace = True)
                        valid_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
                        df_testing.rename(columns = {'label_name': 'label_code'}, inplace = True)

                        # DataLoader for training set
                        train_dataset = CosfireDataset(df_training)
                        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        # DataLoader for validation set
                        val_dataset = CosfireDataset(valid_df)
                        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                        # Instantiate CosfireNet
                        model = CosfireNet(input_size, output_size)
                        
                        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

                        # Lists to store training and validation losses
                        train_losses = []
                        train_losses_eval = []
                        val_losses = []
                        # Variables to keep track of the best model
                        
                        # Train the loop
                        for _ in tqdm(range(epochs), desc='Training Progress', leave=True):
                            model.train()
                            total_train_loss = 0.0
                            for _, (train_inputs, labels) in enumerate(train_dataloader):
                                optimizer.zero_grad()
                                train_outputs = model(train_inputs)
                                loss = DSHLoss(u = train_outputs, y=labels, alpha = alpha, margin = margin)
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
                                    val_loss = DSHLoss(u = val_outputs, y=val_labels, alpha = alpha, margin = margin)
                                    total_val_loss += val_loss.item()

                            # Calculate average validation loss
                            average_val_loss = total_val_loss / len(val_dataloader)
                            val_losses.append(average_val_loss)
                    
                        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
                        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.title('Training and Validation Loss Curves')
                        plt.legend()
                        plt.savefig(output_dir + '/Train_valid_curves.png')
                        plt.close()

                        #########################################################################
                        #################     Evaluate                                                  
                        ##########################################################################

                        #model.eval()
                        valid_dataset_eval = CosfireDataset(valid_df)
                        valid_dataloader_eval = DataLoader(valid_dataset_eval, batch_size=batch_size, shuffle=False)

                        train_dataset_eval = CosfireDataset(df_training)
                        train_dataloader_eval = DataLoader(train_dataset_eval, batch_size=batch_size, shuffle=False)

                        test_dataset = CosfireDataset(df_testing)
                        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                        train_dataset_full = CosfireDataset(df_training)
                        train_dataloader_full = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=False)

                        # Lists to store predictions
                        predictions = []

                        # Perform predictions on the train set
                        with torch.no_grad():
                            for train_inputs, _ in tqdm(train_dataloader_eval, desc='Predicting', leave=True):
                                train_outputs = model(train_inputs)
                                predictions.append(train_outputs.numpy())

                        # Flatten the predictions
                        flat_predictions_train = [item for sublist in predictions for item in sublist]
                        
                        # Append predictions to the df_train DataFrame
                        df_training['predictions'] = flat_predictions_train
                        df_training['label_name'] = df_training['label_code'].map(dic_labels_rev)
                        df_training.to_csv(output_dir +'/df_training.csv',index = False)
                        
                        #################################################################
                        
                        predictions = []
                        # Perform predictions on the valid set
                        with torch.no_grad():
                            for valid_inputs, _ in tqdm(valid_dataloader_eval, desc='Predicting', leave=True):
                                valid_outputs = model(valid_inputs)
                                predictions.append(valid_outputs.numpy())
                        # Flatten the predictions
                        flat_predictions_valid = [item for sublist in predictions for item in sublist]
                        # Append predictions to the valid_df DataFrame
                        valid_df['predictions'] = flat_predictions_valid
                        valid_df['label_name'] = valid_df['label_code'].map(dic_labels_rev)
                        valid_df.to_csv(output_dir +'/valid_df.csv',index = False)

  
                        ##########################################################################
                        # Testing
                        ##########################################################################
                        # Perform predictions on the testing set

                        predictions_test = []
                        with torch.no_grad():
                            for test_inputs, _ in tqdm(test_dataloader, desc='Predicting', leave=True):
                                test_outputs = model(test_inputs)
                                predictions_test.append(test_outputs.numpy())

                        # Flatten the predictions
                        flat_predictions_test = [item for sublist in predictions_test for item in sublist]

                        # Append predictions to the df_testing DataFrame
                        df_testing['predictions'] = flat_predictions_test
                        df_testing['label_name'] = df_testing['label_code'].map(dic_labels_rev)
                        df_testing.to_csv(output_dir +'/df_testing.csv',index = False)
                        
                        #####################################################################
                        #####################################################################

                        #thresholds = np.arange(-1, 1, 0.2)
                        thresholds = range(0, 105, 5)

                        
                        mAP_results_valid = []
                        mAP_results_test = []
                        for _,thresh in enumerate(thresholds):
                            
                            maP_valid,_, _, _, _ = mAP_values(df_training,valid_df,thresh = thresh, percentile = True)
                            mAP_test,_, _, _, _ = mAP_values(df_training, df_testing,thresh = thresh, percentile = True)

                            mAP_results_valid.append(maP_valid)
                            mAP_results_test.append(mAP_test)


                        data = {'mAP_valid': mAP_results_valid,
                                'mAP_test': mAP_results_test,
                                'threshold': thresholds}

                        df = pd.DataFrame(data)
                        df.to_csv(output_dir +'/results.csv',index = False)

                       
                        
                        # Find the index of the maximum mAP value
                        max_map_index = df['mAP_valid'].idxmax()

                        # Retrieve the threshold corresponding to the maximum mAP
                        threshold_max_map = df.loc[max_map_index, 'threshold']
                        maP_valid,_, _, _, _ = mAP_values(df_training,valid_df,thresh = threshold_max_map, percentile = False)
                        
                        

                        #####################################################################
                        #####################################################################

                        
                        threshold_max_map = 0
                        maP_valid_zero,_, _, _, _ = mAP_values(df_training,valid_df,thresh = threshold_max_map, percentile = False)
                      
                                            
                        mAP_test_zero,_, _, _, _ = mAP_values(df_training, df_testing,thresh = threshold_max_map, percentile = False)

                         # Plotting
                        plt.figure(figsize=(8, 6))
                        plt.plot(thresholds, mAP_results_valid, label='mAP_valid')
                        plt.plot(thresholds, mAP_results_test, label='mAP_test')
                        plt.xlabel('Threshold')
                        plt.ylabel('mAP')
                        plt.legend()
                        plt.axhline(y = maP_valid_zero, color = 'green') 
                        plt.axhline(y = mAP_test_zero, color = 'r') 
                        plt.savefig(output_dir + '/Maps_curves.png')
                        plt.close()

                        # Create a dictionary with the variable names and their values
                        data = {'optimal threshold': [threshold_max_map],
                                'Best Validation mAP': [maP_valid_zero],
                                'Test mAP': [mAP_test_zero]}

                        # Create the DataFrame
                        df_results = pd.DataFrame(data)
                        df_results.to_csv(output_dir +'/final_results.csv',index = False)

                        results = {
                            'input_size': input_size,
                            'output_size': output_size,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'threshold_max_map': threshold_max_map,
                            'mAP_valid': maP_valid_zero,
                            'mAP_test': mAP_test_zero,
                            'alpha': alpha,
                            'model_type' : 'model_v8',
                            "margin": margin,
                            'data_path_valid' : data_path_valid.split('/')[2],
                            'data_path' : data_path.split('/')[2]
                           
                        }

                        results_df = pd.DataFrame([results])
                        if not os.path.isfile(output_dir + "/model_selection_valid_and_test_25052024.csv"):
                            df = pd.DataFrame(columns=['input_size', 'output_size', 'learning_rate', 'batch_size', 'epochs','threshold_max_map', 'mAP_valid', "mAP_test",'alpha', 'model_type', 'margin','data_path_valid','data_path'])
                            results_df = pd.concat([df, results_df], ignore_index=True)
                            results_df.to_csv(output_dir + "/model_selection_valid_and_test_25052024.csv", index=False)
                        else:
                            df = pd.read_csv(output_dir + "/model_selection_valid_and_test_25052024.csv")
                        
                            results_df = pd.concat([df, results_df], ignore_index=True)

                            results_df.to_csv(output_dir + "/model_selection_valid_and_test_25052024.csv", index=False)

if __name__ == '__main__' :
    run()
    