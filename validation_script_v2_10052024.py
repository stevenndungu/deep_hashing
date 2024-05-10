#%%
from cosfire_workflow_utils import *
#from early_stopping_pytorch import *

output_dir = './model_selection_model_v6_v2/' 
os.mkdir(output_dir)


df = pd.DataFrame(columns=['input_size', 'output_size', 'learning_rate', 'batch_size', 'epochs','threshold_max_map', 'mAP_valid', "mAP_test",'alpha', 'model_type', 'margin'])
df.to_csv(output_dir + "model_selection_valid_and_test_10052024.csv", index=False)

# Hyperparameters
parser = argparse.ArgumentParser(description='COSFIRENet Training and Evaluation')
parser.add_argument('--data_path', type=str, default= "COSFIREdescriptor_best_train_test_file.mat", help='Path to the COSFIREdescriptor.mat file')
parser.add_argument('--data_path_valid', type=str, default= "COSFIREdescriptor_best_train_valid.mat", help='Path to the COSFIREdescriptor_train_valid.mat file')


parser.add_argument('--input_size', type=int, default=400, help='Input size of the Descriptors')
parser.add_argument('--output_size', type=int, default=36, help='Output size of the COSFIRENet')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--alpha', type=int, default= 1e-5)
parser.add_argument('--margin', type=int, default=36)


args = parser.parse_args()
input_size = args.input_size
output_size = args.output_size
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
data_path = args.data_path
data_path_valid = args.data_path_valid
alpha = args.alpha
margin = args.margin


#model

class CosfireNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(CosfireNet, self).__init__()
        self.hd = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
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



#List of Grid search parameters to iterate over
learning_rate_values = [0.1, 0.01, 0.001]
alphas = [1e-03, 1e-4, 1e-5]
margin_values = [24, 36, 48]
batch_size_values = [16, 32, 64]


def run():
    for learning_rate in learning_rate_values:
        for alpha in alphas:
            for margin in margin_values:
                for batch_size in batch_size_values:
                    path = args.data_path
                    path_valid = args.data_path_valid
                    

                    train_df, valid_df = get_data(path_valid) # data_path_valid:COSFIREdescriptor_best_train_valid.mat
                    _, test_df = get_data(path) #data_path: COSFIREdescriptor_best_train_test_file.mat
                    
                    
                    
                    train_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
                    valid_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
                    test_df.rename(columns = {'label_name': 'label_code'}, inplace = True)

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
                    # best_model_path = 'best_model.pth'

                     # initialize the early_stopping object
                    #early_stopping = EarlyStopping(patience=50, verbose=True)

                    # Training loop
                    for epoch in tqdm(range(epochs), desc='Training Progress', leave=True):
                        model.train()
                        total_train_loss = 0.0
                        for _, (inputs, labels) in enumerate(train_dataloader):
                            optimizer.zero_grad()
                            train_outputs = model(inputs)
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
                                val_loss = DSHLoss(u = val_outputs, y = val_labels, alpha = alpha, margin = margin)
                                total_val_loss += val_loss.item()

                        # Calculate average validation loss
                        average_val_loss = total_val_loss / len(val_dataloader)
                        val_losses.append(average_val_loss)

                        # early_stopping needs the validation loss to check if it has decresed, 
                        # and if it has, it will make a checkpoint of the current model
                        #early_stopping(average_val_loss, model)
                    
                        # if early_stopping.early_stop:
                        #     print("Early stopping")
                        #     break

                        # Save the model if it is the best so far
                        #  if average_val_loss < best_val_loss:
                        #      best_val_loss = average_val_loss
                        #      torch.save(model.state_dict(), best_model_path)



                    ###################################
                    ###       Evaluate              ###
                    ###################################

                    #model = CosfireNet(input_size, output_size)

                    # Load the best model
                    # best_model_path = 'best_model.pth'
                    # model.load_state_dict(torch.load(best_model_path))
                    # load the last checkpoint with the best model
                    #model.load_state_dict(torch.load('checkpoint.pt'))
                    model.eval()

                    valid_dataset = CosfireDataset(valid_df)
                    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

                    test_dataset = CosfireDataset(test_df)
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                    train_dataset_full = CosfireDataset(train_df)
                    train_dataloader_full = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=False)

                    # Lists to store predictions
                    predictions_full = []

                    # Perform predictions on the train set
                    with torch.no_grad():
                        for train_inputs_full, _ in tqdm(train_dataloader_full, desc='Predicting', leave=True):
                            train_outputs_full = model(train_inputs_full)
                            predictions_full.append(train_outputs_full.numpy())

                    # Flatten the predictions
                    flat_predictions_train_full = [item for sublist in predictions_full for item in sublist]

                    # Append predictions to the df_train DataFrame
                    train_df['predictions'] = flat_predictions_train_full
                    
                    #################################################################

                    # Lists to store predictions
                    predictions_valid = []

                    # Perform predictions on the valid set
                    with torch.no_grad():
                        for valid_inputs, _ in tqdm(valid_dataloader, desc='Predicting', leave=True):
                            valid_outputs = model(valid_inputs)
                            predictions_valid.append(valid_outputs.numpy())

                    # Flatten the predictions
                    flat_predictions_valid = [item for sublist in predictions_valid for item in sublist]

                    # Append predictions to the valid_df DataFrame
                    valid_df['predictions'] = flat_predictions_valid
                    
                    ################################################################


                    thresholds = list(range(0,105,5))
                    
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
                    maP_valid,_, _, _, _ = mAP_values(train_df,valid_df,thresh = threshold_max_map, percentile = True)

                    
                    ##########################################################################
                    # Testing
                    ##########################################################################

                    # Perform predictions on the testing set
                        # Lists to store predictions
                    predictions_test = []
                    with torch.no_grad():
                        for test_inputs, _ in tqdm(test_dataloader, desc='Predicting', leave=True):
                            test_outputs = model(test_inputs)
                            predictions_test.append(test_outputs.numpy())

                    # Flatten the predictions
                    flat_predictions_test = [item for sublist in predictions_test for item in sublist]

                    # Append predictions to the test_df DataFrame
                    test_df['predictions'] = flat_predictions_test

                    # Perform predictions on the training set
                    predictions_train_full = []
                    with torch.no_grad():
                        for train_inputs, _ in tqdm(train_dataloader_full, desc='Predicting', leave=True):
                            train_outputs = model(train_inputs)
                            predictions_train_full.append(train_outputs.numpy())

                    # Flatten the predictions
                    flat_predictions_train_full = [item for sublist in predictions_train_full for item in sublist]

                    # Append predictions to the df_train DataFrame
                    train_df['predictions'] = flat_predictions_train_full
                    
                    

                    mAP_test,_, _, _, _ = mAP_values(train_df,test_df,thresh = threshold_max_map, percentile = True)        

                    results = {
                        'input_size': input_size,
                        'output_size': output_size,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'threshold_max_map': threshold_max_map,
                        'mAP_valid': maP_valid,
                        'mAP_test': mAP_test,
                        'alpha': alpha,
                        'model_type' : 'model_v6',
                        "margin": margin                    
                        
                    }

                    results_df = pd.DataFrame([results])

                    df = pd.read_csv(output_dir + "model_selection_valid_and_test_10052024.csv")
                    
                    results_df = pd.concat([df, results_df], ignore_index=True)

                    results_df.to_csv(output_dir + "model_selection_valid_and_test_10052024.csv", index=False)
                        
if __name__ == '__main__' :
    run()
    
    

