#%%
from cosfire_workflow_utils import *


input_size = 400
output_size = 36
learning_rate = 0.1
batch_size = 32
epochs = 800
path = r'I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIREdescriptor_best_train_test_file.mat'

path_valid = r'I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIREdescriptor_best_train_valid.mat'
alpha = 1e-5
margin = 36
#%%
#model

class CosfireNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(CosfireNet, self).__init__()
        self.hd = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(64, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.hd(x)


# Data
class CosfireDataset(Dataset):
    def __init__(self, dataframe):
        #self.data = torch.tensor(preprocessing.normalize(dataframe.iloc[:, :-1].values), dtype=torch.float32)
        self.data = torch.tensor(dataframe.iloc[:, :-1].values, dtype=torch.float32)
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

# Plotting the training and validation loss curves
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()



###################################
###       Evaluate              ###
###################################



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

# Plot the line curve
plt.plot(thresholds, mAP_results,  linestyle='-',color = 'red')
plt.xlabel('Threshold (Percentile)')
plt.ylabel('mAP')
plt.show()

print('The optimal threshold is: ', threshold_max_map)
print('The Best Validation mAP is: ',maP)

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

    
print('At the optimal threshold: ', threshold_max_map)
print('The Test mAP is: ',mAP_test)

# %%


thresholds = list(range(0,105,5))

mAP_results = []
for _,thresh in enumerate(thresholds):
      maP,train_binary, train_label, valid_binary, valid_label = mAP_values(train_df,test_df,thresh, percentile = True)
      mAP_results.append(maP)


data = {'mAP': mAP_results,
         'threshold': thresholds}

df = pd.DataFrame(data)

# Find the index of the maximum mAP value
max_map_index = df['mAP'].idxmax()

# Retrieve the threshold corresponding to the maximum mAP
threshold_max_map = df.loc[max_map_index, 'threshold']
maP_valid,_, _, _, _ = mAP_values(train_df,test_df,thresh = threshold_max_map, percentile = True)

# Plot the line curve
plt.plot(thresholds, mAP_results,  linestyle='-',color = 'red')
plt.xlabel('Threshold (Percentile)')
plt.ylabel('mAP')
plt.show()

print('The optimal threshold is: ', threshold_max_map)
print('The Best Testtt mAP is: ',maP_valid)
# %%
