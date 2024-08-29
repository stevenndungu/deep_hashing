
#%%
import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

#%%
#output_dir = f'MLP_test_2' 

# if not os.path.exists(output_dir):
#    os.mkdir(output_dir)
# else:
#    print(f"The directory {output_dir} already exists.")

# print(f'MLP_test')


class MLP(nn.Module):
   def __init__(self, input_dim, num_classes):
      super(MLP, self).__init__()
      self.hd = nn.Sequential(
         nn.Linear(input_dim, 200),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes))
          

# class MLP(nn.Module):
#    def __init__(self, input_dim, num_classes):
#       super(MLP, self).__init__()
#       self.hd = nn.Sequential(
#          nn.Linear(input_dim, num_classes))
           
           
      
   def forward(self, x):
      return self.hd(x)

input_dim = 372
num_classes = 4
learning_rate = 0.01
num_epochs = 50



valid_accuracy = []
test_accuracy = []

# for num in range(1,27):


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

   #select the optimal number of columns from the classification paper.#Get the optimal 372 descriptors only
   column_list = [f'descrip_{i}' for i in range(1, 373)] + ['label_code']
   df_train = df_train[column_list]
   df_test = df_test[column_list]

   dic_labels = { 'Bent':2,
                  'Compact':3,
                     'FRI':0,
                     'FRII':1
               }


   df_train['label_code'] = df_train['label_code'].map(dic_labels)
   df_test['label_code'] = df_test['label_code'].map(dic_labels)


   return df_train, df_test



num = 22



df_training, df_testing = get_data(rf"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptors\descriptor_set_{num}_train_test.mat")
df_train = preprocessing.normalize(df_training.iloc[:, :-1].values)
y_train = df_training.iloc[:, -1].values

df_test = preprocessing.normalize(df_testing.iloc[:, :-1].values)
y_test = df_testing.iloc[:, -1].values

_, valid_df = get_data(rf"I:\My Drive\deep_learning\deep_hashing\deep_hashing_github\COSFIRE_26_valid_hyperparameters_descriptors\descriptors\descriptor_set_{num}_train_valid.mat")

df_valid = preprocessing.normalize(valid_df.iloc[:, :-1].values)
y_valid = valid_df.iloc[:, -1].values

# Create PyTorch datasets
X_train = torch.tensor(df_train, dtype=torch.float32)
X_train = X_train.clone().detach().requires_grad_(True)
y_train = torch.tensor(np.array(y_train), dtype=torch.long)
y_train = y_train.clone().detach()

X_valid = torch.tensor(df_valid, dtype=torch.float32)
X_valid = X_valid.clone().detach().requires_grad_(True)
y_valid = torch.tensor(np.array(y_valid), dtype=torch.long)
y_valid = y_valid.clone().detach()


X_test = torch.tensor(df_test, dtype=torch.float32)
X_test = X_test.clone().detach().requires_grad_(True)
y_test = torch.tensor(np.array(y_test), dtype=torch.long)
y_test = y_test.clone().detach()

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
valid_dataset = TensorDataset(X_valid, y_valid)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# Create the model
model = MLP(input_dim, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)


# Train the model
train_losses = []
valid_losses = []
test_losses = []
trnvalid_losses =[]

for epoch in range(num_epochs):
   model.train()
   train_loss = 0.0
   
   for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * inputs.size(0)
   
   train_loss /= len(train_loader.dataset)
   train_losses.append(train_loss)
      
   model.eval()
   valid_loss = 0.0
   
   with torch.no_grad():
      for input_labels, val_labels in valid_loader:
            val_outputs = model(input_labels)
            val_loss = criterion(val_outputs, val_labels)
            valid_loss += val_loss.item() * inputs.size(0)
   
   valid_loss /= len(valid_loader.dataset)
   valid_losses.append(valid_loss)

   test_loss_acc = 0.0
   
   with torch.no_grad():
      for input_labels, test_labels in test_loader:
            test_outputs = model(input_labels)
            test_loss = criterion(test_outputs, test_labels)
            test_loss_acc += test_loss.item() * inputs.size(0)
   
   test_loss_acc /= len(test_loader.dataset)
   test_losses.append(test_loss_acc)

   
   # train_valid_loss = 0.0
   
   # with torch.no_grad():
   #    for input_labels, trnval_labels in train_loader:
   #          trn_val_outputs = model(input_labels)
   #          trn_val_loss = criterion(trn_val_outputs, trnval_labels)
   #          train_valid_loss += trn_val_loss.item() * inputs.size(0)
   


   # train_valid_loss /= len(train_loader.dataset)
   # trnvalid_losses.append(train_valid_loss)
   
   #print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")


# Evaluate the model

model.eval()

predictions = []
true_labels = []
outputs_raw = []


with torch.no_grad():
   for inputs, labels in train_loader:
      outputs = model(inputs)
      outputs = torch.nn.functional.softmax(outputs, dim=1) #pass through softmax
      _, predicted = torch.max(outputs.data, 1)
      predictions.extend(predicted.cpu().numpy())
      outputs_raw.extend(outputs.cpu().numpy())
      true_labels.extend(labels.cpu().numpy())

outputs_raw = np.array(outputs_raw)
outputs_raw[outputs_raw<=4.2449587e-40] = 4.2449587e-40
outputs_raw1 = torch.log(torch.tensor(outputs_raw))
true_labels_test = label_binarize(true_labels, classes=[0, 1, 2,3])
train_losses_values = torch.sum(-outputs_raw1*true_labels_test, axis=1)

accuracy_train = accuracy_score(true_labels, predictions)
print(f"Train Accuracy: {accuracy_train:.4f}")
confusion_matrix(true_labels, predictions)

model.eval()

predictions = []
true_labels = []
outputs_raw = []


with torch.no_grad():
   for inputs, labels in valid_loader:
      outputs = model(inputs)
      outputs = torch.nn.functional.softmax(outputs, dim=1) #pass through softmax
      _, predicted = torch.max(outputs.data, 1)
      predictions.extend(predicted.cpu().numpy())
      outputs_raw.extend(outputs.cpu().numpy())
      true_labels.extend(labels.cpu().numpy())

from sklearn.preprocessing import label_binarize

outputs_raw = np.array(outputs_raw)
outputs_raw[outputs_raw<=4.2449587e-40] = 4.2449587e-40
outputs_raw1 = torch.log(torch.tensor(outputs_raw))
true_labels_test = label_binarize(true_labels, classes=[0, 1, 2,3])
validation_losses_values = torch.sum(-outputs_raw1*true_labels_test, axis=1)



# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 398+1), train_losses_values[:len(validation_losses_values)], label='Train Loss')
# plt.plot(range(1, 398+1), validation_losses_values, label='Valid Loss')
# #plt.plot(range(1, num_epochs+1), trnvalid_losses, label='Train eval Loss')

# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Train and Valid Loss Curves')
# plt.legend()
# plt.show()


accuracy_valid = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy: {accuracy_valid:.4f}")
confusion_matrix(true_labels, predictions)

predictions = []
true_labels = []

with torch.no_grad():
   for inputs, labels in test_loader:
      outputs = model(inputs)
      outputs = torch.nn.functional.softmax(outputs, dim=1) #pass through softmax
      _, predicted = torch.max(outputs.data, 1)
      predictions.extend(predicted.cpu().numpy())
      true_labels.extend(labels.cpu().numpy())

confusion_matrix(true_labels, predictions)

accuracy_test = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy_test:.4f}")

# Print classification report
# print("Classification Report:")
# print(classification_report(true_labels, predictions))

# Plot train and valid loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Valid Loss Curves')
plt.legend()
plt.show()
   # plt.savefig(output_dir + '/Train_valid_curves.png')
   # plt.close()
   # %%
   #valid_accuracy.append(accuracy_valid)
   #test_accuracy.append(accuracy_test)


#    results = {'Acc_valid': accuracy_valid,
#          'Acc_test': accuracy_test
#             }
#    results_df = pd.DataFrame([results])

#    if not os.path.isfile(output_dir + "/results_31052024.csv"):
#       df = pd.DataFrame(columns=['Acc_valid', 'Acc_test'])
#       results_df = pd.concat([df, results_df], ignore_index=True)
#       results_df.to_csv(output_dir + "/results_31052024.csv", index=False)
#    else:
#       df = pd.read_csv(output_dir + "/results_31052024.csv")

#       results_df = pd.concat([df, results_df], ignore_index=True)

#       results_df.to_csv(output_dir + "/results_31052024.csv", index=False)


# results_df['descriptor_num'] = list(range(1,27))
# results_df.to_csv(output_dir +'/results.csv',index = False)



#%%
import os
import shutil

num=1

shutil.copy2('final_model_selection_train_valid_test_v_{num}/model_selection_train_valid_and_test_13062024_v1_layers.csv', 'results_files/model_selection_train_valid_and_test_13062024_v1_layers_{num}.csv')