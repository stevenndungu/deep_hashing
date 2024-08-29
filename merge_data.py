#%%
import pandas as pd
import re

layer_vsn = 'v4_layers'
df_base = pd.read_csv(f'results_folder/model_selection_train_valid_and_test_13062024_{layer_vsn}_1.csv')
print('Original size: ', df_base.shape)
df_duplicates = df_base[df_base.duplicated(subset=['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha', 'margin', 'l1_reg', 'l2_reg'], keep=False)]
df_base = df_base.drop_duplicates(subset=['input_size','output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'])
print('De-duplicated size: ', df_base.shape)
df_base = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg','mAP_valid', 'mAP_test', 'mAP_valid_zero', 'mAP_test_zero','mAP_valid_abs_values', 'mAP_test_abs_values']]

#%%
for i in range(2,27):
    print('num: ', i)
    df_temp = pd.read_csv(f'results_folder/model_selection_train_valid_and_test_13062024_{layer_vsn}_{i}.csv')
    
    print('original size: ',df_temp.shape)
    df_temp = df_temp.drop_duplicates(subset=['input_size','output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'])
    print('De-duplicated size: ', df_temp.shape)
    df_temp = df_temp[['output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg','mAP_valid', 'mAP_test', 'mAP_valid_zero', 'mAP_test_zero','mAP_valid_abs_values', 'mAP_test_abs_values']]
    df_temp.columns = ['output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg',f'mAP_valid_{i}', f'mAP_test_{i}', f'mAP_valid_zero_{i}', f'mAP_test_zero_{i}',f'mAP_valid_abs_values_{i}', f'mAP_test_abs_values_{i}']
  
    keys = ['output_size', 'learning_rate', 'batch_size', 'alpha', 'margin', 'l1_reg', 'l2_reg']
    df_base = pd.merge(df_base, df_temp, on=keys, suffixes=('_df1', '_df2'))
    print('df_base merged size: ', df_base.shape)
    #break
    
    
    

#%%

df_base.to_csv(f'./results_folder/merged_validation_runs_all_13062024_{layer_vsn}.csv', index=False)

valid_columns = [col for col in df_base.columns if re.match(r'mAP_valid(?:_\d+)?$', col)]
df_base_valid = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'] + valid_columns]
df_base_valid.to_csv(f'./results_folder/merged_validation_runs_wide_format_13062024_{layer_vsn}.csv', index=False)

test_columns = [col for col in df_base.columns if re.match(r'mAP_test(?:_\d+)?$', col)]
df_base_test = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'] + test_columns]
df_base_test.to_csv(f'./results_folder/merged_test_runs_wide_format_13062024_{layer_vsn}.csv', index=False)

# valid_columns = [col for col in df_base.columns if re.match(r'mAP_valid(?:_\d+)?$', col)]
# df_base_valid = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'] + valid_columns]
# df_base_valid.to_csv(f'./results_folder/merged_valid_runs_wide_format_13062024_{layer_vsn}.csv', index=False)

valid_columns_abs = [col for col in df_base.columns if re.match(r'mAP_valid_abs_values(?:_\d+)?$', col)]
df_base_valid_abs = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'] + valid_columns_abs]
df_base_valid_abs.to_csv(f'./results_folder/merged_validation_runs_wide_format_abs_13062024_{layer_vsn}.csv', index=False)

test_columns_abs = [col for col in df_base.columns if re.match(r'mAP_test_abs_values(?:_\d+)?$', col)]
df_base_test_abs = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'] + test_columns_abs]
df_base_test_abs.to_csv(f'./results_folder/merged_test_runs_wide_format_abs_13062024_{layer_vsn}.csv', index=False)

valid_columns_zero = [col for col in df_base.columns if re.match(r'mAP_valid_zero(?:_\d+)?$', col)]
df_base_valid_zero = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'] + valid_columns_zero]
df_base_valid_zero.to_csv(f'./results_folder/merged_validation_runs_wide_format_zero_13062024_{layer_vsn}.csv', index=False)

test_columns_zero = [col for col in df_base.columns if re.match(r'mAP_test_zero(?:_\d+)?$', col)]
df_base_test_zero = df_base[['input_size', 'output_size', 'learning_rate', 'batch_size',  'alpha', 'margin','l1_reg', 'l2_reg'] + test_columns_zero]
df_base_test_zero.to_csv(f'./results_folder/merged_test_runs_wide_format_zero_13062024_{layer_vsn}.csv', index=False)



# %%
