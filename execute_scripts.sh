#!/bin/bash

# # Set the maximum job count
# max_job_count=30

# # Loop through the script names
# for script in run_*
# do
#     job_count=$(squeue -u p307791 | grep "reg*" | wc -l)

#     if [ "$job_count" -lt "$max_job_count" ]; then
#         # Execute the script
#         sbatch "$script"
#         rm "$script"
#         echo "Job count: $job_count (New job started)"
#     else
#         echo "Job limit of $max_job_count reached. Pausing..."
#         echo "Current job count: $job_count"
#         sleep 5
#     fi

#     # Add a brief delay between iterations to avoid excessive checking
#     sleep 5
# done


# for folder in final_model_selection_train_valid_test_v_trtr_*

#     do 
#        echo "$folder Submitted!"
#        rm -rf "$folder"      

#     done

for script in run_*
    do 
       sbatch "$script"
       echo "$script Submitted!"
       #rm "$script"      

    done





 