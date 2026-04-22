# Adaptive and Diversity Aware TM-FAL

## 2. Prepare datasets
Download and prepare Fed-ISIC dataset following:
https://github.com/JiayiChen815/FEAL

## 3. Train
CUDA_VISIBLE_DEVICES=0 python main_cls_ours.py --dataset FedISIC --al_method ours --query_model both --query_ratio 0 --budget 500 --al_round 5 --max_round 100 --batch_size 32 --base_lr 5e-4 --kl_weight 1e-2 --display_freq 20

here  if you want to run tmfal use the following command 

!CUDA_VISIBLE_DEVICES=0 python main_cls_ours.py --dataset FedISIC --al_method ours2 --query_model both --query_ratio 0 --budget 500 --al_round 5 --max_round 100 --batch_size 32 --base_lr 5e-4 --kl_weight 1e-2 --display_freq 20
and if you want to run the novel approach use the following command 
!CUDA_VISIBLE_DEVICES=0 python main_cls_ours.py --dataset FedISIC --al_method novel --query_model both --query_ratio 0 --budget 500 --al_round 5 --max_round 100 --batch_size 32 --base_lr 5e-4 --kl_weight 1e-2 --display_freq 20

or if you want to run on kaggle use following
============================================================================================================================================================================

Running TMFAL on fed isic dataset:

To run the code for Temporal model based federated active medical image classification:
First get the code for the TMFAL(fedisic) by clicking on  "add input" and insert the below link and get the code folder:
https://www.kaggle.com/datasets/varunramawat/tmfal-code

now get the fed isic dataset  by clicking on add input and insert the below link :
https://www.kaggle.com/datasets/varunramawat/fedisicdataset

now write these commands in each cell 
!cp -r /kaggle/input/datasets/varunramawat/tmfal-code/"TM-FAL-main (copy)" /kaggle/working/ 

%cd /kaggle/working/"TM-FAL-main (copy)

!CUDA_VISIBLE_DEVICES=0 python main_cls_ours.py --dataset FedISIC --al_method ours2 --query_model both --query_ratio 0 --budget 500 --al_round 5 --max_round 100 --batch_size 32 --base_lr 5e-4 --kl_weight 1e-2 --display_freq 20

now run all the cells using a t4 gpu

if you want to check the logs of the above implementation you could check them on the following link
https://www.kaggle.com/code/varunramawat/vc-implementation


============================================================================================================================================================================

Running proposed novel idea on fed isic dataset: 

To run the code for Adaptive and diversity aware Temporal model based federated active medical image classification:
First get the code  by clicking on  "add input" and insert the below link and get the code folder:
https://www.kaggle.com/datasets/varunramawat/tmfal-novel-implementation


now get the fed isic dataset  by clicking on add input and insert the below link :
https://www.kaggle.com/datasets/varunramawat/fedisicdataset

now write these commands in each cell 
!cp -r /kaggle/input/datasets/varunramawat/tmfal-novel-implementation/TMFAL_implementation /kaggle/working/ 

%cd /kaggle/working/TMFAL_implementation/"TM-FAL-main (copy)"/

!CUDA_VISIBLE_DEVICES=0 python main_cls_ours.py --dataset FedISIC --al_method novel --query_model both --query_ratio 0 --budget 500 --al_round 5 --max_round 100 --batch_size 32 --base_lr 5e-4 --kl_weight 1e-2 --display_freq 20

now run all the cells using a t4 gpu

if you want to check the logs of the novel idea you could check them on the following link
https://www.kaggle.com/code/varunramawat/notebook3aaaa45b7a
============================================================================================================================================================================
