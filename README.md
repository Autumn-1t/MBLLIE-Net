# MBLLIE-Net
Multi-Branch Low-Light Image Iterative Enhancement Network


## Model Architecture
![image](https://github.com/Autumn-1t/MBLLIE-Net/blob/main/image/Fig.png) 
The overview of the proposed MBLLIE-Net.


### Install ###
- python>=3.6
- torch>=1.10
- torchvision>=0.11
- opencv-python>=4.5
- scikit-image==0.17.2
- lpips==0.1.4
- pillow>=8.4.0

### Dataset ###
You can refer to the following links to download the datasets
- [LOL-v1](https://daooshee.github.io/BMVC2018website/)
- [LOLv2-Synthetic] comes from [LOL](https://daooshee.github.io/BMVC2018website/)
- [TT100K-re](https://drive.google.com/drive/folders/123ggr-KPGoMDNyc3SEiq95XVODAYGRAu?usp=drive_link)

### dataset Structure ###
//For training and testing data, we recommend this structure.     

*├─datasets   
**└─dataset   
***└─train   
****├─input    
****├─target   
***└─test  
****├─input    
****├─target   


### Train ###
All log files during training will be saved to `./checkpoints`.

*First, prepare training data and test data as suggested by '### dataset Structure ###'

**Second, you can check '''./options/base_options.py''' to set appropriate training configuration

***Lastly, You can run '''train.py'''


### Test ###
All test results will be saved to `./results`.

*First, Put the low-light image and label image into './datasets/training_data/testlow' and './datasets/training_data/testhigh' ,
#To accommodate unpaired test data, our code can allow for label images not present#

** Then, You can run '''test.py'''
