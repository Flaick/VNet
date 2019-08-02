# 3D Segmentation by VNet
## Pipeline           
Phase1：Coarse Segmentation for Kidney AND Tumor(consider two organs as one, 0: background, 1: Kidney AND Tumor) to get a bounding box of Kidney and Tumor. Post processing such as Maximum connected domain is applied to remove the noise.            
    
Phase2：Detail Segmentation for Kidney OR Tumor(0: background, 1:Kidney, 2:Tumor).Post processing such as Maximum connected domain, changing threshold in sigmod(times a weight for the tumor probability map)

## Model
### VNet(Phase 1)         
![image](https://github.com/Flaick/VNet/blob/master/src/VNet.png)            
Fig from the paper <V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation>  

### 2-step Model(Phase 2)
![image](https://github.com/Flaick/VNet/blob/master/src/2-step.png)          
Can be easily modified to 2-step model.

## Dataset
200 CT cases pre-processed by clip operation to range (-250,250) from KiTS19 competition.

## Loss Function
Jointly use
1. CE loss         
2. Dice loss        
3. Focal loss      
Reference:   
https://blog.csdn.net/m0_37477175/article/details/83004746
# Metrics
Dice:
Results of Phase1:           
![image](https://github.com/Flaick/VNet/blob/master/src/re1.jpg)            
Results of Phase2:       
![image](https://github.com/Flaick/VNet/blob/master/src/re2.jpg)
