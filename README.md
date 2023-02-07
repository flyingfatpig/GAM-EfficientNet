

## GAM-EfficientNet

Title: Deep Learning-Assisted Diagnosis of Chronic Atrophic Gastritis in Endoscopy

Description: Add GAM module to EfficientNetV2, and perform the identification of chronic atrophic gastritis.


## Code

```bash
|-- GAM-EfficientNet
 	|-- class_indices.json
 	|-- grad_cam.py
 	|-- main_effi_gam.py
  |-- model.py
  |-- my_dataset.py
  |-- predict.py
  |-- test.py
  |-- train.py
  |-- utils.py
```



- class_indices.json：Category files
- grad_cam.py: Heatmap generation class, GradCAM.
- main_effi_gam.py: Generate heatmap entry function.
- model.py: Implementation of the model, including the implementation of EfficientNetV2, GAM.
- my_dataset.py: Custom management dataset class, including image path, image classification and other information.
- predict.py: After the model is trained, this file can be used to make predictions for a single image.
- test.py: Prediction of all images in the specified directory.
- utils.py: Custom help class. Contains functions such as splitting and getting of dataset.



The pre-trained model used for model training can be get from the following address:

url: https://pan.baidu.com/s/1YmqO4gO6PlZjSINIBxE-Vg  

password: prub

 # 
