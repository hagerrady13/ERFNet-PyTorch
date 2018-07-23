# ERFNET-PyTorch
A PyTorch implementation for [ERFNet](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf), for Semantic Segmentation on Pascal VOC.

### Project Structure:
```
├── agents
|  └── erfnet.py # the main training agent 
├── graphs
|  └── models
|  |  └── erfnet.py  # model definition for semantic segmentation
|  |  └── erfnet_imagenet.py  # model definition for imagenet
|  └── losses
|  |  └── loss.py # contains the cross entropy 
├── datasets  # contains all dataloaders for the project
|  └── voc2012.py # dataloader for Pascal Voc dataset
├── data
├── utils # utilities folder containing metrics , config parsing, etc
|  └── assets
├── main.py
├── run.sh
```

### Data Preparation:
Pascal Voc 2012 data

### Model:
We are using the same model architecture as given in the paper.

![alt text](./utils/assets/erfNet.PNG "ERFNet model")


### Experiment configs:
```
- Input size: 256x256x3
- Batch size: 64
- Learning rate: 5e-4
- learning_rate_patience: 100
- Betas for Adam: 0.5 and 0.999
- Number of epochs: 150
```
### Usage:
- To run the project, you need to add your configurations into the folder configs/. An example of the configurations that should be passed can be found here
- ```sh run.sh ```
- To run on a GPU, you need to enable cuda in the config file.

### Results:

**Segmented Images after training**:

### Requirements:
- Pytorch: 0.4.0
- torchvision: 0.2.1
- tensorboardX: 1.2


### References:
- PyTorch official implementation: https://github.com/Eromera/erfnet_pytorch 

### License:
This project is licensed under MIT License - see the LICENSE file for details.


