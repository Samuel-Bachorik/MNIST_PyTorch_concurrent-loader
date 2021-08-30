# PyTorch MNIST training with custom loader and model

# Loader
Developed my own dataset loader for 60 000 MNIST training and 10 000 testing images.<br/>
This loader is running concurrently with Python multiprocesing library. -Thanks to this we have massive speed up<br/>
<br/>
Loader can return trainig or testing dataset. <br/>
You can choose it in `get_dataset` function, training to `True` or `False`. `training = True` will return training dataset. <br/>
<br/>
`loader.get_dataset(training_paths, training=True)`<br/>
# Model
AI model created with `PyTorch`.<br/>
The model is being trained on GPU, model's device is set to `"CUDA"` <br/>
<br/>
Model architecture
   - `4x Conv2d` layers with `4x ReLu` activation functions
   - `1x Linear` layer with `LogSoftmax`
# Model accuracy 
Model accuracy is 99.65% with these training parameters
   - Optimizer = Adam
   - Learning rate = 0.0001
   - Epochs = 50
   - Orthogonal weight initialization
   - Model's bias set to zero


You can test model accuracy in `test_accuracy.py`
# Dataset 
You can download dataset here
[Dataset](https://drive.google.com/file/d/1SfBOq8swmSZf2C1X3HV0cDd08TxkqjNq/view?usp=sharing)<br/>
