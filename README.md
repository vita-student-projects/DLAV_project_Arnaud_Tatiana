# DLAV Phase 3 - Trajectory Prediction Model
DLAV_phase_1.ipynb implements an end-to-end model that predicts future trajectories (60x3) based on:
- Camera RGB image (200x300x3)
- Driving command (forward, left or right)
- Vehicle motion history (sdc_history_feature; 21x3) 
- Depth camera map (200x300x1) 

## Model Architecture

The model is based on a modular architecture. Each of the four input is first processed by an tailored encoder. The extracted features are then merged using a decoder, which predicts the future trajectory of the ego vehicle. Each class module is described below.

**DrivingDataset class (DrivingDatasetPLUS2)**  
Handles input data processing and augmentation. Camera images are resized to 224x224 pixels, the standard input size for ResNet CNN architecture.
If augment=True applies ColorJitter, RandomHorizontalFlip, RandomAffine, and GaussianBlur.
Computes velocity magnitude from past positions and concatenates them with x, y positions into a [21, 3] tensor.
Augmentation includes random noise in position and velocity, and random occlusions of 3 timesteps in hystory data.



**CommandEncoder class**  
Encodes the string driving commands (forward, left, right) into a learnable vector representation.
Note: This functionality is implemented but may not be working as expected.

**CameraEncoder (Spatial Encoder)**  
Extracts visual features using a ResNet18 architecture and transforms them into a compact, informative vector.
- Uses an 18-layer CNN to process camera input (224x224x3) and output a 1x512 vector (per batch). CNN perform well for tasks that involve grid-like data such as picture, especially where local patterns and spatial hierarchies matter. It is thus a great way to detect meaningful visual information such as other cars, lights, road mapping etc. 
- Since ResNet18 performs well and was trained on large dataset (ImageNet), this training can be transfered to our specific task. Indeed, the ImageNet and nuPlan dataset are similar enough for the low-level features detected in the first layers to be meaningful in both: the first layers weights can thus be frozen. The end layers must however be trained to efficiently capture features specific to the dataset. The final classification layer is replaced with a fully connected layer to map the CNN output (1x512) to the model's required output size (1x256). 


**HistoryEncoder (Temporal Encoder)**  
Transforms a sequence of past vehicle states into a meaningful vector using a Transformer.  
-To preserve the order of the sequence, a learnable positional embedding is added to the projected input
- Transformer blocks are great at handling sequential data, which makes this type of architecture a first choice considering the temporal dimension of the positions. Each layer consists of a multi-head attention layer, residual connections and normalizations and FFN layers. 
- Before passing through the transformer block, history vectors are projected into a higher-dimensional space using a linear layer in order to allow the model for more flexibility and expression. A learned positional embedding is then added to preserve temporal order.
- For simplicity, only the final timestep's output is used. Since it has been through several layers of multi-head attention blocks, it is assumed that all the useful information of the timeserie can be summarized in the final timestep's output.

**AttentionDecoderV1 (Multimodal Fusion Decoder)**  
Combines camera, history, command, and depth features using a Transformer-based attention mechanism to predict future trajectories over 60 timesteps.

- Each input (camera: 1x1024, history: 1×128, command: 1×32, depth: 1×256) is first projected into a shared latent space (1x256) using linear layers.
- The four projected features are stacked and passed through a Transformer encoder to capture dependencies across modalities.
- The outputs are averaged into a single fused vector, which is then passed through a linear layer and reshaped into the final predicted trajectory (60×3).

**DrivingPlannerV1**  
Top-level module that integrates CameraEncoder, HistoryEncoder, CommandEncodera, and DepthEncoder and predicts the trajectory via AttentionDecoder.

## Training

During training, hyperparameters had to be optimized in order to get a performant model: Dropout rate was tested between 0 and 0.5, best is 0.1. Tranformer layer number between 1 and 4, best is 2. Different combinaisons of camera, history and command features latent space dimension, best is (256,128,32) (the trend is to be expected as the camera was the input containing the most information, followed by history and command). Two architecture of decoder were tested, one simple using features concatenations and FFN and one attention-based, which was better. The learning rate had to be decreased to 1e-4 in order for the loss not to explode.  

Different architectures involving auxiliary loss were tested but did not yield better results. Data augmentation was introduced for both camera, depth and history input. Only camera (coupled with history) augmentation was retained as this gave the best performance. Different dimensionality and learning rates were once again tested, the best case involing increased camera and decoder latent dimensions and a decreasing learning rate schedule.  

In order to better scale to real world data, additional data augmentation methods were used. These included picture normalization, saturation, blurring, etc, as well as noise addition in history features. The history input was also enhanced by adding calculated past velocity. This allowed for a wider generalization, yielding better results on real world data.

The training routine is defined in the train function:
- Uses MSE (Mean Squared Error) between the estimated and ground truth future trajectory as the loss function.
- Evaluates ADE (Average Displacement Error) and FDE (Final Displacement Error) during validation. The logger class was not used as the initial metrics printing was deemed satisfactory.
- Optimized with Adam (Adaptive Moment Estimation) and a learning rate of 1e-4.  
Best ADE Achieved on validation dataset: 1.59

### Running instructions:  
The training routine is defined in the train function. Run all the necessary cells (e.g. DrivingDataset, model architecture,  etc.) to view the results of the implementation, including loss, ADE and FDE. Our best model is saved in the drive and later loaded (DrivingPlanner) to run the final visualization cells.  

Since the last models were trained on the SCITAS sever due to limited Google Colab ressources, the Jupyter notebook is not run (although it contains all the neccessary code) and the best model output is available in the log file.  


## Results

The relatively simple architecture used for this project already worked quite well, achieving a final ADE of 1.59. Additional auxiliary perception task did not enhanced the performance, however data augmentation prooved quite effective at generalizing to real-world data.  

When looking more in details however, we noticed the performance of the model depended a lot on the complexity of the trajectory to predict. Four clusters could be identified, representing typical trajectories a vehicle could faced: 
- At stop or stopping; occured 36% of the time, performs well with a mean ADE of 0.8.
- On highway (straight at constant pace); occurs 42% of the time, perfoms well with a mean ADE of 1.2
- From stop; occurs 14% of the time, mixed performance with a mean ADE of 1.9.
- In large turn; occurs 8% of the time, performs the worse with a mean ADE of 3.9.   

As the results show, the model performs well on the two first clusters as they are easy to predict. Indeed, these cases can be easily interpreted with recognisable pattern: quickly decreasing velocity when coming to a stop, lots of cars when stuck in traffic, long large road for highway (prediction is the logic continuation of the history), etc...  
However, when accelarating at a crossroad of at sharp turns, the model performs worse.  

## Conclusion

Altough the model is quite basic, it does perfom quite well for simple situations, which happen the majority of the time. However, an automonous vehicle must be safe in all situation, especially in the complex, more dangerous ones. To improve this performance, the model could be further refined: better implementation of auxiliary losses for both depth map and segmentation (which would also help to understand the stong and weak points of the model), auto-regressor encoder, etc..  
Another way to improve the performance would be to better preprocess the data. The addition of the velocity in the vehicle motion history input did improve the performance, more meaningful features could be manually precalculated and used in the model (accelaration, lateral velocity, etc..). The training dataset could also be rebalanced: about 80% of the dataset represents simple trajectories. These are easier to predict and do not need too much example, whereas the complex trajectories (which are harder to predict and would need more examples) are underrepresented.   

#### Group:
Arnaud Sandell (arnvo) and Tatiana Pina Cunha (TatyPC)
