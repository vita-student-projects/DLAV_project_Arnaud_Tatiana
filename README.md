# DLAV Phase 1 - Trajectory Prediction Model
DLAV_phase_1.ipynb implements an end-to-end model that predicts future trajectories (60x3) based on:
- Camera RGB image (200x300x3)
- Driving command (forward, left or right)
- Vehicle motion history (sdc_history_feature; 21x3)  

## Model Architecture

The model is based on a modular architecture. Each of the three input is first processed by an tailored encoder. The extracted features are then merged using a decoder, which predicts the future trajectory of the ego vehicle. Each class module is described below.

**DrivingDataset class**  
Handles input data processing and augmentation. Camera images are resized to 224x224 pixels, the standard input size for ResNet CNN architecture.

**CommandEncoder class**  
Encodes the string driving commands (forward, left, right) into a learnable vector representation.
Note: This functionality is implemented but may not be working as expected.

**CameraEncoder (Spatial Encoder)**  
Extracts visual features using a ResNet18 architecture and transforms them into a compact, informative vector.
- Uses an 18-layer CNN to process camera input (224x224x3) and output a 1x512 vector (per batch). CNN perform well for tasks that involve grid-like data such as picture, especially where local patterns and spatial hierarchies matter. It is thus a great way to detect meaningful visual information such as other cars, lights, road mapping etc. 
- Since ResNet performs well and was trained on large dataset (ImageNet), this training can be transfered to our specific task. Indeed, the ImageNet and nuPlan dataset are similar enough for the low-level features detected in the first layers to be meaningful in both: the first layers weights can thus be frozen. The end layers must however be trained to efficiently capture features specific to the dataset. The final classification layer is replaced by a fully connected layer mapping the CNN output to the model's required output size (1x256).  

**HistoryEncoder (Temporal Encoder)**  
Transforms a sequence of past vehicle states into a meaningful vector using a Transformer.  
- Transformer blocks are great at handling sequential data, which makes this type of architecture a first choice considering the temporal dimension of the positions. Each layer consists of a multi-head attention layer, residual connections and normalizations and FFN layers.
- Before passing through the transformer block, history vectors are projected into a higher-dimensional space using a linear layer in order to allow the model for more flexibility and expression. A learned positional embedding is then added to preserve temporal order.
- For simplicity, only the final timestep's output is used. Since it has been through several layers of multi-head attention blocks, it is assumed that all the useful information of the timeserie can be summarized in the final timestep's output.

**AttentionDecoderV1 (Multimodal Fusion Decoder)**  
Combines camera, history, and command features using a Transformer-based attention mechanism to predict the future trajectory over 60 timesteps.
- Before undergoing attention, the encoded extracted features for the camera (1x256), history (1x128), and command (1x32) are each projected into a shared latent space (1x128).
- A cross-attention block process the three inputs, the outputs are averaged into a single fused vector. This allows the model to extract information while having an understanding of the dependencies between the different inputs. A final linear layer flattens this vector into the predicted future trajectory dimension space (60x3).

**DrivingPlannerV1**  
Top-level module that integrates CameraEncoder, HistoryEncoder, and CommandEncoder, and decodes via AttentionDecoder.

## Training

During training, hyperparameters had to be optimized in order to get a performant model: Dropout rate was tested between 0 and 0.5, best is 0.1. Tranformer layer number between 1 and 4, best is 2. Different combinaisons of camera, history and command features latent space dimension, best is (256,128,32) (the trend is to be expected as the camera was the input containing the most information, followed by history and command). Two architecture of decoder were tested, one simple using features concatenations and FFN and one attention-based, which was better. The learning rate had to be decreased to 1e-4 in order for the loss not to explode.

The training routine is defined in the train function:
- Uses MSE (Mean Squared Error) between the estimated and ground truth future trajectory as the loss function.
- Evaluates ADE (Average Displacement Error) and FDE (Final Displacement Error) during validation. The logger class was not used as the initial metrics printing was deemed satisfactory.
- Optimized with Adam (Adaptive Moment Estimation) and a learning rate of 1e-4.  
Best ADE Achieved on validation dataset: 1.965

### Running instructions:  
The training routine is defined in the train function. Run all the necessary cells (e.g. DrivingDataset, model architecture,  etc.) to view the results of the implementation, including loss, ADE and FDE. Our best model is saved in the drive and later loaded (DrivingPlanner) to run the final visualization cells.