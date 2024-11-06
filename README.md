
<div align="center">
    <a href="https://github.com/shahzaib-123/VisionTransfromer-Scratch/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/shahzaib-123/VisionTransfromer-Scratch?color=yellow&label=Project%20Stars&style=for-the-badge"></a>
    <a href="https://github.com/shahzaib-123/VisionTransfromer-Scratch/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/shahzaib-123/VisionTransfromer-Scratch?color=brightgreen&label=issues&style=for-the-badge"></a>
    <a href="https://github.com/shahzaib-123/VisionTransfromer-Scratch/forks"><img alt="GitHub forks" src="https://img.shields.io/github/forks/shahzaib-123/VisionTransfromer-Scratch?color=9cf&label=forks&style=for-the-badge"></a>
</div>
<br>

<div align="center">
    <a href="shahzaib-123" target="_blank">
        <img src="https://github.com/shahzaib-123/VisionTransfromer-Scratch/blob/main/vit.gif" 
        alt="Logo" height="300" width="auto">
    </a>
</div>

<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=4000&pause=5000&background=FFFFFF00&center=true&vCenter=true&multiline=true&width=435&lines=ViT in Action">
</div>

# Vision Transformer (ViT) Implementation in PyTorch
This repository contains a Vision Transformer (ViT) model implemented from scratch using PyTorch and trained on the CIFAR-10 dataset. This project demonstrates the power of ViTs for image classification by breaking down images into sequences of patches and processing them with transformer layers.

## About
The Vision Transformer model processes images as sequences of patches, allowing it to capture complex spatial relationships in a way similar to how transformers handle sequences in NLP tasks. This implementation leverages the CIFAR-10 dataset for training and testing, which consists of 60,000 32x32 color images in 10 classes.

## Project Structure
```
.
├── models/
│   ├── __init__.py
│   ├── msa.py
│   ├── vit.py
│   └── vit_block.py
├── utilities/
│   ├── __init__.py
│   └── UTILS.py
├── data/
├── runs/
├── requirements.sh
├── train.py
├── vit.gif
└──README.md

```
## Installation
To set up this project, you will need  following packages:
- torch
- torchvision
- scikit-learn
- numpy
- matplotlib

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/shahzaib-123/VisionTransfromer-Scratch.git
    cd VisionTransfromer-Scratch
    ```
2. Install Dependencies:
    ```bash
    chmod +x requirements.sh
    ./requirements.sh
    ```
3. Run the training script:
    ```bash
    python train.py
    ```
4. Results are saved in the runs directory:
    * Model Weights: `runs/model_weights.pth`
    * Confusion Matrix: confusion matrix saved in `runs/confusion_matrix.csv`
    * Loss Curve per epoch: Train and test loss curves `runs/loss_curve.png`
    * Training Loss per Step: Detailed training loss `runs/train_loss_steps.png`
    * Accuracy Curve: Accuracy over epochs `runs/accuracy_curve.png`
5. Configuration Options: You can modify model parameters in `models/vit.py` and adjust training hyperparameters, like learning rate and batch size, in `train.py`

## Adding Your Own Data
To add your own dataset, follow these steps:
1. **Prepare your data**:
   - Ensure your data is in a format compatible with PyTorch. For image data, you can structure it in folders where each folder represents a class. For example:

     ```
     data/
     ├── train/
     │   ├── class1/
     │   ├── class2/
     │   └── class3/
     └── test/
         ├── class1/
         ├── class2/
         └── class3/
     ```
2. **Modify the Dataset Loading Code**:
   In the `train.py` file, you can modify the data loading section to point to your custom dataset. Replace the CIFAR-10 dataset with `ImageFolder` or any other dataset class from `torchvision.datasets`.

   Example for using custom data with `ImageFolder`:

   ```python
   from torchvision import datasets, transforms

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Resize((32, 32)),  # Resize images if necessary
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])

   train_data = datasets.ImageFolder(root='path_to_your_train_data', transform=transform)
   test_data = datasets.ImageFolder(root='path_to_your_test_data', transform=transform)

   train_loader = DataLoader(train_data, shuffle=True, batch_size=128)
   test_loader = DataLoader(test_data, shuffle=False, batch_size=128)
   ```
  
## Model Architecture
The Vision Transformer model is defined in `models/` with related utilities in `utilities/` and includes the following major components:
* Patchify Function: Splits each input image into patches. In this model, each CIFAR-10 image (32x32) is divided into a grid of smaller patches
* Positional Embeddings: The `get_positional_embeddings` function generates sinusoidal positional embeddings for each patch, allowing the model to retain spatial structure [https://arxiv.org/abs/1706.03762].
* Multi-Head Self-Attention: `MSA` class Implements multi-head self-attention by dividing embeddings into multiple heads, applying self-attention, and then concatenating the heads to capture different parts of the input sequence in parallel.
* Transformer Encoder Block: The `ViTBlock` class contains a standard transformer encoder layer with Layer Normalization, multi-head self-attention, and a feed-forward neural network with a GELU activation function.
* Vision Transformer `ViT` Model
  * Patch Embedding and Linear Mapper: Converts patches into embeddings using a linear layer.
  * Class Token: A special token representing the class, added to the sequence of patch embeddings.
  * Positional Embedding: Positional encodings are added to patch embeddings to maintain spatial information.
  * Transformer Layers: Multiple ViT blocks process the embeddings to learn representations.
  * Classification Head: Maps the final output of the class token to class probabilities using a softmax function.

Example of ViT initialization:
```python
model = ViT((3, 32, 32), n_patches=8, n_blocks=2, hidden_d=32, n_heads=2, out_d=10).to(device)
```

## References
* <a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale<a/>
* <a href="https://arxiv.org/abs/1706.03762">Sinusoidal Positional Embeddings<a/>
* <a href="https://medium.com/@brianpulfer">Pytorch Vision Transformer Tutorial<a/>
* <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR10 dataset<a/>
* <a href="https://openai.com/index/image-gpt/">Patch Embedding in Vision Transformers<a/>
* <a href="https://pytorch.org/docs/stable/index.html">PyTorch Documentation<a/>
* <a href="https://research.google/blog/transformers-for-image-recognition-at-scale/">Transformers for Image Recognition at Scale<a/>


## Contact
* shahasghar054@gmail.com
