"""
Image encoder using the ResNet
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional

class ImageEncoder(nn.Module):
    """
    ResNet-based image encoder for social media images
    
    Input: Image tensor [batch_size, 3, 224, 224]
    Output: Fixed-size embedding (default: 256-dim)
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        hidden_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2
    ):
        """
        Args:
            model_name: ResNet variant (resnet18, resnet34, resnet50, resnet101)
            hidden_dim: Output embedding dimension
            pretrained: Use pretrained ImageNet weights
            freeze_backbone: Whether to freeze ResNet parameters
            dropout: Dropout probability
        """  
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim

        #Load pretrained model
        if model_name == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim  = 512
        elif model_name == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        #Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        #freeze backbone if specified
        if freeze_backbone:  
            for param in self.features.parameters():
                param.requires_grad = False
            print("Resnet parametes frozen (feature extraction mode)")
        else:  
            print("Resnet paramenters trainable (fine-tuning mode)")

        #custom encoder layer 
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )    

        #Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            images: Image tensors [batch_size, 3, 224, 224]
            
        Returns:
            embeddings: Image embeddings [batch_size, hidden_dim]
        
        """
        #Extract features
        if self.features.training and any(p.requires_grad for p in self.features.parameters()):
            features = self.features(images)
        else:
            with torch.no_grad():
              features = self.features(images)    

        # Flatten: [batch_size, feature_dim, height, width] -> [batch_size, features] 
        features = features.view(features.size(0), -1)

        # Pass through custom encoder
        embeddings = self.encoder(features)

        #apply layer normalization
        embeddings = self.layer_norm(embeddings)

        return embeddings

    def get_embeddings_dim(self) -> int:
        return self.hidden_dim

    def unfreeze(self, num_layers: Optional[int] = None):
        """
        Unfreeze  for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
         
        if num_layers is None:
           for param in self.features.parameters():
               param.requires_grad = True
        else:
            #unfreeze last N layers
            layers = list(self.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True 

    def freeze(self):
        for params in self.features.parameters():
            params.requires_grad = False

def get_image_transforms(
    image_size: int = 224,
    augment: bool  = False
)  -> transforms.Compose:
    """
    Get image preprocessing transforms
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation
        
    Returns:
        Composed transforms
    """
    if augment:
        # trainig transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        #validation/test tranforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])   
    return transform

# Testing code
if __name__ == "__main__":
    print("="*70)
    print("Testing Image Encoder")
    print("="*70)
    
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create encoder
    encoder = ImageEncoder(
        model_name="resnet50",
        hidden_dim=256,
        freeze_backbone=True
    )
    encoder = encoder.to(device)
    encoder.eval()
    
    # Create dummy images
    batch_size = 8
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"Input images shape: {dummy_images.shape}")
    
    # Get embeddings
    with torch.no_grad():
        embeddings = encoder(dummy_images)
    
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected: [{batch_size}, 256] ✓")
    
    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")
    
    # Test different ResNet variants
    print("\n" + "="*70)
    print("Testing Different ResNet Variants")
    print("="*70)
    
    for model_name in ["resnet18", "resnet34", "resnet50"]:
        encoder_variant = ImageEncoder(
            model_name=model_name,
            hidden_dim=256,
            pretrained=False,  # Faster for testing
            freeze_backbone=True
        ).to(device)
        encoder_variant.eval()
        
        with torch.no_grad():
            emb = encoder_variant(dummy_images)
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder_variant.parameters())
        trainable_params = sum(p.numel() for p in encoder_variant.parameters() if p.requires_grad)
        
        print(f"\n{model_name.upper()}:")
        print(f"  Output shape: {emb.shape}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
    
    # Test transforms
    print("\n" + "="*70)
    print("Testing Image Transforms")
    print("="*70)
    
    from PIL import Image
    import numpy as np
    
    # Create dummy PIL image
    dummy_pil = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    # Get transforms
    train_transform = get_image_transforms(augment=True)
    test_transform = get_image_transforms(augment=False)
    
    # Apply transforms
    train_tensor = train_transform(dummy_pil)
    test_tensor = test_transform(dummy_pil)
    
    print(f"Train transform output: {train_tensor.shape}")
    print(f"Test transform output: {test_tensor.shape}")
    
    print("\n" + "="*70)
    print("Image Encoder Test Complete! ✓")
    print("="*70)

                                                  

