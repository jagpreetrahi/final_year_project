"""
Multi modal fusion
Combinig the text and the images model
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class SimpleFusion(nn.Module):

    def __init__(
        self, input_dim: int = 512, output_dim: int = 256
    ):
        super(SimpleFusion, self).__init__()

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor
    ) -> torch.Tensor:
        """ Concatenate and fuse """
        combined = torch.cat([text_emb, image_emb], dim=1)
        return self.fusion(combined) 

class AttentionFusion(nn.Module):
    """ Attention based fusion - learns importance of each modality """

    def __init__(self, feature_dim : int = 256):
        super(AttentionFusion, self).__init__()

        #attention networks for each modality
        self.text_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(), # intialize internal module state
            nn.Linear(128, 1)
        )

        self.image_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor
    )   -> torch.Tensor:
        """ Apply attention-based fusion """
        # Calculate attention scores
        text_score = self.text_attention(text_emb)
        image_score = self.image_attention(image_emb)

        #calculate score and apply softwax
        scores = torch.cat([text_score, image_score], dim=1)
        weights = self.softmax(scores) # [batch, 2]

        #apply weights
        text_weight = weights[:, 0].unsqueeze(1)
        image_weight = weights[:, 1].unsqueeze(1)

        #weighted combination
        fused = text_weight * text_emb + image_emb * image_weight

        return fused
    
class GatedFusion(nn.Module):

    def __init__(self, feature_dim: int = 256):
        super(GatedFusion, self).__init__()    
         
        #gate network
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        #output transformation
        self.transform = nn.Linear(feature_dim, feature_dim)

    def forward(
       self,
       text_emb: torch.Tensor,
       image_emb: torch.Tensor
    )   -> torch.Tensor:
        """Apply gate fusion """
        #concatenate for gate fusion
        concat = torch.cat([text_emb, image_emb], dim=1)

        #compute gate values
        gate_values = self.gate(concat) #[batch, feature_dim]

        #apply gating
        gated_text = gate_values * text_emb
        gated_image = (1 - gate_values) * image_emb 

        #combine and transform
        fused = self.transform(gated_text + gated_image)

        return fused

class MutltiModalModel(nn.Module):
    """
    Complete multi-modal recommendation model
    
    Combines text and image encoders with fusion mechanism
    """

    def __init__(
        self,
        text_encoder,
        image_encoder,
        fusion_type: str = "attention",
        num_classes : int = 10   
    ):
        """
        Args:
            text_encoder: TextEncoder instance
            image_encoder: ImageEncoder instance  
            fusion_type: Fusion strategy (simple, attention, gated)
            num_classes: Number of output classes
        """
        super(MutltiModalModel, self).__init__()

        self.fusion_type = fusion_type
        self.text_encoder = text_encoder
        self.image_encoder=  image_encoder

        text_dim = text_encoder.get_embeddings_dim()
        image_dim = image_encoder.get_embeddings_dim()

        #Fusion layer 
        if fusion_type == "simple":
            self.fusion = SimpleFusion(
                input_dim=text_dim + image_dim,
                output_dim=256
            )
        elif fusion_type == "attention":
            assert text_dim == image_dim, "Text and image dims must match for attention fusion"
            self.fusion = AttentionFusion(feature_dim=text_dim)
        elif fusion_type == "gated":
            assert  text_dim == image_dim, "Text and image dims must match for gated fusion"
            self.fusion = GatedFusion(feature_dim=text_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")   

        #classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )  

        # layer norm for stability 
        self.layer_norm = nn.LayerNorm(256)

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        images: torch.Tensor,
        return_embeddings : bool = False
    )   -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Forward Pass """
        #encoded text
        text_emb = self.text_encoder(text_input_ids, text_attention_mask)

        #encode image
        image_emb = self.image_encoder(images)

        #fuse modalities
        fused = self.fusion(text_emb, image_emb)
        fused = self.layer_norm(fused)

        #classification
        logits = self.classifier(fused)

        if return_embeddings:
            return logits, fused
        else:
            return logits

    def get_embeddings(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        images: torch.Tensor
    )    -> torch.Tensor:
        """ Extract embedding only for recommendation """
        with torch.no_grad():
            _, embeddings = self.forward(
                text_input_ids,
                text_attention_mask,
                images,
                return_embeddings=True
            )
        return embeddings    

         