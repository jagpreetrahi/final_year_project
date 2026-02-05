"""
Text encoder model using Bert

"""
from typing import Dict
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    """
      BERT -based text encoder for social media posts 

      Input Text string
      ouptut: fiixed size embeedding (default : 256 dim)
    """

    def __init__(
        self,
        model_name: str = "bert_base_uncased",
        hidden_dim: int = 256,
        freeze_bert : bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
        :param model_name: Pretrained bert model
        :param hidden_dim: Output embedding dimension
        :param freeze_bert: whether to freezes bert parameters
        :param dropout: Dropout probability
        
        """
        super(TextEncoder, self).__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim

        #Load pretrained bert
        self.bert = BertModel.from_pretrained(model_name)

        #freeze BERT parameters if specifies
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("BERT parameters frozen (feature extration mode)")   
        else:
            print("BERT parameters frozen (fine-tuning  mode)")       


        #custom encoder layer on top of Bert
        self.encoder = nn.Sequential(
            nn.Linear(768, 512), # BERT base outputs 768-dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        #Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            embeddings: Text embeddings [batch_size, hidden_dim]

        """   
        #BERT output 
        if self.bert.training and any(p.requires_grad for p in self.bert.parameters()):
            #If fine-tuning, compute gradient 
            bert_output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
        else:
            #if frozen , no gradient needed
            with torch.no_grad():
                bert_output = self.bert(
                    input_ids= input_ids,
                    attention_mask= attention_mask
                )

        #use the CLS token representation
        cls_embedding = bert_output.last_hidden_state[:, 0, :] # [batch_size, 768]

        #pass through custom encoder
        embeddings = self.encoder(cls_embedding)

        #apply the normalization
        embeddings = self.layer_norm(embeddings)

        return  embeddings

    def get_embeddings_dim(self) -> int:
        """Return output embedding dimensions"""
        return self.hidden_dim   

    def unfreeze_bert(self):
        """Unfreeze BERT Parameter for fine-tuning"""
        for param in self.bert.parameters():
            param.requires_grad = True
        print("Bert parameters unfrozen")

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        print("BERT parameters freeze")    

def get_text_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizer:
    """
      get Bert tokenizer
      Args: model_name = BERT model name

      Returns :
        tokenizer : BertTokenizer iNstances
    """
    return BertTokenizer.from_pretrained(model_name)

def tokenize_texts(
    texts: list,
    tokenizer: BertTokenizer,
    max_length: int = 128,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
     
    Tokenize a batch of texts
    
    Args:
        texts: List of text strings
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        device: Device to put tensors on
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    if device is not None:
        encoded =  {k: v.to(device) for k , v in encoded.items()}

    return encoded    

# Testing code
if __name__ == "__main__":
    print("="*70)
    print("Testing Text Encoder")
    print("="*70)
    
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create encoder
    encoder = TextEncoder(hidden_dim=256, freeze_bert=True)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Get tokenizer
    tokenizer = get_text_tokenizer()
    
    # Sample texts
    texts = [
        "Just watched an amazing movie! 🎬",
        "Delicious homemade pizza tonight 🍕",
        "Working on my AI project, it's so interesting!",
        "Beautiful sunset at the beach today ☀️"
    ]
    
    print("Sample texts:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    # Tokenize
    encoded = tokenize_texts(texts, tokenizer, device=device)
    
    print(f"\nTokenized shapes:")
    print(f"  Input IDs: {encoded['input_ids'].shape}")
    print(f"  Attention mask: {encoded['attention_mask'].shape}")
    
    # Get embeddings
    with torch.no_grad():
        embeddings = encoder(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    
    print(f"\nOutput embeddings shape: {embeddings.shape}")
    print(f"Expected: [4, 256] ✓")
    
    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")





