"""
Phase 6: Federated Learning Implementation

WHY: We want to train models while keeping user data private. Each user's data
     stays on their device. Only model updates are shared, not raw data.

WHAT: Federated Learning (FL) is a distributed training approach:
      - Server maintains a global model
      - Clients (users) train on their local data
      - Clients send model updates (weights) to server
      - Server aggregates updates into new global model

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

# federated client
class FederatedClient:
    """
    Represents a single client (user) in federated learning
    
    Each client:
    - Has their own local dataset
    - Trains model on local data
    - Sends updates to server
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4    
    ):
        """
        Args:
            client_id: Unique client identifier
            model: Model to train (copy of global model)
            train_loader: DataLoader with client's local data
            device: Device to train on
            learning_rate: Learning rate for local training
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.learning_rate = learning_rate

        # criterian and optimizer  what are these two things ?
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam( # what is the adam here?
            self.model.parameters(),
            lr=learning_rate
        )

        print(f"CLient ${client_id} intialized with  ${len(train_loader.dataset)} samples")

    def train(self, num_epochs: int = 5) -> Dict:
        """
        Train model on local data
        
        Args:
            num_epochs: Number of local training epochs
            
        Returns:
            Dictionary with training statistics
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch in self.train_loader:
                # move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mark = batch['attention_mark'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # forward pass
                logits = self.model(input_ids, attention_mark, images)
                loss = self.criterion(logits, labels)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            total_loss += epoch_loss

        avg_loss = total_loss / (num_epochs * len(self.train_loader))
        accuracy = 100 * correct / total

        return {
            'client_id': self.client_id,
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples' : len(self.train_loader.dataset)
        }

    def get_model_weights(self) -> Dict:
        """
        Get current model weights
        
        Returns:
            Dictionary of model weights
        """ 
        return copy.deepcopy(self.model.state_dict())

    def set_model_weights(self, weights: Dict):
        """
        Set model weights from server
        
        Args:
            weights: Dictionary of model weights
        """ 
        self.model.load_state_dict(copy.deepcopy(weights))

class FederatedServer:
    """
    Central server for federated learning
    
    Server:
    - Maintains global model
    - Selects clients for each round
    - Aggregates client updates
    - Distributes new global model
    """  

    def __init__(
        self,
        model: nn.Module,
        device: torch.device    
    ):
        """
        Args:
            model: Global model architecture
            device: Device to run on
        """ 
             
        print(f"Server initialized")
        print(f"Global model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def select_clients(
        self,
        clients: List[FederatedClient],
        num_clients: int
    ) -> List[FederatedClient]:
        """
        Randomly select clients for training round
        
        Args:
            clients: List of all clients
            num_clients: Number of clients to select
            
        Returns:
            List of selected clients
        """
        num_to_select = min(num_clients, len(clients))
        selected_indices = np.random.choice(
            len(clients),
            num_to_select,
            replace=False
        )     

        selected = [clients[i] for i in selected_indices] # How this syntax work
        print(f"Selected {len(selected)} clients: {[c.client_id for c in selected]}")

        return selected
    
    def aggregate_weights(
        self,
        client_weights: List[Dict],
        client_num_samples: List[int]    
    ) -> Dict:
        """
        Aggregate client weights using FedAvg (weighted average by dataset size)
        
        Args:
            client_weights: List of client model weights
            client_num_samples: Number of samples each client has
            
        Returns:
            Aggregated model weights
        """
        # total samples 
        total_samples = sum(client_num_samples)

        # initialize aggregated weights
        aggregated = copy.deepcopy(client_weights[0])

        # weight each client by their dataset sixe
        for key in aggregated.keys():
            #start with zeros
            aggregated[key] = torch.zeros_like(aggregated[key])

            # weighted sum
            for i , client_weight in enumerate(client_weights):
                weight = client_num_samples[i] / total_samples
                aggregated[key] += client_weight[key] * weight

        return aggregated

    def distributed_model(self, clients: List[FederatedClient]):
        """
        Send global model to all clients
        
        Args:
            clients: List of clients
        """ 
        global_weights = self.global_model.state_dict()

        for client in clients:
            client.set_model_weights(global_weights)

    def update_global_model(self, aggregated_weights: Dict):
        """
        Update global model with aggregated weights
        
        Args:
            aggregated_weights: Aggregated client weights
        """

        self.global_model.load_state_dict(aggregated_weights)

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate global model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.global_model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention-mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.global_model(input_ids, attention_mask, images)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size()
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total   

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        } 

class FederatedTrainer:
    """
    Main federated learning trainer
    
    Coordinates entire FL process:
    - Initialize server and clients
    - Run FL rounds
    - Track metrics
    """  

    def __init__(
       self,
       model: nn.Module,
       train_dataset,
       test_dataset,
       num_clients: int,
       clients_per_round: int,
       device: torch.device,
       batch_size: int = 32     
    ):
       """
        Args:
            model: Model architecture
            train_dataset: Full training dataset
            test_dataset: Test dataset
            num_clients: Total number of clients
            clients_per_round: Clients participating per round
            device: Device to train on
            batch_size: Batch size for training
        """
       self.device = device
       self.num_clients = num_clients
       self.clients_per_round = clients_per_round
       self.batch_size = batch_size

       # initialize server
       self.server = FederatedServer(model, device)

       # create clients with data splits
       self.client = self._create_clients(train_dataset)

       # test loader (server use this )
       self.test_loader = DataLoader(
           test_dataset,
           batch_size=batch_size,
           shuffle=False
       )

       # track metrics
       self.metrics = {
            'rounds': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
       
    def _create_clients(self, train_dataset) -> List[FederatedClient]:
        """
        Create clients and split data among them
        
        Args:
            train_dataset: Full training dataset
            
        Returns:
            List of FederatedClient objects
        """

        # split dataset indices among clients
        num_samples = len(train_dataset)
        samples_per_client = num_samples // self.num_clients

        clients = []

        for i in range(self.num_clients):
            # get indices for this client
            start_idx = i * samples_per_client
            if i == self.num_clients - 1: # last client gets remaining samples
                end_idx = num_samples
            else:
                end_idx = ( i+ 1) * samples_per_client

            client_indices = list(range(start_idx, end_idx))

            # create subset for this client
            client_dataset = Subset(train_dataset, client_indices)

            # create dataloader
            client_loader = DataLoader(
                client_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )  

            # create client
            client_model = copy.deepcopy(self.server.global_model)
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_loader=client_loader,
                device=self.device
            )  
            clients.append(client)

        return clients
    
    def train_round(self, round_num: int, local_epochs: int = 5) -> Dict:
        """
        Execute one federated learning round
        
        Args:
            round_num: Current round number
            local_epochs: Number of epochs each client trains
            
        Returns:
            Dictionary with round statistics
        """
        print(f"\n{'='*70}")
        print(f"ROUND {round_num}")
        print(f"{'='*70}")

        # step 1 : select clients
        selected_clients = self.server.select_clients(
            self.clients,
            self.clients_per_round
        )

        # steps 2 : distributed global model
        self.server.distributed_model(selected_clients)

        # step 3 local training
        print(f"\nLocal training...")
        client_weights = []
        client_num_samples = []
        client_stats = []

        for client in tqdm(selected_clients, desc="Training clients"):
            # train locally
            stats = client.train(num_epochs=local_epochs)
            client_stats.append(stats)

            # get updated weights
            weights = client.get_model_weights()
            client_weights.append(weights)
            client_num_samples.append(stats['num_samples'])

        # step 4 : aggregate weights
        print(f"\nAggregating weights...")   
        aggregated_weights = self.server.aggregate_weights(
            client_weights,
            client_num_samples
        ) 

        # step 5: Update global model
        self.server.update_global_model(aggregated_weights)

        # step evaluat
        print(f"\nEvaluating global model...")

        # step 6
        test_results = self.server.evaluate(self.test_loader)

        # calcualte average client stats
        avg_client_loss = np.mean([s['loss'] for s in client_stats])
        avg_client_acc  = np.mean([s['accuracy'] for s in client_stats])

        # store metrics 
        self.metrics['rounds'].append(round_num)
        self.metrics['train_loss'].append(avg_client_loss)
        self.metrics['train_acc'].append(avg_client_acc)
        self.metrics['test_loss'].append(test_results['loss'])
        self.metrics['test_acc'].append(test_results['accuracy'])

        # Print summary
        print(f"\nRound {round_num} Summary:")
        print(f"  Avg Client Loss: {avg_client_loss:.4f}")
        print(f"  Avg Client Acc: {avg_client_acc:.2f}%")
        print(f"  Global Test Loss: {test_results['loss']:.4f}")
        print(f"  Global Test Acc: {test_results['accuracy']:.2f}%")

        return {
            'round': round_num,
            'train_loss': avg_client_loss,
            'train_acc' : avg_client_acc,
            'test_loss': test_results['loss'],
            'test_acc': test_results['accuracy']
        }
    
    def train(self, num_rounds: int, local_epochs: int = 5):
        """
        Run complete federated learning training
        
        Args:
            num_rounds: Number of FL rounds
            local_epochs: Local epochs per round
        """
        print(f"\n{'='*70}")
        print(f"FEDERATED LEARNING TRAINING")
        print(f"{'='*70}")
        print(f"Clients: {self.num_clients}")
        print(f"Clients per round: {self.clients_per_round}")
        print(f"Total rounds: {num_rounds}")
        print(f"Local epochs: {local_epochs}")

        for round_num in range(1, num_rounds + 1):
            self.train_round(round_num, local_epochs)

        print(f"\n{'='*70}")
        print(f"FEDERATED LEARNING COMPLETE!")
        print(f"{'='*70}")
        print(f"Final Test Accuracy: {self.metrics['test_acc'][-1]:.2f}%")

    def get_global_model(self) -> nn.Module:
        """ Return trained global model """
        return self.server.global_model        
    
# =====================
# Testing Code
# ====================

if __name__ == "__main__":
    print("="*70)
    print("Testing Federated Learning - Phase 6")
    print("="*70)
    
    # This would normally import your actual model and dataset
    # For now, we'll show the structure
    
    print("\nFederated Learning structure ready!")
    print("\nTo use:")
    print("1. Import your MultiModalModel")
    print("2. Load your datasets")
    print("3. Create FederatedTrainer")
    print("4. Run trainer.train(num_rounds=10)")
    
    print("\n" + "="*70)
    print("Federated Learning Implementation Complete! ✓")
    print("="*70)
            
               


        
              


