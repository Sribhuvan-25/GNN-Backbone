import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
import networkx as nx

class GNNExplainerRegression:
    """GNN explainer for regression tasks"""
    
    def __init__(self, model, device):
        """
        Initialize the explainer
        
        Args:
            model: The trained GNN model
            device: Device to run the explanation on
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set model to evaluation mode
    
    def explain_graph(self, data, node_names=None, save_path=None, target_idx=0):
        """
        Generate explanation for a graph by learning an edge mask
        
        Args:
            data: PyG Data object for a graph
            node_names: Names of the nodes (features)
            save_path: Path to save the edge importance matrix
            target_idx: Index of the target variable to explain
            
        Returns:
            edge_importance_matrix: Matrix of edge importance scores
            explanation: Text explanation of the most important edges
        """
        # Copy data to device
        data = data.to(self.device)
        
        # Initialize edge mask - start with random values to encourage learning
        edge_mask = torch.rand(data.edge_index.shape[1], dtype=torch.float, 
                              requires_grad=True, device=self.device)
        
        # Setup optimizer for the edge mask
        optimizer = Adam([edge_mask], lr=0.01)
        
        # Number of epochs for explanation
        num_epochs = 200
        
        # Store original edge index
        original_edge_index = data.edge_index.clone()
        
        # Extract target
        target = data.y[:, target_idx].squeeze()
        
        # Get original prediction for reference
        with torch.no_grad():
            original_out = self.model(data.x, data.edge_index, data.batch)
            if isinstance(original_out, tuple):
                original_pred = original_out[0][:, target_idx] if original_out[0].shape[1] > 1 else original_out[0].squeeze()
            else:
                original_pred = original_out[:, target_idx] if original_out.shape[1] > 1 else original_out.squeeze()
        
        # Track convergence
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        # Training loop to learn edge mask
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Apply edge mask by creating a new edge_index with masked edges
            # Use sigmoid to make mask values between 0 and 1
            mask_probs = torch.sigmoid(edge_mask)
            
            # Use temperature annealing for better learning
            temperature = max(0.1, 1.0 - epoch / num_epochs)  # Decrease temperature over time
            
            # Apply temperature to mask probabilities
            temp_mask_probs = torch.sigmoid(edge_mask / temperature)
            
            # Create masked edge index by keeping edges with high probability
            threshold = 0.3  # Keep edges with probability > 0.3
            keep_edges = temp_mask_probs > threshold
            
            # Ensure we keep at least some edges
            if keep_edges.sum() == 0:
                # If no edges meet threshold, keep the top 10% of edges
                num_keep = max(1, int(0.1 * edge_mask.shape[0]))
                _, top_indices = torch.topk(temp_mask_probs, num_keep)
                keep_edges = torch.zeros_like(temp_mask_probs, dtype=torch.bool)
                keep_edges[top_indices] = True
            
            # Create masked edge index
            masked_edge_index = original_edge_index[:, keep_edges]
            
            # Forward pass with masked edges
            out = self.model(data.x, masked_edge_index, data.batch)
            
            # Handle different model output formats
            if isinstance(out, tuple):
                pred = out[0]
                if len(pred.shape) == 1:
                    pred = pred.view(-1, 1)
                if pred.shape[1] > 1:
                    pred = pred[:, target_idx]
            else:
                pred = out
                if len(pred.shape) == 1:
                    pred = pred.view(-1, 1)
                if pred.shape[1] > 1:
                    pred = pred[:, target_idx]
            
            pred = pred.squeeze()
            
            # Loss: prediction should match original prediction (not target)
            # This encourages finding edges that are important for the model's decision
            prediction_loss = F.mse_loss(pred, original_pred)
            
            # Sparsity regularization: encourage fewer edges
            sparsity_loss = 0.01 * mask_probs.sum() / mask_probs.shape[0]  # Normalize by number of edges
            
            # Total loss
            total_loss = prediction_loss + sparsity_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Check for early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(f"Explainer epoch {epoch+1}/{num_epochs}: Loss = {total_loss.item():.6f}")
        
        print(f"Final explainer loss: {best_loss:.6f}")
        
        # Get final edge importance scores
        with torch.no_grad():
            final_mask_probs = torch.sigmoid(edge_mask)
        
        # Create edge importance matrix
        num_nodes = data.x.shape[0]
        edge_importance_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        # Fill importance matrix with learned edge mask values
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[0, i], data.edge_index[1, i]
            edge_importance_matrix[u, v] = final_mask_probs[i]
        
        # Make importance matrix symmetric (for undirected graphs)
        edge_importance_matrix = torch.maximum(edge_importance_matrix, edge_importance_matrix.t())
        
        # Generate text explanation of most important edges
        explanation = self._generate_explanation(edge_importance_matrix, node_names)
        
        # Save edge importance matrix if path is provided
        if save_path is not None:
            if node_names is not None:
                # Create DataFrame with node names
                importance_df = pd.DataFrame(
                    edge_importance_matrix.cpu().detach().numpy(),
                    index=node_names,
                    columns=node_names
                )
                importance_df.to_csv(save_path)
            else:
                # Save as numpy array
                np.savetxt(save_path, edge_importance_matrix.cpu().detach().numpy(), delimiter=',')
        
        return edge_importance_matrix, explanation
    
    def _generate_explanation(self, edge_importance_matrix, node_names=None):
        """Generate text explanation from edge importance matrix"""
        # Convert to numpy
        importance = edge_importance_matrix.cpu().detach().numpy()
        
        # Get top edges
        n = importance.shape[0]
        top_edges = []
        
        for i in range(n):
            for j in range(i+1, n):  # Only consider upper triangle for undirected graph
                if importance[i, j] > 0.2:  # Threshold for importance
                    top_edges.append((i, j, importance[i, j]))
        
        # Sort by importance
        top_edges.sort(key=lambda x: x[2], reverse=True)
        
        # Generate explanation text
        explanation = "Top important feature interactions:\n"
        
        for i, (u, v, imp) in enumerate(top_edges[:10]):  # Show top 10 edges
            if node_names is not None:
                u_name = node_names[u]
                v_name = node_names[v]
                explanation += f"{i+1}. {u_name} ↔ {v_name}: {imp:.3f}\n"
            else:
                explanation += f"{i+1}. Feature {u} ↔ Feature {v}: {imp:.3f}\n"
        
        return explanation 