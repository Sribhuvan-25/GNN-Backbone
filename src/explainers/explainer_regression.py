import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
import networkx as nx
from sklearn.manifold import TSNE

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
    
    def get_node_importance(self, edge_importance_matrix, node_names=None):
        """
        Calculate node importance from edge importance matrix
        
        Args:
            edge_importance_matrix: Matrix of edge importance scores
            node_names: Names of the nodes (features)
            
        Returns:
            node_importance: Array of node importance scores
        """
        # Calculate node importance as sum of all edges connected to each node
        node_importance = torch.sum(edge_importance_matrix, dim=1)
        
        # Normalize by number of connections (degree centrality)
        node_degrees = torch.sum(edge_importance_matrix > 0, dim=1).float()
        node_degrees = torch.clamp(node_degrees, min=1)  # Avoid division by zero
        normalized_importance = node_importance / node_degrees
        
        # Generate sorted importance report
        sorted_indices = torch.argsort(normalized_importance, descending=True)
        
        print("\n=== NODE IMPORTANCE REPORT ===")
        print("Top 20 most important nodes (microbial families):")
        print("-" * 60)
        
        for i, idx in enumerate(sorted_indices[:20]):
            idx = idx.item()
            importance = normalized_importance[idx].item()
            raw_importance = node_importance[idx].item()
            degree = node_degrees[idx].item()
            
            if node_names is not None:
                node_name = node_names[idx]
                print(f"{i+1:2d}. {node_name[:45]:45s} | Score: {importance:.4f} | Raw: {raw_importance:.4f} | Degree: {degree:.0f}")
            else:
                print(f"{i+1:2d}. Node {idx:3d}                                     | Score: {importance:.4f} | Raw: {raw_importance:.4f} | Degree: {degree:.0f}")
        
        print("-" * 60)
        print(f"Node importance calculated as: sum(edge_weights) / degree")
        print(f"Total nodes analyzed: {edge_importance_matrix.shape[0]}")
        
        return normalized_importance.cpu().numpy(), sorted_indices.cpu().numpy()
    
    def create_node_pruned_graph(self, data, edge_importance_matrix, node_names=None, 
                                importance_threshold=0.2, min_nodes=10):
        """
        Create a pruned graph based on node importance instead of edge pruning
        
        Args:
            data: PyG Data object
            edge_importance_matrix: Matrix of edge importance scores
            node_names: Names of the nodes
            importance_threshold: Threshold for keeping nodes
            min_nodes: Minimum number of nodes to keep
            
        Returns:
            pruned_data: New PyG Data object with only important nodes and their edges
            kept_nodes: Indices of kept nodes
            pruned_node_names: Names of kept nodes
        """
        # Get node importance scores
        node_importance, sorted_indices = self.get_node_importance(edge_importance_matrix, node_names)
        
        # Determine which nodes to keep based on importance
        important_nodes = np.where(node_importance > importance_threshold)[0]
        
        # If too few nodes meet threshold, keep top N nodes
        if len(important_nodes) < min_nodes:
            print(f"\nOnly {len(important_nodes)} nodes exceed threshold {importance_threshold}")
            print(f"Keeping top {min_nodes} most important nodes instead")
            important_nodes = sorted_indices[:min_nodes]
        
        print(f"\nNODE-BASED PRUNING:")
        print(f"Original graph: {data.x.shape[0]} nodes")
        print(f"Pruned graph: {len(important_nodes)} nodes ({len(important_nodes)/data.x.shape[0]*100:.1f}% retained)")
        print(f"Importance threshold used: {importance_threshold}")
        
        # Create mapping from old node indices to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(important_nodes)}
        
        # Filter node features
        pruned_x = data.x[important_nodes]
        
        # Filter edges - keep only edges between important nodes
        edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        new_edges = []
        
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            if u in old_to_new and v in old_to_new:
                # Both nodes are important, keep this edge with new indices
                new_u, new_v = old_to_new[u], old_to_new[v]
                new_edges.append([new_u, new_v])
                edge_mask[i] = True
        
        if len(new_edges) > 0:
            pruned_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        else:
            # If no edges remain, create empty edge index
            pruned_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"Edges: {data.edge_index.shape[1]} → {pruned_edge_index.shape[1]} ({pruned_edge_index.shape[1]/data.edge_index.shape[1]*100:.1f}% retained)")
        
        # Create new data object
        pruned_data = data.clone()
        pruned_data.x = pruned_x
        pruned_data.edge_index = pruned_edge_index.to(data.edge_index.device)
        
        # Get names of kept nodes
        pruned_node_names = None
        if node_names is not None:
            pruned_node_names = [node_names[i] for i in important_nodes]
        
        return pruned_data, important_nodes, pruned_node_names
    
    def create_attention_based_node_pruning(self, data, model, node_names=None, 
                                          attention_threshold=0.2, min_nodes=10):
        """
        Create node-based pruning using GAT attention scores
        
        Args:
            data: PyG Data object
            model: Trained GAT model with attention mechanism
            node_names: Names of the nodes
            attention_threshold: Threshold for keeping nodes based on attention
            min_nodes: Minimum number of nodes to keep
            
        Returns:
            pruned_data: New PyG Data object with only important nodes
            kept_nodes: Indices of kept nodes  
            pruned_node_names: Names of kept nodes
            attention_scores: Node attention importance scores
        """
        print(f"\nATTENTION-BASED NODE PRUNING:")
        print(f"Using GAT attention scores for node importance")
        
        # Extract attention scores from GAT model
        attention_scores = self._extract_gat_attention_scores(data, model)
        
        if attention_scores is None:
            print("Warning: Could not extract attention scores, falling back to edge-based importance")
            return self.create_node_pruned_graph(data, torch.eye(data.x.shape[0]), node_names, 
                                               attention_threshold, min_nodes)
        
        # Determine important nodes based on attention scores
        important_nodes = np.where(attention_scores > attention_threshold)[0]
        
        # Sort nodes by attention score for fallback selection
        sorted_indices = np.argsort(attention_scores)[::-1]
        
        # If too few nodes meet threshold, keep top N nodes
        if len(important_nodes) < min_nodes:
            print(f"Only {len(important_nodes)} nodes exceed attention threshold {attention_threshold}")
            print(f"Keeping top {min_nodes} most important nodes by attention score")
            important_nodes = sorted_indices[:min_nodes]
        
        print(f"Original graph: {data.x.shape[0]} nodes")
        print(f"Pruned graph: {len(important_nodes)} nodes ({len(important_nodes)/data.x.shape[0]*100:.1f}% retained)")
        print(f"Attention threshold used: {attention_threshold}")
        
        # Report top attention nodes
        print(f"Top 10 nodes by attention score:")
        for i, idx in enumerate(sorted_indices[:10]):
            name = node_names[idx] if node_names else f"Node {idx}"
            score = attention_scores[idx]
            kept = "✓" if idx in important_nodes else "✗"
            print(f"  {i+1:2d}. {name[:40]:40s} | Score: {score:.4f} {kept}")
        
        # Create mapping from old node indices to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(important_nodes)}
        
        # Filter node features
        pruned_x = data.x[important_nodes]
        
        # Filter edges - keep only edges between important nodes
        new_edges = []
        
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            if u in old_to_new and v in old_to_new:
                # Both nodes are important, keep this edge with new indices
                new_u, new_v = old_to_new[u], old_to_new[v]
                new_edges.append([new_u, new_v])
        
        if len(new_edges) > 0:
            pruned_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        else:
            # If no edges remain, create empty edge index
            pruned_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"Edges: {data.edge_index.shape[1]} → {pruned_edge_index.shape[1]} ({pruned_edge_index.shape[1]/data.edge_index.shape[1]*100:.1f}% retained)")
        
        # Create new data object
        pruned_data = data.clone()
        pruned_data.x = pruned_x
        pruned_data.edge_index = pruned_edge_index.to(data.edge_index.device)
        
        # Get names of kept nodes
        pruned_node_names = None
        if node_names is not None:
            pruned_node_names = [node_names[i] for i in important_nodes]
        
        return pruned_data, important_nodes, pruned_node_names, attention_scores
    
    def _extract_gat_attention_scores(self, data, model):
        """
        Extract attention scores from GAT model for node importance
        
        Args:
            data: PyG Data object
            model: Trained GAT model
            
        Returns:
            node_attention_scores: Array of attention importance for each node
        """
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Move data to same device as model
            data = data.to(next(model.parameters()).device)
            
            # Forward pass through model to get attention weights
            with torch.no_grad():
                # Check if model has GAT layers with attention
                attention_weights_list = []
                
                # Hook to capture attention weights from GAT layers
                def attention_hook(module, input, output):
                    if hasattr(module, '_alpha'):
                        # GAT layer stores attention weights in _alpha
                        attention_weights_list.append(module._alpha.detach())
                
                # Register hooks for all GAT layers
                hooks = []
                for name, module in model.named_modules():
                    if 'conv' in name and hasattr(module, '_alpha'):
                        hooks.append(module.register_forward_hook(attention_hook))
                
                # Forward pass
                _ = model(data.x, data.edge_index, data.batch)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                if len(attention_weights_list) == 0:
                    print("No GAT attention weights found in model")
                    return None
                
                # Aggregate attention weights across layers
                # Use the last layer's attention as it's most relevant for final prediction
                final_attention = attention_weights_list[-1]
                
                # Convert edge attention weights to node attention scores
                node_attention_scores = self._edge_attention_to_node_scores(
                    data.edge_index, final_attention, data.x.shape[0]
                )
                
                return node_attention_scores.cpu().numpy()
                
        except Exception as e:
            print(f"Error extracting GAT attention scores: {e}")
            return None
    
    def _edge_attention_to_node_scores(self, edge_index, edge_attention, num_nodes):
        """
        Convert edge attention weights to node-level attention scores
        
        Args:
            edge_index: Edge connectivity
            edge_attention: Attention weights for each edge
            num_nodes: Total number of nodes
            
        Returns:
            node_scores: Attention score for each node
        """
        # Initialize node scores
        node_scores = torch.zeros(num_nodes, device=edge_attention.device)
        node_degree = torch.zeros(num_nodes, device=edge_attention.device)
        
        # Sum attention weights for each node (both incoming and outgoing)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            attention = edge_attention[i] if edge_attention.dim() == 1 else edge_attention[i].sum()
            
            # Add attention to both source and destination nodes
            node_scores[src] += attention
            node_scores[dst] += attention
            node_degree[src] += 1
            node_degree[dst] += 1
        
        # Normalize by degree to get average attention per edge
        node_degree = torch.clamp(node_degree, min=1)  # Avoid division by zero
        node_scores = node_scores / node_degree
        
        return node_scores
    
    def create_tsne_embedding_plot(self, embeddings, targets, target_names=None, save_path=None):
        """
        Create t-SNE plot of embeddings to visualize clustering/data distribution
        
        Args:
            embeddings: Node embeddings from GNN model
            targets: Target values for coloring
            target_names: Names of target variables
            save_path: Path to save the plot
        """
        print("\nCreating t-SNE plot for embeddings...")
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        n_targets = targets.shape[1] if len(targets.shape) > 1 else 1
        
        if n_targets == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_targets, figsize=(10*n_targets, 8))
            if n_targets == 1:
                axes = [axes]
        
        for i in range(n_targets):
            ax = axes[i]
            
            # Get target values for this variable
            if n_targets == 1:
                target_vals = targets if len(targets.shape) == 1 else targets[:, 0]
                title = target_names[0] if target_names else "Target"
            else:
                target_vals = targets[:, i]
                title = target_names[i] if target_names else f"Target {i+1}"
            
            # Create scatter plot colored by target values
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=target_vals, cmap='viridis', alpha=0.7, s=60)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label=f'{title} Values')
            
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.set_title(f't-SNE Embedding Visualization - {title}\nClustering/Data Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE plot saved to: {save_path}")
        
        plt.close()
        
        return embeddings_2d
    
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