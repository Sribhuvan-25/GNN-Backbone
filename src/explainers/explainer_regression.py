import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
import networkx as nx
from sklearn.manifold import TSNE
import inspect

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
    
    def get_node_importance(self, edge_importance_matrix, node_names=None, save_path=None):
        """
        Calculate node importance from edge importance matrix
        
        Args:
            edge_importance_matrix: Matrix of edge importance scores
            node_names: Names of the nodes (features)
            save_path: Optional path to save the importance report
            
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
        
        # Save to file if path is provided
        if save_path is not None:
            import pandas as pd
            import os
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare data for CSV
            importance_data = []
            for i, idx in enumerate(sorted_indices):
                idx = idx.item()
                importance = normalized_importance[idx].item()
                raw_importance = node_importance[idx].item()
                degree = node_degrees[idx].item()
                
                node_name = node_names[idx] if node_names is not None else f"Node_{idx}"
                
                importance_data.append({
                    'rank': i + 1,
                    'node_name': node_name,
                    'normalized_importance': importance,
                    'raw_importance': raw_importance,
                    'degree': degree
                })
            
            # Save as CSV
            df = pd.DataFrame(importance_data)
            csv_path = save_path.replace('.txt', '.csv') if save_path.endswith('.txt') else f"{save_path}.csv"
            df.to_csv(csv_path, index=False)
            
            # Also save as human-readable text
            txt_path = save_path.replace('.csv', '.txt') if save_path.endswith('.csv') else f"{save_path}.txt"
            with open(txt_path, 'w') as f:
                f.write("=== NODE IMPORTANCE REPORT ===\n")
                f.write("Top 20 most important nodes (microbial families):\n")
                f.write("-" * 60 + "\n")
                
                for i, idx in enumerate(sorted_indices[:20]):
                    idx = idx.item()
                    importance = normalized_importance[idx].item()
                    raw_importance = node_importance[idx].item()
                    degree = node_degrees[idx].item()
                    
                    if node_names is not None:
                        node_name = node_names[idx]
                        f.write(f"{i+1:2d}. {node_name[:45]:45s} | Score: {importance:.4f} | Raw: {raw_importance:.4f} | Degree: {degree:.0f}\n")
                    else:
                        f.write(f"{i+1:2d}. Node {idx:3d}                                     | Score: {importance:.4f} | Raw: {raw_importance:.4f} | Degree: {degree:.0f}\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"Node importance calculated as: sum(edge_weights) / degree\n")
                f.write(f"Total nodes analyzed: {edge_importance_matrix.shape[0]}\n")
            
            print(f"Node importance report saved to: {txt_path}")
            print(f"Node importance CSV saved to: {csv_path}")
        
        return normalized_importance.cpu().numpy(), sorted_indices.cpu().numpy()
    
    def get_embedding_based_node_importance(self, model, data_list, node_names=None, save_path=None):
        """
        Calculate node importance based on embedding magnitudes from trained GNN
        This method is more sophisticated as it uses the learned representations
        
        Args:
            model: Trained GNN model
            data_list: List of data objects
            node_names: Names of the nodes (features)  
            save_path: Optional path to save the importance report
            
        Returns:
            node_importance: Array of embedding-based node importance scores
        """
        print("\n=== EMBEDDING-BASED NODE IMPORTANCE ===")
        print("Calculating importance from learned GNN embeddings...")
        
        model.eval()
        embedding_magnitudes = []
        
        # Process a subset of samples for efficiency
        num_samples = min(10, len(data_list))
        
        with torch.no_grad():
            for i in range(num_samples):
                data = data_list[i].to(self.device)
                
                # Get embeddings from trained model (your models return (prediction, embedding))
                try:
                    prediction, embeddings = model(data)
                    
                    # Handle different embedding formats
                    if embeddings.dim() == 1:
                        # Global pooled embedding - need to get node-level embeddings
                        # This means we need to get intermediate representations
                        node_embeddings = self._get_node_level_embeddings(model, data)
                    else:
                        # Already node-level embeddings
                        node_embeddings = embeddings
                    
                    # Calculate L2 norm for each node embedding
                    if node_embeddings is not None:
                        if node_embeddings.dim() == 3:  # [batch, nodes, features]
                            node_embeddings = node_embeddings.squeeze(0)
                        
                        # Calculate magnitude (L2 norm) for each node
                        magnitudes = torch.norm(node_embeddings, p=2, dim=-1)
                        embedding_magnitudes.append(magnitudes)
                        
                except Exception as e:
                    print(f"Warning: Could not extract embeddings from sample {i}: {e}")
                    continue
        
        if not embedding_magnitudes:
            print("ERROR: Could not extract embeddings from any samples")
            return None, None
        
        # Average embedding magnitudes across samples
        avg_embedding_magnitude = torch.stack(embedding_magnitudes).mean(dim=0)
        
        # Sort by importance (descending)
        sorted_indices = torch.argsort(avg_embedding_magnitude, descending=True)
        
        print(f"Embedding-based importance calculated for {len(avg_embedding_magnitude)} nodes")
        print("Top 20 most important nodes (based on embedding magnitude):")
        print("-" * 70)
        
        for i, idx in enumerate(sorted_indices[:20]):
            idx = idx.item()
            importance = avg_embedding_magnitude[idx].item()
            
            if node_names is not None and idx < len(node_names):
                node_name = node_names[idx]
                print(f"{i+1:2d}. {node_name[:45]:45s} | Embedding Magnitude: {importance:.4f}")
            else:
                print(f"{i+1:2d}. Node {idx:3d}                                     | Embedding Magnitude: {importance:.4f}")
        
        print("-" * 70)
        print(f"Embedding importance method: L2 norm of learned node representations")
        print(f"Total nodes analyzed: {len(avg_embedding_magnitude)}")
        
        # Save to file if path is provided
        if save_path is not None:
            import pandas as pd
            import os
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare data for CSV
            importance_data = []
            for i, idx in enumerate(sorted_indices):
                idx = idx.item()
                importance = avg_embedding_magnitude[idx].item()
                
                node_name = node_names[idx] if node_names is not None and idx < len(node_names) else f"Node_{idx}"
                
                importance_data.append({
                    'rank': i + 1,
                    'node_name': node_name,
                    'embedding_magnitude': importance
                })
            
            # Save as CSV
            df = pd.DataFrame(importance_data)
            csv_path = save_path.replace('.txt', '.csv') if save_path.endswith('.txt') else f"{save_path}.csv"
            df.to_csv(csv_path, index=False)
            
            # Also save as human-readable text
            txt_path = save_path.replace('.csv', '.txt') if save_path.endswith('.csv') else f"{save_path}.txt"
            with open(txt_path, 'w') as f:
                f.write("=== EMBEDDING-BASED NODE IMPORTANCE ===\n")
                f.write("Top 20 most important nodes (based on embedding magnitude):\n")
                f.write("-" * 70 + "\n")
                
                for i, idx in enumerate(sorted_indices[:20]):
                    idx = idx.item()
                    importance = avg_embedding_magnitude[idx].item()
                    
                    if node_names is not None and idx < len(node_names):
                        node_name = node_names[idx]
                        f.write(f"{i+1:2d}. {node_name[:45]:45s} | Embedding Magnitude: {importance:.4f}\n")
                    else:
                        f.write(f"{i+1:2d}. Node {idx:3d}                                     | Embedding Magnitude: {importance:.4f}\n")
                
                f.write("-" * 70 + "\n")
                f.write(f"Embedding importance method: L2 norm of learned node representations\n")
                f.write(f"Total nodes analyzed: {len(avg_embedding_magnitude)}\n")
            
            print(f"Embedding-based importance report saved to: {txt_path}")
            print(f"Embedding-based importance CSV saved to: {csv_path}")
        
        return avg_embedding_magnitude.cpu().numpy(), sorted_indices.cpu().numpy()
    
    def _get_node_level_embeddings(self, model, data):
        """
        Extract node-level embeddings from GNN model
        """
        try:
            # Try to get intermediate representations
            # This depends on your model architecture
            x = data.x
            edge_index = data.edge_index
            edge_weight = getattr(data, 'edge_weight', None)
            
            # Forward pass through model layers to get node embeddings
            if hasattr(model, 'conv_layers') and model.conv_layers:
                # Multi-layer model
                for conv_layer in model.conv_layers[:-1]:  # All but last layer
                    if edge_weight is not None:
                        x = conv_layer(x, edge_index, edge_weight)
                    else:
                        x = conv_layer(x, edge_index)
                    x = torch.relu(x)
                return x
            elif hasattr(model, 'conv1'):
                # Two-layer model
                if edge_weight is not None:
                    x = torch.relu(model.conv1(x, edge_index, edge_weight))
                else:
                    x = torch.relu(model.conv1(x, edge_index))
                return x
            else:
                # Single layer or different architecture
                return None
                
        except Exception as e:
            print(f"Could not extract node-level embeddings: {e}")
            return None
    
    def extract_universal_attention_scores(self, model, data, node_names=None):
        """
        UNIVERSAL attention score extraction for ANY GNN architecture
        Uses different methods based on model type for maximum compatibility
        """
        model_type = type(model).__name__.lower()
        
        if 'gat' in model_type:
            # GAT: Use explicit attention weights
            print("Extracting attention scores from GAT model")
            return self._extract_gat_attention_scores(model, data, node_names)
        elif 'rggc' in model_type or 'resgated' in model_type:
            # RGGC: Use gating mechanism + gradient importance
            print("Extracting attention scores from RGGC gating mechanism")
            return self._extract_rggc_attention_scores(model, data, node_names)
        else:
            # GCN or other: Use gradient-based importance
            print("Extracting attention scores using gradient-based method")
            return self._extract_gradient_attention_scores(model, data, node_names)
    
    def _extract_gat_attention_scores(self, model, data, node_names=None):
        """Extract attention scores from GAT models (explicit attention weights)"""
        try:
            model.eval()
            with torch.no_grad():
                # Forward pass to get attention weights
                x, edge_index = data.x, data.edge_index
                
                # Get attention from first layer (most interpretable)
                if hasattr(model, 'conv1') and hasattr(model.conv1, '__call__'):
                    # Extract attention weights during forward pass
                    alpha = None
                    def hook_fn(module, input, output):
                        nonlocal alpha
                        if hasattr(module, 'alpha'):
                            alpha = module.alpha
                    
                    handle = model.conv1.register_forward_hook(hook_fn)
                    _ = model.conv1(x, edge_index)
                    handle.remove()
                    
                    if alpha is not None:
                        # Aggregate attention scores per node (sum of incoming attention)
                        num_nodes = x.size(0)
                        node_attention = torch.zeros(num_nodes, device=self.device)
                        edge_index_np = edge_index.cpu().numpy()
                        
                        for i in range(alpha.size(0)):
                            target_node = edge_index_np[1, i]  # Target node
                            node_attention[target_node] += alpha[i].item()
                        
                        return node_attention.cpu().numpy()
            
            print("Warning: Could not extract GAT attention weights, falling back to gradients")
            return self._extract_gradient_attention_scores(model, data, node_names)
            
        except Exception as e:
            print(f"GAT attention extraction failed: {e}, using gradient method")
            return self._extract_gradient_attention_scores(model, data, node_names)
    
    def _extract_rggc_attention_scores(self, model, data, node_names=None):
        """Extract attention-like scores from RGGC models using gating + gradients"""
        try:
            model.eval()
            
            # Method 1: Extract gating values (RGGC's built-in attention mechanism)
            gate_scores = self._extract_rggc_gate_values(model, data)
            
            # Method 2: Gradient-based importance for additional insight
            gradient_scores = self._extract_gradient_attention_scores(model, data, node_names)
            
            # Combine both methods (weighted average)
            if gate_scores is not None and gradient_scores is not None:
                combined_scores = 0.6 * gate_scores + 0.4 * gradient_scores
                print("Combined RGGC gating + gradient scores")
                return combined_scores
            elif gate_scores is not None:
                print("Using RGGC gating scores only")
                return gate_scores
            else:
                print("Using gradient scores only for RGGC")
                return gradient_scores
                
        except Exception as e:
            print(f"RGGC attention extraction failed: {e}, using gradient method")
            return self._extract_gradient_attention_scores(model, data, node_names)
    
    def _extract_rggc_gate_values(self, model, data):
        """Extract gate values from RGGC layers (the built-in attention mechanism)"""
        try:
            x, edge_index = data.x, data.edge_index
            num_nodes = x.size(0)
            
            # Track gate activations during forward pass
            gate_activations = []
            
            def gate_hook(module, input, output):
                # RGGC layers have gating - capture the gate values
                if hasattr(module, 'gate') or 'gate' in str(module).lower():
                    gate_activations.append(output)
            
            # Register hooks on RGGC layers
            handles = []
            for name, module in model.named_modules():
                if 'conv' in name and ('rggc' in str(module).lower() or 'resgated' in str(module).lower()):
                    handle = module.register_forward_hook(gate_hook)
                    handles.append(handle)
            
            # Forward pass to trigger hooks
            with torch.no_grad():
                _ = model(x, edge_index, torch.zeros(num_nodes, dtype=torch.long, device=self.device))
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            if gate_activations:
                # Aggregate gate activations across layers
                gate_scores = torch.zeros(num_nodes, device=self.device)
                for activation in gate_activations:
                    if activation.size(0) == num_nodes:
                        gate_scores += torch.norm(activation, dim=-1)  # L2 norm per node
                
                return (gate_scores / len(gate_activations)).cpu().numpy()
            else:
                return None
                
        except Exception as e:
            print(f"Gate extraction failed: {e}")
            return None
    
    def _extract_gradient_attention_scores(self, model, data, node_names=None):
        """Extract node importance using gradient-based method (universal approach)"""
        try:
            model.train()  # Need gradients
            x, edge_index = data.x, data.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
            
            # Ensure input requires gradients
            x.requires_grad_(True)
            
            # Forward pass
            if len(inspect.signature(model.forward).parameters) == 3:
                output = model(x, edge_index, batch)
                prediction = output[0] if isinstance(output, tuple) else output
            else:
                output = model(x, edge_index)
                prediction = output[0] if isinstance(output, tuple) else output
            
            # Get gradient w.r.t. input features
            if prediction.dim() > 1:
                prediction = prediction.mean()  # Scalar for gradient computation
            
            gradients = torch.autograd.grad(
                outputs=prediction,
                inputs=x,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            # Calculate node importance as L2 norm of gradients per node
            node_importance = torch.norm(gradients, dim=-1, p=2)
            
            model.eval()  # Back to eval mode
            return node_importance.detach().cpu().numpy()
            
        except Exception as e:
            print(f"Gradient-based attention extraction failed: {e}")
            # Fallback: uniform importance
            return np.ones(data.x.size(0))
    
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
        
        # For explainer-based pruning, we want to be more aggressive
        # Use percentile-based selection instead of absolute threshold
        total_nodes = len(node_importance)
        
        # Keep top 50-70% of nodes based on importance (more aggressive pruning)
        nodes_to_keep = max(min_nodes, int(total_nodes * 0.6))  # Keep 60% of nodes
        
        print(f"\nAGGRESSIVE NODE-BASED PRUNING:")
        print(f"Nodes exceeding threshold {importance_threshold}: {len(important_nodes)}")
        print(f"Using percentile-based selection: keeping top {nodes_to_keep} out of {total_nodes} nodes")
        
        # Always use top-k selection for consistent pruning
        important_nodes = sorted_indices[:nodes_to_keep]
        
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
        UNIVERSAL node-based pruning using attention-like scores for ANY GNN type
        - GAT: Uses explicit attention weights
        - RGGC: Uses gating mechanism + gradients  
        - GCN: Uses gradient-based node importance
        
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
        print(f"\nUNIVERSAL ATTENTION-BASED NODE PRUNING:")
        print(f"Automatically detecting GNN type and using appropriate attention extraction")
        
        # Use universal attention score extraction (works for ANY GNN type)
        attention_scores = self.extract_universal_attention_scores(model, data, node_names)
        
        if attention_scores is None or len(attention_scores) == 0:
            print("Warning: Could not extract attention scores from any method")
            print("This should not happen as gradient method provides fallback")
            return None, None, None, None
        
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
                u_name = node_names[u] if u < len(node_names) else f"Node_{u}"
                v_name = node_names[v] if v < len(node_names) else f"Node_{v}"
                explanation += f"{i+1}. {u_name} ↔ {v_name}: {imp:.3f}\n"
            else:
                explanation += f"{i+1}. Feature {u} ↔ Feature {v}: {imp:.3f}\n"
        
        return explanation 