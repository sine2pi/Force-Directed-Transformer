
### Standard Attention vs. Force Vector Models

#### There's a fundamental limitation in how standard attention mechanisms operate:

Standard Attention (What's Actually Implemented)

- Uses scalar dot products for similarity scoring
- Produces weighted averages (linear combinations)
- No concept of reciprocal forces or vectors
- No natural representation of curved space
- Fundamentally linear operations

The Force-Directed Model (What I Conceptually Propose)

- Would operate with vector forces rather than scalar similarities
- Would calculate direction-dependent interactions
- Would model repulsion/attraction rather than just similarity
- Would naturally create curved relationships in the embedding space as more points are added .. (arbitrary precision)
- Second example adds a splash of topology..

First basic test complete and posted at the bottom of the readme along with all unit tests ready to run.



![vector1](https://github.com/user-attachments/assets/c54dc57f-a129-4062-bbca-c2853f31f16c)


Blue dots for emission tokens and red dots for receptivity tokens.

Green arrows representing the forces between tokens, showing both direction and magnitude.

Forces:

          tensor([[[[ 0.0000, -0.0000],
                    [ 2.0000,  0.0000],
                    [ 0.3795, -0.1265]],
                   [[ 0.0000,  0.0000],
                    [-0.0000,  0.0000],
                    [-0.7071,  0.7071]],
                   [[ 0.7071, -0.7071],
                    [-0.7071,  0.7071],
                    [ 0.0000,  0.0000]]]])


![vector2](https://github.com/user-attachments/assets/db877ba9-b18c-4f84-bd23-1423bebb2a25)

Forces:

          tensor([[[-0.6392, -0.7031],
                   [ 0.0000,  0.0000],
                   [-0.0067, -0.0197]],
                  [[-0.6392, -0.7031],
                   [ 0.0000,  0.0000],
                   [-0.0067, -0.0197]],
                  [[ 0.0305,  0.1363],
                   [ 0.0000,  0.0000],
                   [-0.4995, -1.2486]]])
```python


class ForceDirectedAttention(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.dims = dims
        self.heads = heads
        
        self.force_emitter = nn.Linear(dims, dims)
        
        self.force_receptor = nn.Linear(dims, dims)
        
        self.direction_modulator = nn.Parameter(torch.randn(heads, dims))
        
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)
    
    def forward(self, x):
        batch, seq_len = x.shape[:2]
        
        emissions = self.force_emitter(x)
        receptivity = self.force_receptor(x)
        
        q = self.q_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        
        token_i = emissions.unsqueeze(2)
        token_j = receptivity.unsqueeze(1)
        
        force_directions = token_i - token_j
        force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
        normalized_forces = force_directions / (force_magnitudes + 1e-8)
        
        direction_scores = torch.zeros(batch, self.heads, seq_len, seq_len, device=x.device)
        
        for h in range(self.heads):
            head_modulator = self.direction_modulator[h].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            
            dot_products = normalized_forces * head_modulator
            
            head_scores = dot_products.sum(dim=-1)
            
            direction_scores[:, h] = head_scores
        
        broadcast_magnitudes = force_magnitudes.squeeze(-1).unsqueeze(1)
        
        force_field = direction_scores * torch.exp(-broadcast_magnitudes)
        
        weights = F.softmax(force_field, dim=-1)
        
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        
        return self.output(output)


class HybridForceTopologicalAttention(nn.Module):
    def __init__(self, dims, heads, hop_levels=3):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.hop_levels = hop_levels
        
        # Force-related components
        self.force_emitter = nn.Linear(dims, dims)
        self.force_receptor = nn.Linear(dims, dims)
        self.direction_modulator = nn.Parameter(torch.randn(heads, dims))
        
        # Topological components
        self.edge_projector = nn.Linear(dims * 2, heads)
        self.hop_weights = nn.Parameter(torch.ones(hop_levels) / hop_levels)
        
        # Integration components
        self.force_topo_balance = nn.Parameter(torch.tensor(0.5))
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)
    
    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]
        
        # Standard projections for values
        q = self.q_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        
        # Calculate force vectors
        emissions = self.force_emitter(x)
        receptivity = self.force_receptor(x)
        
        token_i = emissions.unsqueeze(2)  # [batch, seq, 1, dim]
        token_j = receptivity.unsqueeze(1)  # [batch, 1, seq, dim]
        
        # Force direction vectors
        force_directions = token_i - token_j  # [batch, seq, seq, dim]
        force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
        normalized_forces = force_directions / (force_magnitudes + 1e-8)
        
        # Direction-sensitive force effects
        direction_scores = torch.einsum('bstn,hd->bshtn', normalized_forces, self.direction_modulator)
        force_field = direction_scores * torch.exp(-force_magnitudes)
        force_attention = torch.sum(force_field, dim=-1)  # [batch, heads, seq, seq]
        
        # Calculate topological edges
        xi = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        xj = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pairs = torch.cat([xi, xj], dim=-1)
        
        # Basic connectivity
        edge_logits = self.edge_projector(pairs).permute(0, 3, 1, 2)  # [batch, heads, seq, seq]
        direct_edges = torch.sigmoid(edge_logits)
        
        # Multi-hop paths - calculate powers of adjacency matrix
        topo_paths = [direct_edges]
        current_paths = direct_edges
        
        for i in range(1, self.hop_levels):
            # Compute next-hop connections
            next_hop = torch.matmul(current_paths, direct_edges) / (seq_len ** 0.5)
            topo_paths.append(next_hop)
            current_paths = next_hop
        
        # Combine different hop lengths with learned weights
        topo_attention = sum(w * path for w, path in zip(F.softmax(self.hop_weights, dim=0), topo_paths))
        
        # Integrate force and topological attention
        balance = torch.sigmoid(self.force_topo_balance)
        combined_attention = balance * force_attention + (1 - balance) * topo_attention
        
        # Apply mask if provided
        if mask is not None:
            combined_attention = combined_attention + mask
            
        # Get attention weights and compute output
        weights = F.softmax(combined_attention, dim=-1)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        
        return self.output(output)


```


```python
import unittest
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def calculate_forces(emissions, receptivity, positions, decay_factor=2, epsilon=1e-8):
    force_directions = emissions.unsqueeze(2) - receptivity.unsqueeze(1)
    
    distances = torch.norm(force_directions, dim=-1, keepdim=True)
    
    normalized_forces = force_directions / (distances + epsilon)
    
    charges = (emissions.unsqueeze(2) * receptivity.unsqueeze(1)).sum(dim=-1, keepdim=True)
    magnitudes = charges / (distances ** decay_factor + epsilon)
    
    forces = normalized_forces * magnitudes
    
    return forces


class ForceDirectedAttention(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.dims = dims
        self.heads = heads
        
        self.force_emitter = nn.Linear(dims, dims)
        
        self.force_receptor = nn.Linear(dims, dims)
        
        self.direction_modulator = nn.Parameter(torch.randn(heads, dims))
        
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)
    
    def forward(self, x):
        batch, seq_len = x.shape[:2]
        
        emissions = self.force_emitter(x)
        receptivity = self.force_receptor(x)
        
        q = self.q_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        
        token_i = emissions.unsqueeze(2)
        token_j = receptivity.unsqueeze(1)
        
        force_directions = token_i - token_j
        force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
        normalized_forces = force_directions / (force_magnitudes + 1e-8)
        
        direction_scores = torch.zeros(batch, self.heads, seq_len, seq_len, device=x.device)
        
        for h in range(self.heads):
            head_modulator = self.direction_modulator[h].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            
            dot_products = normalized_forces * head_modulator
            
            head_scores = dot_products.sum(dim=-1)
            
            direction_scores[:, h] = head_scores
        
        broadcast_magnitudes = force_magnitudes.squeeze(-1).unsqueeze(1)
        
        force_field = direction_scores * torch.exp(-broadcast_magnitudes)
        
        weights = F.softmax(force_field, dim=-1)
        
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        
        return self.output(output)


class TestForceDirectedAttention(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dims = 64
        self.heads = 8
        self.batch_size = 8
        self.seq_len = 8
        
        self.model = ForceDirectedAttention(
            dims=self.dims,
            heads=self.heads
        ).to(self.device)
        
        self.x = torch.randn(self.batch_size, self.seq_len, self.dims, device=self.device)
        
    def test_initialization(self):
        """Test that the model initializes correctly with different parameters."""
        model = ForceDirectedAttention(dims=64, heads=4)
        self.assertEqual(model.dims, 64)
        self.assertEqual(model.heads, 4)
        
        model = ForceDirectedAttention(dims=128, heads=8)
        self.assertEqual(model.dims, 128)
        self.assertEqual(model.heads, 8)
        
        self.assertIsInstance(model.force_emitter, torch.nn.Linear)
        self.assertIsInstance(model.force_receptor, torch.nn.Linear)
        self.assertIsInstance(model.q_proj, torch.nn.Linear)
        self.assertIsInstance(model.k_proj, torch.nn.Linear)
        self.assertIsInstance(model.v_proj, torch.nn.Linear)
        self.assertIsInstance(model.output, torch.nn.Linear)
        
        self.assertEqual(model.direction_modulator.shape, (8, 128))
        
    def test_forward_output_shape(self):
        """Test that the forward pass produces outputs of the expected shape."""
        output = self.model(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dims))
        
        x = torch.randn(5, self.seq_len, self.dims, device=self.device)
        output = self.model(x)
        self.assertEqual(output.shape, (5, self.seq_len, self.dims))
        
        x = torch.randn(self.batch_size, 20, self.dims, device=self.device)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 20, self.dims))
        
    def test_force_direction_calculation(self):
        """Test that force direction vectors are calculated correctly."""
        batch_size = 8
        seq_len = 3
        dims = 4
        heads = 2
        
        model = ForceDirectedAttention(dims=dims, heads=heads)
        
        with torch.no_grad():
            model.force_emitter.weight.fill_(0)
            model.force_emitter.bias.fill_(0)
            model.force_receptor.weight.fill_(0)
            model.force_receptor.bias.fill_(0)
            
            for i in range(dims):
                model.force_emitter.weight[i, i] = 1
                model.force_receptor.weight[i, i] = 1
        
        x = torch.zeros(batch_size, seq_len, dims)
        x[0, 0, :] = torch.tensor([1., 0., 0., 0.])
        x[0, 1, :] = torch.tensor([0., 1., 0., 0.])
        x[0, 2, :] = torch.tensor([0., 0., 1., 0.])
        
        emissions = model.force_emitter(x)
        receptivity = model.force_receptor(x)
        token_i = emissions.unsqueeze(2)
        token_j = receptivity.unsqueeze(1)
        force_directions = token_i - token_j
        
        self.assertTrue(torch.allclose(force_directions[0, 0, 1], torch.tensor([1., -1., 0., 0.])))
        self.assertTrue(torch.allclose(force_directions[0, 0, 2], torch.tensor([1., 0., -1., 0.])))
        self.assertTrue(torch.allclose(force_directions[0, 1, 0], torch.tensor([-1., 1., 0., 0.])))

    def test_normalized_forces(self):
        """Test that normalized force vectors have unit norm."""
        with torch.no_grad():
            emissions = self.model.force_emitter(self.x)
            receptivity = self.model.force_receptor(self.x)
            token_i = emissions.unsqueeze(2)
            token_j = receptivity.unsqueeze(1)
            force_directions = token_i - token_j
            force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
            normalized_forces = force_directions / (force_magnitudes + 1e-8)
        
        norms = torch.norm(normalized_forces, dim=-1)
        mask = force_magnitudes.squeeze(-1) > 1e-6
        self.assertTrue(torch.allclose(norms[mask], torch.ones_like(norms[mask]), atol=1e-6))

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 across the sequence dimension."""
        torch.manual_seed(42)
        np.random.seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dims = 64
        heads = 4
        batch_size = 8
        seq_len = 10
        
        model = ForceDirectedAttention(dims=dims, heads=heads).to(device)
        x = torch.randn(batch_size, seq_len, dims, device=device)
        
        with torch.no_grad():
            emissions = model.force_emitter(x)
            receptivity = model.force_receptor(x)
            
            self.q = model.q_proj(x).reshape(batch_size, seq_len, heads, -1).transpose(1, 2)
            self.k = model.k_proj(x).reshape(batch_size, seq_len, heads, -1).transpose(1, 2)
            self.v = model.v_proj(x).reshape(batch_size, seq_len, heads, -1).transpose(1, 2)
            
            token_i = emissions.unsqueeze(2)
            token_j = receptivity.unsqueeze(1)
            force_directions = token_i - token_j
            force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
            normalized_forces = force_directions / (force_magnitudes + 1e-8)
            
            direction_scores = torch.zeros(batch_size, heads, seq_len, seq_len, device=device)
            
            for h in range(heads):
                head_modulator = model.direction_modulator[h].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                dot_products = normalized_forces * head_modulator               
                head_scores = dot_products.sum(dim=-1)
                direction_scores[:, h] = head_scores
            broadcast_magnitudes = force_magnitudes.squeeze(-1).unsqueeze(1)
            force_field = direction_scores * torch.exp(-broadcast_magnitudes)
            weights = torch.nn.functional.softmax(force_field, dim=-1)
        weight_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-8))

    def test_directional_sensitivity(self):
        """Test that the model is sensitive to directional patterns."""
        batch_size = 8
        seq_len = 8
        dims = 8
        heads = 8
        
        model = ForceDirectedAttention(dims=dims, heads=heads).to(self.device)
        
        seq1 = torch.zeros(batch_size, seq_len, dims, device=self.device)
        seq2 = torch.zeros(batch_size, seq_len, dims, device=self.device)
        
        pattern = [4, 2, 1, 3, 0]
        for i, p in enumerate(pattern):
            seq1[0, i, :] = float(p) / (seq_len - 1)
        
        pattern = [0, 4, 1, 3, 2]
        for i, p in enumerate(pattern):
            seq2[0, i, :] = float(p) / (seq_len - 1)
        
        output1 = model(seq1)
        output2 = model(seq2)
        
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_output_values_deterministic(self):
        """Test that with fixed seeds, outputs are deterministic."""
        torch.manual_seed(42)
        model1 = ForceDirectedAttention(
            dims=self.dims, heads=self.heads
        ).to(self.device)
        
        torch.manual_seed(42)
        model2 = ForceDirectedAttention(
            dims=self.dims, heads=self.heads
        ).to(self.device)
        
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
        
        torch.manual_seed(42)
        x = torch.randn(self.batch_size, self.seq_len, self.dims, device=self.device)
        
        output1 = model1(x)
        output2 = model2(x)
        self.assertTrue(torch.allclose(output1, output2))


    batch_size = 1
    seq_len = 3
    dims = 2

    emissions = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
    receptivity = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]])
    positions = None

    forces = calculate_forces(emissions, receptivity, positions)

    print("Forces:")
    print(forces)

    import matplotlib.pyplot as plt

    emissions = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
    receptivity = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]])
    positions = None

    forces = calculate_forces(emissions, receptivity, positions).squeeze(0)

    plt.figure(figsize=(8, 8))
    for i in range(emissions.shape[1]):
        for j in range(receptivity.shape[1]):
            plt.scatter(emissions[0, i, 0], emissions[0, i, 1], color='blue', label='Emission' if i == 0 else "")
            plt.scatter(receptivity[0, j, 0], receptivity[0, j, 1], color='red', label='Receptivity' if j == 0 else "")
            
            force = forces[i, j].numpy()
            plt.arrow(emissions[0, i, 0], emissions[0, i, 1], force[0], force[1], 
                    color='green', head_width=0.1, length_includes_head=True)

    plt.legend()
    plt.title("Force Vectors Between Tokens")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```


