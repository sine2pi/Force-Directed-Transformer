
class ForceDirectedAttention(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.dims = dims
        self.heads = heads
        
        # Each token learns how to emit force vectors
        self.force_emitter = nn.Linear(dims, dims)
        
        # Each token learns its response to incoming forces
        self.force_receptor = nn.Linear(dims, dims)
        
        # Direction-dependent strength modulation
        self.direction_modulator = nn.Parameter(torch.randn(heads, dims))
        
        # Force calculation projections
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)
    
    def forward(self, x):
        batch, seq_len = x.shape[:2]
        
        # Calculate emission and reception properties
        emissions = self.force_emitter(x)
        receptivity = self.force_receptor(x)
        
        # Standard projections
        q = self.q_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        
        # Calculate force vectors between all pairs of tokens
        # Instead of simple dot products, we compute vector forces
        token_i = emissions.unsqueeze(2)  # [batch, seq, 1, dim]
        token_j = receptivity.unsqueeze(1)  # [batch, 1, seq, dim]
        
        # Force direction vector (not just magnitude)
        force_directions = token_i - token_j  # [batch, seq, seq, dim]
        force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
        normalized_forces = force_directions / (force_magnitudes + 1e-8)
        
        # Create a tensor to hold the direction scores for each head
        direction_scores = torch.zeros(batch, self.heads, seq_len, seq_len, device=x.device)
        
        # Calculate direction scores for each head separately
        for h in range(self.heads):
            # Get the direction modulator for this head
            # Shape: [1, 1, 1, dims]
            head_modulator = self.direction_modulator[h].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            
            # Calculate dot products between normalized forces and head modulator
            # Shape: [batch, seq_len, seq_len, dims]
            dot_products = normalized_forces * head_modulator
            
            # Sum over the dims dimension to get scores
            # Shape: [batch, seq_len, seq_len]
            head_scores = dot_products.sum(dim=-1)
            
            # Store the scores for this head
            direction_scores[:, h] = head_scores
        
        # Reshape force_magnitudes for broadcasting with direction_scores
        # Shape: [batch, 1, seq_len, seq_len]
        broadcast_magnitudes = force_magnitudes.squeeze(-1).unsqueeze(1)
        
        # Combined force effect (analogous to attention scores)
        # Shape: [batch, heads, seq_len, seq_len]
        force_field = direction_scores * torch.exp(-broadcast_magnitudes)
        
        # Convert force field to attention weights
        weights = F.softmax(force_field, dim=-1)
        
        # Apply weights to values
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        
        return self.output(output)

