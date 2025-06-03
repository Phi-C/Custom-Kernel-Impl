import math
import torch

def attention_base(Q, K, V):
    head_dim = Q.size(1)

    scale = 1.0 / math.sqrt(head_dim)
    attn = torch.softmax((Q @ K.T) * scale, dim=-1)
    output = attn @ V
    return output

# flash attn V2's tile implementation
def flash_attn_v2(Q, K, V, O, L):
    q_block_size = 16
    kv_block_size = 16
    
    device = Q.device
    dtype = Q.dtype
    
    query_len = Q.size(0)
    key_value_len = K.size(0)
    head_dim = Q.size(1)

    query_tile = (query_len + q_block_size - 1) // q_block_size
    key_value_tile = (key_value_len + kv_block_size - 1) // kv_block_size


    for i in range(query_tile):
        Q_tile = Q[i * q_block_size: min((i + 1) * q_block_size, query_len), :]
        q_element_num = min(q_block_size, query_len - i * q_block_size)
        O_tile = torch.zeros(q_element_num, head_dim)
        dominant = torch.zeros(q_element_num)
        max_data = torch.full((q_element_num,), -float('inf'))
        for j in range(key_value_tile):
            K_tile = K[j * kv_block_size: min((j + 1) * kv_block_size, key_value_len), :]           # [kv_block_size, head_dim]
            V_tile = V[j * kv_block_size: min((j + 1) * kv_block_size, key_value_len), :]           # [kv_block_size, head_dim]
            S_tile = torch.matmul(Q_tile, K_tile.T) / math.sqrt(head_dim)                           # [q_block_size, kv_block_size]
            
            max_tile = torch.max(S_tile, dim=1, keepdim=True).values                                # [q_block_size, kv_block_size]
            max_tile_squeezed = max_tile.squeeze(1)
            
            prev_max_data = max_data
            max_data = torch.maximum(max_data, max_tile_squeezed)                                   # [q_block_size]
            
            scale_prev = torch.exp(prev_max_data - max_data)
            scale_cur = torch.exp(max_tile_squeezed - max_data)
            
            P_tile = torch.exp(S_tile - max_data.unsqueeze(1))

            dominant = dominant * scale_prev + torch.sum(P_tile, dim=1)                             # [q_block_size]
            O_tile = scale_prev.unsqueeze(1) * O_tile + (P_tile @ V_tile)                           # [q_block_size, head_dim]
            
        O_tile = O_tile / dominant.unsqueeze(1)
        O[i * q_block_size: i * q_block_size + q_element_num, :] = O_tile
        L[i * q_block_size: i * q_block_size + q_element_num, 0] = dominant
        
    return O, L

    
if __name__ == "__main__":
    query_len = 13
    key_value_len = 1028
    head_dim = 128

    Q = torch.randn(query_len, head_dim)
    K = torch.randn(key_value_len, head_dim)
    V = torch.randn(key_value_len, head_dim)
    O = torch.zeros_like(Q)
    L = torch.zeros(query_len, 1)
    
    baseline = attention_base(Q, K, V)
    fa_result, dominants = flash_attn_v2(Q, K, V, O, L)
    print(f"baseline = {baseline}")
    print(f"fa result = {fa_result}")
    assert torch.allclose(baseline, fa_result, rtol=3e-2)

        

