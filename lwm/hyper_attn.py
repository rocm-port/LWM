import jax
import jax.numpy as jnp
import flax.linen as nn
import LWM.lwm.generic_random
from functools import partial

class HyperAttention(nn.Module):
    def __init__(self, input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=4096, dtype=jnp.float32, rng=LWM.lwm.generic_random.generic_random()):
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.rng = rng
        self.lsh = AngularLSH(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim), rng = self.rng, dtype=dtype)

    def forward(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, scale=None, causal=False, return_lse=False):
        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape
        scale = dim ** (-0.5) if scale is None else scale

        # Without causal masking
        if not causal:
            attn, lse = self.forward_no_causal_mask(query, key, value, scale)

        # With causal masking
        else:
            if n_key <= self.min_seq_len:
                attn, lse = exact_attention(query, key, value, scale, causal=True)
            else:

                # If n_query is odd we pad inputs by adding all-zero rows
                if n_query % 2:
                    query = jnp.pad(query, (0,0,0,1), mode='constant',value=0.)
                    key = jnp.pad(key, (0,0,0,1), mode='constant',value=0.)
                    value = jnp.pad(value, (0,0,0,1), mode='constant',value=0.)

                q_bd = query.reshape(batch_size, 2*n_heads, query.shape[2]//2, query.shape[-1])
                k_bd = key.reshape(batch_size, 2*n_heads, key.shape[2]//2, key.shape[-1])
                v_bd = value.reshape(batch_size, 2*n_heads, key.shape[2]//2, value.shape[-1])

                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)

                attn_bd = attn_bd.reshape(batch_size, n_heads, -1, dim)

                lse_bd = lse_bd.reshape(batch_size, n_heads, -1, 1)

                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2]//2:, :],
                    key[:, :, :key.shape[2]//2, :],
                    value[:, :, :key.shape[2]//2, :], scale)

                attn_up, lse_up = attn_bd[:,:,:query.shape[2]//2,:], lse_bd[:,:,:query.shape[2]//2,:]
                attn_down, lse_down = add_self_attentions(
                    attn_bd[:,:,query.shape[2]//2:,:],
                    lse_bd[:,:,query.shape[2]//2:,:],
                    attn_unmasked,
                    lse_unmasked)

                attn = jnp.concatenate((attn_up, attn_down), dim=-2)
                lse = jnp.concatenate((lse_up, lse_down), dim=-2)

                # If n_query was odd exclude the last rows
                if n_query % 2:
                    attn = attn[:,:,:-1,:]
                    lse = lse[:,:,:-1,:]

        if not return_lse:
            return attn
        else:
            return attn, lse

    def forward_no_causal_mask(self, query, key, value, scale):
        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        if self.min_seq_len > n_query:
            return exact_attention(query, key, value, scale, causal=False)

        # 1. Sorted block-diagonal via sortLSH
        query_sort_idx = jnp.argsort(self.lsh.hash(query), axis=2, stable=True) # batch_size x head_size x n
        key_sort_idx = jnp.argsort(self.lsh.hash(key), axis=2, stable=True)
        query_sort_idx_inv = jnp.argsort(query_sort_idx, axis=2, stable=True) # for recovering the row order

        key_block_size = self.block_size

        query_sorted = indexing(query, query_sort_idx, key_block_size)
        key_sorted = indexing(key, key_sort_idx, key_block_size)
        value_sorted = indexing(value, key_sort_idx, key_block_size)

        if key_block_size > 0:

            num_blocks = key_sorted.shape[2] // key_block_size
            query_block_size = query_sorted.shape[2] // num_blocks

            # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
            query_split_per_block = query_sorted.reshape(-1, 1, query_block_size, dim)
            key_split_per_block = key_sorted.reshape(-1, 1, key_block_size, dim)
            value_split_per_block = value_sorted.reshape(-1, 1, key_block_size, dim)

            attn_block, lse_block = exact_attention(
                query_split_per_block, key_split_per_block, value_split_per_block,
                softmax_scale=scale, causal=False)

            attn_block = attn_block.reshape(batch_size, head_size, query_sorted.shape[2], -1)

            lse_block = lse_block.reshape(batch_size, head_size, query_sorted.shape[2], -1)

            # When inputs are padded, then unpad them
            if query_sorted.shape[2] != n_query: #query.shape[2]:
                attn_block, lse_block = attn_block[:,:,:n_query,:], lse_block[:,:,:n_query,:]
                query_sorted = query_sorted[:,:,:n_query,:]
                key_sorted = key_sorted[:,:,:n_key,:]
                value_sorted = value_sorted[:,:,:n_key,:]

        else:
            query_block_size = -1
            query_block_size = -1
            attn_block, lse_block = 0, 0

        # 2. Residual low-rank part via uniform sampling
        # Sample indices uniformly at random
        sample_size = self.sample_size
        if sample_size > 0 and (n_query > query_block_size) and (n_key > key_block_size):
            sampled_set = self.rng.randint((batch_size, head_size, sample_size), n_key)

            # Compute mask for hiding A_ij computed in block-diagonal attention
            offset_n = jnp.expand_dims(jnp.arange(n_query), (0, -1))
            weights = n_key / sample_size
            value_subset = indexing(value_sorted, sampled_set)
            key_subset = indexing(key_sorted, sampled_set)

            block_mask = (offset_n // query_block_size) == (sampled_set // key_block_size).reshape(-1, 1, sample_size)
            block_mask = block_mask.reshape(batch_size, head_size, -1, sample_size)
            block_mask = block_mask.astype(query_sorted.dtype)
            block_mask *= jnp.finfo(query_sorted.dtype).min # adding -inf added to QK^T

            attn_res, lse_res = exact_attention(query_sorted, key_subset, value_subset, scale, causal=False, bias=block_mask)

            lse_res = lse_res + jnp.log(weights)

            # Add two attentions
            if key_block_size > 0:
                attn, lse = add_self_attentions(attn_block, lse_block, attn_res, lse_res)
            else:
                attn, lse = attn_res, lse_res
        else:
            attn, lse = attn_block, lse_block

        # Re-order rows with the inverse order for query_sorted -> query
        attn = indexing(attn, query_sort_idx_inv)
        lse = indexing(lse, query_sort_idx_inv)
        return attn, lse

class AngularLSH(nn.Module):
    def __init__(self, num_projs, dim, rng=None, dtype = jnp.float32):
        super().__init__()
        self.num_projs = num_projs

        if num_projs > 0:
            if (rng is not None):
                self.proj_dir = rng.normal(dim + (num_projs,), dtype=dtype)
            else:
                self.proj_dir = jax.random.normal(jax.random.key(0), dim + (num_projs,), dtype=dtype)

            self.perm = self._unit_hamming_distance_array(self.num_projs)
            self.enc_vec = 2 ** jnp.arange(self.num_projs).reshape(1, 1, 1, self.num_projs)

    def _unit_hamming_distance_array(self, size_n):
        if size_n == 1:
            return jnp.arange(2)
        a = self._unit_hamming_distance_array(size_n - 1)
        return jnp.concatenate((a, jnp.flip(a) + 2 ** (size_n - 1)))

    def hash(self, mat):
        if self.num_projs < 0:
            return jnp.zeros(mat.shape[:-1], device=mat.device, dtype=jnp.int32)
        #mask =  mat @ self.proj_dir
        #mask = jnp.einsum('...nd,...dr -> ...nr', mat, self.proj_dir)
        mask = jnp.matmul(mat, self.proj_dir)
        mask = mask > 0
        bin_ids = (mask * self.enc_vec).sum(-1)
        return self.perm[bin_ids]

    def __repr__(self):
        return f"AngularLSH(num_proj={self.num_projs}, proj_dir.shape={self.proj_dir.shape})"

def indexing(x: jnp.ndarray, indices: jnp.ndarray, chunk_size: int=-1):
    """
    inputs:
        - x: 4d-tensor with shape [b, h, n, d]
        - indices: 3d-tensor with shape [b, h, s] where each entry should be in [0, n-1]
    output:
        - out: 4d-tensor with shape [b, h, s, d] where out[i,j] = x[i,j][indices[i,j],:]

    A naive implementation:
        out = torch.zeros(b, h, s, d)
        for i in range(b):
            for j in range(h):
                out[i,j] = x[i,j][idx[i,j],:]
        return out
    """
    #gathered = jnp.take(x, jnp.broadcast_to(jnp.expand_dims(indices, -1),indices.shape + (x.shape[-1],)), 2)
    gathered = jax.vmap(jax.vmap(lambda a, b:a[b], (0,0)), (0,0))(x, indices)
    if chunk_size < 0 or (chunk_size > 0 and x.shape[-2] % chunk_size == 0):
        return gathered
        #return x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
    else:
        #x = x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        x = gathered
        new_n = jnp.ceil(x.shape[2] / chunk_size) * chunk_size
        if new_n <= 0 or new_n - x.shape[2] <= 0:
            import pdb; pdb.set_trace();
        return jnp.pad(x, (0,0,0,new_n-x.shape[2]), mode='constant',value=0.)

def add_self_attentions(attn1, lse1, attn2, lse2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
    output:
        - attn
        = (attn1 * exp(lse1) + attn2 * exp(lse2)) / (exp(lse1) + exp(lse2))
        = (attn1 + attn2 * exp(lse2 - lse1)) / (1 + exp(lse2-lse1))
        = attn1 * c + attn2 * (1-c), where c=1/(1 + exp(lse2-lse1)),
        - lse
        = log(exp(lse1) + exp(lse2))
        = log(exp(lse1) * (1 + exp(lse2 - lse1)))
        = lse1 + log(1 + exp(lse2 - lse1)) = lse1 - log(c)
    """
    c = (1 / (1 + jnp.exp(lse2 - lse1))).astype(dtype=attn1.dtype)
    attn = c * attn1 + (1-c) * attn2
    lse = lse1 - jnp.log(c + jnp.finfo(lse1.dtype).eps)
    return attn, lse

def exact_attention(query, key, value, softmax_scale, causal=False, bias=None):
    # inline standard exact attention for now ...
    # put in flash attention or blockwise attention or something better later
    qk = query @ jnp.matrix_transpose(key) * softmax_scale
    if causal:
        qk += (jnp.ones(query.shape[2], key.shape[2], device=query.device) * jnp.finfo(query.dtype).min).triu(1).reshape(1,1,query.shape[2], key.shape[2])
    # flash attention adds bias after causal mask
    if (bias is not None):
        qk = qk + bias
    out = jax.nn.softmax(qk, axis=-1) @ value
    lse = jax.scipy.special.logsumexp(qk, axis=-1, keepdims=True)
    return out, lse
