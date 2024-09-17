LongNetDecoder_12_layers_768_dim = {
        'decoder_layers': 12,  # Matching the encoder's depth (can be fewer if needed)
        'decoder_embed_dim': 768,  # Same as encoder embedding dimension
        'decoder_ffn_embed_dim': 3072,  # Matching encoder feed-forward network dimension
        'decoder_attention_heads': 16,  # Match encoder's attention heads
        'dilated_ratio': '[1, 2, 4, 8, 16]',  # Keep the same for compatible long-range attention
        'segment_length': '[1024, 2048, 4096, 8192, 16384]',  # Match encoder segmentation
        'flash_attention': True,  # Ensure same attention mechanisms
        'block_shift': True,  # Match encoder behavior
        'use_xmoe': False,  # Keeping MoE disabled for consistency with encoder
        'moe_top1_expert': False,
        'moe_freq': 0,
        'moe_expert_count': 0,
    }