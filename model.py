
from nni.nas.hub.pytorch import AutoformerSpace


def builder(name: str, num_classes: int = 1000):
    init_kwargs = {'qkv_bias': True, 'drop_rate': 0.0, 'drop_path_rate': 0.1, 'global_pool': True, 'num_classes': num_classes}
    # if name == 'tiny':
    #     init_kwargs.update({
    #         'search_embed_dim': (240, 216, 192),
    #         'search_mlp_ratio': (4.0, 3.5, 3.0),
    #         'search_num_heads': (4, 3),
    #         'search_depth': (14, 13, 12),
    #     })
    # elif name == 'small':
    #     init_kwargs.update({
    #         'search_embed_dim': (448, 384, 320),
    #         'search_mlp_ratio': (4.0, 3.5, 3.0),
    #         'search_num_heads': (7, 6, 5),
    #         'search_depth': (14, 13, 12),
    #     })
    # elif name == 'base':
    #     init_kwargs.update({
    #         'search_embed_dim': (624, 576, 528),
    #         'search_mlp_ratio': (4.0, 3.5, 3.0),
    #         'search_num_heads': (10, 9, 8),
    #         'search_depth': (16, 15, 14),
    #     })
    # else:
    #     raise ValueError(f"Unknown model name {name}.")
    if name == 'tiny':
        init_kwargs.update({
            'search_embed_dim': (192, 216, 240),
            'search_mlp_ratio': (3.0, 3.5, 4.0),
            'search_num_heads': (3, 4),
            'search_depth': (12, 13, 14),
        })
    elif name == 'small':
        init_kwargs.update({
            'search_embed_dim': (320, 384, 448),
            'search_mlp_ratio': (3.0, 3.5, 4.0),
            'search_num_heads': (5, 6, 7),
            'search_depth': (12, 13, 14),
        })
    elif name == 'base':
        init_kwargs.update({
            'search_embed_dim': (528, 576, 624),
            'search_mlp_ratio': (3.0, 3.5, 4.0),
            'search_num_heads': (8, 9, 10),
            'search_depth': (14, 15, 16),
        })
    else:
        raise ValueError(f"Unknown model name {name}.")

    model_space = AutoformerSpace(**init_kwargs)

    return model_space
