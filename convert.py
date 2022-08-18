from collections import OrderedDict

import torch
import nni.retiarii.hub.pytorch as hub


def convert2supernet(official_state_dict, nni_supernet_state_dict):
    state_dict = OrderedDict()

    for k, v in official_state_dict.items():
        if k == "pos_embed":
            nv = nni_supernet_state_dict["pos_embed.pos_embed"]
            state_dict["pos_embed.pos_embed"] = v[..., :nv.size(-1)]
        elif k == "cls_token":
            nv = nni_supernet_state_dict["cls_token.cls_token"]
            state_dict["cls_token.cls_token"] = v[..., :nv.size(-1)]
        elif k == "patch_embed_super.proj.weight":
            nv = nni_supernet_state_dict["patch_embed.weight"]
            state_dict["patch_embed.weight"] = v[:nv.size(0)]
        elif k == "patch_embed_super.proj.bias":
            nv = nni_supernet_state_dict["patch_embed.bias"]
            state_dict["patch_embed.bias"] = v[:nv.size(0)]
        elif k in ["norm.weight", "norm.bias", "head.weight", "head.bias"]:
            nv = nni_supernet_state_dict[k]
            indices = [slice(0, min(i, j)) for i, j in zip(v.shape, nv.shape)]
            state_dict[k] = v[indices]
        elif k.startswith("blocks"):
            if "qkv.weight" in k:
                k = "blocks."+k
                nk = k.replace("qkv", "q")
                nv = nni_supernet_state_dict[nk]
                state_dict[nk] = v[0:3*nv.size(0):3, :nv.size(1)]
                nk = k.replace("qkv", "k")
                state_dict[nk] = v[1:3*nv.size(0):3, :nv.size(1)]
                nk = k.replace("qkv", "v")
                state_dict[nk] = v[2:3*nv.size(0):3, :nv.size(1)]

            elif "qkv.bias" in k:
                k = "blocks."+k
                nk = k.replace("qkv", "q")
                nv = nni_supernet_state_dict[nk]
                state_dict[nk] = v[:nv.size(0)]
                nk = k.replace("qkv", "k")
                state_dict[nk] = v[nv.size(0):2*nv.size(0)]
                nk = k.replace("qkv", "v")
                state_dict[nk] = v[2*nv.size(0):3*nv.size(0)]
            else:
                k = "blocks."+k
                nv = nni_supernet_state_dict[k]
                indices = [slice(0, min(i, j)) for i, j in zip(v.shape, nv.shape)]
                state_dict[k] = v[indices]
        else:
            pass
    
    return state_dict



def convert2subnet(official_state_dict, nni_subnet_state_dict):
    state_dict = OrderedDict()
    for k, v in official_state_dict.items():
        if k == "pos_embed":
            nv = nni_subnet_state_dict["pos_embed.pos_embed"]
            state_dict["pos_embed.pos_embed"] = v[..., :nv.size(-1)]
        elif k == "cls_token":
            nv = nni_subnet_state_dict["cls_token.cls_token"]
            state_dict["cls_token.cls_token"] = v[..., :nv.size(-1)]
        elif k == "patch_embed_super.proj.weight":
            nv = nni_subnet_state_dict["patch_embed.weight"]
            state_dict["patch_embed.weight"] = v[:nv.size(0)]
        elif k == "patch_embed_super.proj.bias":
            nv = nni_subnet_state_dict["patch_embed.bias"]
            state_dict["patch_embed.bias"] = v[:nv.size(0)]
        elif k in ["norm.weight", "norm.bias", "head.weight", "head.bias"]:
            nv = nni_subnet_state_dict[k]
            indices = [slice(0, min(i, j)) for i, j in zip(v.shape, nv.shape)]
            state_dict[k] = v[indices]

        elif k.startswith("blocks"):
            # skip layers
            layer = k.split(".")[1]
            if f"blocks.{layer}.fc2.bias" not in nni_subnet_state_dict:
                continue
            if "qkv.weight" in k:
                nk = k.replace("qkv", "q")
                nv = nni_subnet_state_dict[nk]
                state_dict[nk] = v[0:3*nv.size(0):3, :nv.size(1)]
                nk = k.replace("qkv", "k")
                state_dict[nk] = v[1:3*nv.size(0):3, :nv.size(1)]
                nk = k.replace("qkv", "v")
                state_dict[nk] = v[2:3*nv.size(0):3, :nv.size(1)]

            elif "qkv.bias" in k:
                nk = k.replace("qkv", "q")
                nv = nni_subnet_state_dict[nk]
                state_dict[nk] = v[:nv.size(0)]
                nk = k.replace("qkv", "k")
                state_dict[nk] = v[nv.size(0):2*nv.size(0)]
                nk = k.replace("qkv", "v")
                state_dict[nk] = v[2*nv.size(0):3*nv.size(0)]
            else:
                nv = nni_subnet_state_dict[k]
                indices = [slice(0, min(i, j)) for i, j in zip(v.shape, nv.shape)]
                state_dict[k] = v[indices]
        else:
            pass

    return state_dict


size = "base"
name = f"autoformer-{size}"
init_kwargs = {'qkv_bias': True, 'drop_rate': 0.0, 'drop_path_rate': 0.1, 'global_pool': True, 'num_classes': 1000}
if name == 'autoformer-tiny':
    mlp_ratio = [3.5, 3.5, 3.0, 3.5, 3.0, 3.0, 4.0, 4.0, 3.5, 4.0, 3.5, 4.0, 3.5] + [3.0]
    num_head = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3] + [3]
    arch = {
        'embed_dim': 192,
        'depth': 13
    }
    for i in range(14):
        arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
        arch[f'num_head_{i}'] = num_head[i]

    init_kwargs.update({
        'search_embed_dim': (240, 216, 192),
        'search_mlp_ratio': (4.0, 3.5, 3.0),
        'search_num_heads': (4, 3),
        'search_depth': (14, 13, 12),
    })
elif name == 'autoformer-small':
    mlp_ratio = [3.0, 3.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 4.0] + [3.0]
    num_head = [6, 6, 5, 7, 5, 5, 5, 6, 6, 7, 7, 6, 7] + [5]
    arch = {
        'embed_dim': 384,
        'depth': 13
    }
    for i in range(14):
        arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
        arch[f'num_head_{i}'] = num_head[i]

    init_kwargs.update({
        'search_embed_dim': (448, 384, 320),
        'search_mlp_ratio': (4.0, 3.5, 3.0),
        'search_num_heads': (7, 6, 5),
        'search_depth': (14, 13, 12),
    })

elif name == 'autoformer-base':
    mlp_ratio = [3.5, 3.5, 4.0, 3.5, 4.0, 3.5, 3.5, 3.0, 4.0, 4.0, 3.0, 4.0, 3.0, 3.5] + [3.0, 3.0]
    num_head = [9, 9, 9, 9, 9, 10, 9, 9, 10, 9, 10, 9, 9, 10] + [8, 8]
    arch = {
        'embed_dim': 576,
        'depth': 14
    }
    for i in range(16):
        arch[f'mlp_ratio_{i}'] = mlp_ratio[i]
        arch[f'num_head_{i}'] = num_head[i]

    init_kwargs.update({
        'search_embed_dim': (624, 576, 528),
        'search_mlp_ratio': (4.0, 3.5, 3.0),
        'search_num_heads': (10, 9, 8),
        'search_depth': (16, 15, 14),
    })

model_space = hub.AutoformerSpace(**init_kwargs)

official_state_dict = torch.load(f"weights/official-supernet-{size}.pth", map_location="cpu")["model"]

nni_supernet_state_dict = model_space.state_dict()
state_dict = convert2supernet(official_state_dict, nni_supernet_state_dict)
model_space.load_state_dict(state_dict)
# torch.save(model_space.state_dict(), f"weights/supernet-{size}.pth")

model = model_space.load_searched_model(f"autoformer-{size}", pretrained=False, download=False)
nni_subnet_state_dict = model.state_dict()
state_dict = convert2subnet(official_state_dict, nni_subnet_state_dict)
model.load_state_dict(state_dict)
# torch.save(model.state_dict(), f"weights/subnet-{size}.pth")

# for k, v in official_state_dict.items():
#     print(k, v.shape)
# for k, v in nni_supernet_state_dict.items():
#     print(k, v.shape)
# for k, v in nni_subnet_state_dict.items():
#     print(k, v.shape)
