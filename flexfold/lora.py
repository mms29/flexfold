import torch.nn as nn
import importlib
deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
ds4s_is_installed = deepspeed_is_installed and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
if deepspeed_is_installed:
    import deepspeed
import torch

FROZEN = 0
LORA = 1
FULL = 2

lora_light = {
    "structure_module":{
        "linear_in" : LORA,#,147.84 K
        "ipa.linear_q" :LORA,#, 73.92 K
        "ipa.linear_q_points.linear" :LORA,#, 55.44 K
        "ipa.linear_kv" : LORA,#,147.84 K
        "ipa.linear_kv_points.linear" : LORA,#,166.32 K
        "ipa.linear_b": LORA,#,: 1.55 K
        "ipa.linear_out" : FROZEN,#,811.39 K
        "transition.layers.0.linear_1" : LORA,#,147.84 K
        "transition.layers.0.linear_2" : FROZEN,#,147.84 K
        "transition.layers.0.linear_3" : FROZEN,#,147.84 K
        "bb_update.linear": LORA,#,: 2.31 K
        "angle_resnet.linear_in" :LORA,#, 49.28 K
        "angle_resnet.linear_initial" :LORA,#, 49.28 K
        "angle_resnet.layers.0.linear_1" :LORA,#, 16.51 K
        "angle_resnet.layers.0.linear_2" :FROZEN,#, 16.51 K
        "angle_resnet.layers.1.linear_1" :LORA,#, 16.51 K
        "angle_resnet.layers.1.linear_2" :FROZEN,#, 16.51 K
        "angle_resnet.linear_out": FROZEN,#, : 1.81 K
    }
}

lora_heavy = {
    "structure_module":{
        "linear_in" : LORA,#,147.84 K
        "ipa.linear_q" :LORA,#, 73.92 K
        "ipa.linear_q_points.linear" :LORA,#, 55.44 K
        "ipa.linear_kv" : LORA,#,147.84 K
        "ipa.linear_kv_points.linear" : LORA,#,166.32 K
        "ipa.linear_b": LORA,#,: 1.55 K
        "ipa.linear_out" : LORA,#,811.39 K
        "transition.layers.0.linear_1" : LORA,#,147.84 K
        "transition.layers.0.linear_2" : LORA,#,147.84 K
        "transition.layers.0.linear_3" : LORA,#,147.84 K
        "bb_update.linear": LORA,#,: 2.31 K
        "angle_resnet.linear_in" :LORA,#, 49.28 K
        "angle_resnet.linear_initial" :LORA,#, 49.28 K
        "angle_resnet.layers.0.linear_1" :LORA,#, 16.51 K
        "angle_resnet.layers.0.linear_2" :LORA,#, 16.51 K
        "angle_resnet.layers.1.linear_1" :LORA,#, 16.51 K
        "angle_resnet.layers.1.linear_2" :LORA,#, 16.51 K
        "angle_resnet.linear_out": LORA,#, : 1.81 K
    }
}

lora_frozen = {k:{k2:FROZEN for k2,_ in v.items()} for k,v in lora_light.items()}
no_lora = {k:{k2:FULL for k2,_ in v.items()} for k,v in lora_light.items()}

class LoRA(nn.Module):
    def __init__(self, original_layer, rank: int):
        super(LoRA, self).__init__()
        self.rank = rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Freeze the original weights
        self.original_weight = nn.Parameter(original_layer.weight, requires_grad=False)

        # Copy bias and precision
        self.bias =         original_layer.bias
        self.precision  =original_layer.precision

        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(self.out_features, self.rank))  # shape: [out_features, rank]
        self.B = nn.Parameter(torch.zeros(self.rank, self.in_features))  # shape: [rank, in_features]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.dtype
        deepspeed_is_initialized = (
                deepspeed_is_installed and
                deepspeed.comm.comm.is_initialized()
        )
        weight_update = torch.matmul(self.A, self.B)

        if self.precision is not None:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=self.precision) if self.bias is not None else None
                return nn.functional.linear(input.to(dtype=self.precision),
                                            (self.original_weight + weight_update).to(dtype=self.precision),
                                            bias).to(dtype=d)

        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return nn.functional.linear(input, (self.original_weight + weight_update).to(dtype=d), bias)


        return nn.functional.linear(input, self.original_weight + weight_update, self.bias)

def set_module_by_name(model, module_name, new_module):
    modules = module_name.split('.')
    current_module = model
    for submodule in modules[:-1]:
        current_module = getattr(current_module, submodule)
    setattr(current_module, modules[-1], new_module)


def get_module_config(config, name):
    module = name.split(".")[0]
    for n in config[module]:
        if name.endswith(n):
            return config[module][n]
    raise KeyError("name not in config")


def apply_lora_config_to_model(model, config, rank=4):

    stats = {
        "lora":0,
        "frozen":0,
        "full":0,
        "other":0,
    }
    sumparam = lambda x : sum([t.numel() for t in x.parameters()])
    def setgrad(x, val):
        for p  in x.parameters():
            p.requires_grad =val

    for name, module in [(n,m) for n,m in model.named_modules()]:
        if isinstance(module, nn.Linear):
            try : 
                layer_config = get_module_config(config=config, name=name)
            except KeyError:
                continue
            if layer_config == LORA:
                set_module_by_name(model, name, LoRA(module, rank=rank))  
                stats["lora"] += sumparam(module)
            elif layer_config == FROZEN:
                setgrad(module, False)
                stats["frozen"] += sumparam(module)
            elif layer_config == FULL:
                setgrad(module, True)
                stats["full"] += sumparam(module)
            else:
                raise
        else:
            stats["other"] += sumparam(module)

    # for s in stats:
    #     stats[s] /= 1e6
    # print(stats)
    return model