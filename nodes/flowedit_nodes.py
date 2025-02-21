import torch
from tqdm import trange

from comfy.samplers import KSAMPLER, CFGGuider, sampling_function


class FlowEditGuider(CFGGuider):
    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self.cfgs = {}
        self.num_repeats = 1

    def set_conds(self, **kwargs):
        self.inner_set_conds(kwargs)

    def set_cfgs(self, **kwargs):
        self.cfgs = {**kwargs}

    def set_num_repeats(self, num_repeats):
        self.num_repeats = num_repeats

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        latent_type = model_options['transformer_options']['latent_type']
        positive = self.conds.get(f'{latent_type}_positive', None)
        negative = self.conds.get(f'{latent_type}_negative', None)
        cfg = self.cfgs.get(latent_type, self.cfg)

        if self.num_repeats == 1:
            return sampling_function(self.inner_model, x, timestep, negative, positive, cfg, model_options=model_options, seed=seed)

        # Multiple samples case
        predictions = None
        for i in range(self.num_repeats):
            current_seed = None if seed is None else seed + i
            current_pred = sampling_function(
                self.inner_model, x, timestep, negative, positive, cfg, 
                model_options=model_options, seed=current_seed
            )
            if predictions is None:
                predictions = current_pred
            else:
                predictions += current_pred
        
        return predictions / self.num_repeats


# Basic node for model compatibility
class HYFlowEditGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "source_cond": ("CONDITIONING", ),
                        "target_cond": ("CONDITIONING", ),
                     }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "hunyuanloom"

    def get_guider(self, model, source_cond, target_cond):
        guider = FlowEditGuider(model)
        guider.set_conds(source_positive=source_cond, target_positive=target_cond)
        guider.set_cfg(1.0)
        return (guider,)

class HYFlowEditGuiderAdvNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "source_cond": ("CONDITIONING", ),
                        "target_cond": ("CONDITIONING", ),
                        "num_repeats": ("INT", {"default": 1, "min": 1, "max": 10}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "hunyuanloom"

    def get_guider(self, model, source_cond, target_cond, num_repeats):
        guider = FlowEditGuider(model)
        guider.set_conds(source_positive=source_cond, target_positive=target_cond)
        guider.set_cfg(1.0)
        guider.set_num_repeats(num_repeats)
        return (guider,)


class HYFlowEditGuiderCFGNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "source_cond": ("CONDITIONING", ),
                        "source_uncond": ("CONDITIONING", ),
                        "target_cond": ("CONDITIONING", ),
                        "target_uncond": ("CONDITIONING", ),
                        "source_cfg": ("FLOAT", {"default": 2, "min": 0, "max": 0xffffffffffffffff, "step": 0.01 }),
                        "target_cfg": ("FLOAT", {"default": 4.5, "min": 0, "max": 0xffffffffffffffff, "step": 0.01 }),
                     }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "hunyuanloom"

    def get_guider(self, model, source_cond, source_uncond, target_cond, target_uncond, source_cfg, target_cfg):
        guider = FlowEditGuider(model)
        guider.set_conds(source_positive=source_cond, source_negative=source_uncond, 
                        target_positive=target_cond, target_negative=target_uncond)
        guider.set_cfgs(source=source_cfg, target=target_cfg)
        return (guider,)


class HYFlowEditGuiderCFGAdvNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "source_cond": ("CONDITIONING", ),
                        "source_uncond": ("CONDITIONING", ),
                        "target_cond": ("CONDITIONING", ),
                        "target_uncond": ("CONDITIONING", ),
                        "source_cfg": ("FLOAT", {"default": 2, "min": 0, "max": 0xffffffffffffffff, "step": 0.01 }),
                        "target_cfg": ("FLOAT", {"default": 4.5, "min": 0, "max": 0xffffffffffffffff, "step": 0.01 }),
                        "num_repeats": ("INT", {"default": 1, "min": 1, "max": 10}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "hunyuanloom"

    def get_guider(self, model, source_cond, source_uncond, target_cond, target_uncond, source_cfg, target_cfg, num_repeats):
        guider = FlowEditGuider(model)
        guider.set_conds(source_positive=source_cond, source_negative=source_uncond, 
                        target_positive=target_cond, target_negative=target_uncond)
        guider.set_cfgs(source=source_cfg, target=target_cfg)
        guider.set_num_repeats(num_repeats)
        return (guider,)

def get_flowedit_sample(skip_steps, refine_steps, generator):
    @torch.no_grad()
    def flowedit_sample(model, x_init, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        
        model_options = extra_args.get('model_options', {})
        transformer_options = model_options.get('transformer_options', {})
        transformer_options = {**transformer_options}
        model_options['transformer_options'] = transformer_options
        extra_args['model_options'] = model_options

        source_extra_args = {**extra_args, 'model_options': { 'transformer_options': { 'latent_type': 'source'} }}

        sigmas = sigmas[skip_steps:]

        x_tgt = x_init.clone()
        N = len(sigmas)-1
        s_in = x_init.new_ones([x_init.shape[0]])
        noise_mask = extra_args.get('denoise_mask', None)
        if noise_mask is None:
            noise_mask = torch.ones_like(x_init)
        else:
            extra_args['denoise_mask'] = None
            source_extra_args['denoise_mask'] = None

        for i in trange(N, disable=disable):
            sigma = sigmas[i]
            noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)

            zt_src = (1-sigma)*x_init + sigma*noise
            
            if i < N-refine_steps:
                zt_tgt = x_tgt + zt_src - x_init
                vt_src = model(zt_src, sigma*s_in, **source_extra_args)
            else:
                if i == N-refine_steps:
                    zt_tgt = x_tgt + (zt_src - x_init)
                    x_tgt = x_tgt + (zt_src - x_init) * noise_mask
                else:
                    zt_tgt = x_tgt * (noise_mask) + (1-noise_mask) * ( (1-sigma)*x_tgt + sigma*noise )
                vt_src = 0
                
            transformer_options['latent_type'] = 'target'
            vt_tgt = model(zt_tgt, sigma*s_in, **extra_args)
            
            v_delta = vt_tgt - vt_src
            x_tgt += (sigmas[i+1] - sigmas[i]) * v_delta * noise_mask
            
            if callback is not None:
                callback({'x': x_tgt, 'denoised': x_tgt, 'i': i+skip_steps, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

        return x_tgt
    
    return flowedit_sample


class HYFlowEditSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "skip_steps": ("INT", {"default": 4, "min": 0, "max": 0xffffffffffffffff }),
            "drift_steps": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
        }, "optional": {
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "hunyuanloom"

    def build(self, skip_steps, drift_steps, seed):
        generator = torch.manual_seed(seed)
        sampler = KSAMPLER(get_flowedit_sample(skip_steps, drift_steps, generator))
        return (sampler, )
