# https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86
'''
https://arxiv.org/abs/2312.00858
start_step, end_step: apply this method when the timestep is between start_step and end_step
cache_interval: interval of caching (1 means no caching)
cache_depth: depth of caching
pow_curve: increase the cache interval over time, clamped and scale to the start and end steps (0 means disabled)
force_step: after this step, ignore the interval and always use the cache
'''

import torch
import math
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, timestep_embedding, th, apply_control

class DeepCacheStore:
    def __init__(self, depth, interval, start_step, end_step, force_step, pow_curve):
        self.step_shape = None
        self.current_timestep = -1
        self.total_steps = 0

        # TODO: Cache age... entries older than interval should probably not be reused.

        # Cache is a keyed dictionary based on batch size. Also store step and timestep
        # information per key. Different batch sizes should not affect each other. For example,
        # a PAG step with a single batch, versus a regular batch.
        self.cache = {}

        self.depth = depth
        self.interval = interval
        self.start_step = start_step
        self.end_step = end_step
        self.force_step = force_step
        self.pow_curve = pow_curve

    def __get_cache_interval(self):
        if self.pow_curve <= 0:
            return self.interval
        
        step_mod = self.current_timestep - self.start_step
        step_mod_end = 1000 - self.start_step
        step_frac = (step_mod / step_mod_end) ** self.pow_curve

        return 1 + round((self.interval - 1) * step_frac)

    def __have_cache_for_key(self):
        return self.step_shape in self.cache

    def __is_cache_active(self):
        return self.start_step <= self.current_timestep <= self.end_step

    def __is_caching_step(self):
        if self.force_step > 0 and self.current_timestep >= self.force_step and self.__have_cache_for_key():
            return False

        cache = self.cache.get(self.step_shape)
        interval = self.__get_cache_interval()

        return cache is None or cache["step"] % interval == 0

    def setup(self, value, step):
        self.step_shape = value.shape[0]
        self.current_timestep = step

    def skip_unet_block(self, block_id, i=0, block_count=0):
        if not self.__is_cache_active():
            return False
        
        if self.__is_caching_step():
            return False
        
        match block_id:
            case "input":
                return i > self.depth
            case "middle":
                return True
            case "output":
                return i < block_count - self.depth - 1

        print(f'Invalid block id {block_id}!')
        return False

    def cache_unet_block(self, i, block_count):
        is_cache_level = i == block_count - self.depth - 1
        is_active = self.__is_cache_active()
        is_cache_step = self.__is_caching_step()

        # active, cache_step
        return (is_active and is_cache_level, is_cache_step)
    
    def get_cache(self):
        cache = self.cache.get(self.step_shape)
        return None if cache is None else cache["value"]
    
    def set_cache(self, value):
        self.cache[self.step_shape] = { "value": value, "step": 0, "timestep": -1 }

    def update_step(self):
        if self.__is_cache_active():
            if self.__have_cache_for_key():
                key = self.step_shape
                timestep = self.cache[key]["timestep"]

                # Don't increment step if the timestep hasn't changed.
                if self.current_timestep > timestep:
                    self.cache[key]["step"] += 1
                    self.cache[key]["timestep"] = self.current_timestep
        else:
            for key in self.cache:
                self.cache[key]["step"] = 0

class DeepCache:
    def apply(self, model, arg_cache_interval, cache_depth, start_step, end_step, force_step, pow_curve):
        new_model = model.clone()
        m = new_model.model
        unet = m.diffusion_model

        cache_store = DeepCacheStore(cache_depth, arg_cache_interval, start_step, end_step, force_step, pow_curve)

        def apply_model(model_function, kwargs):

            nonlocal unet, m, cache_store
            
            xa = kwargs["input"]
            t = kwargs["timestep"]
            c_concat = kwargs["c"].get("c_concat", None)
            c_crossattn = kwargs["c"].get("c_crossattn", None)
            y = kwargs["c"].get("y", None)
            control = kwargs["c"].get("control", None)
            transformer_options = kwargs["c"].get("transformer_options", None)

            # This code must be kept in sync.
            # https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/29be1da7cf2b5dccfc70fbdd33eb35c56a31ffb7/ldm_patched/modules/model_base.py#L67
            sigma = t
            xc = m.model_sampling.calculate_input(sigma, xa)
            if c_concat is not None:
                xc = torch.cat([xc] + [c_concat], dim=1)

            context = c_crossattn
            dtype = m.get_dtype()

            if m.manual_cast_dtype is not None:
                dtype = m.manual_cast_dtype

            xc = xc.to(dtype)
            t = m.model_sampling.timestep(t).float()
            context = context.to(dtype)
            extra_conds = {}
            for o in kwargs:
                extra = kwargs[o]
                if hasattr(extra, "dtype"):
                    if extra.dtype != torch.int and extra.dtype != torch.long:
                        extra = extra.to(dtype)
                extra_conds[o] = extra

            x = xc
            timesteps = t
            y = None if y is None else y.to(dtype)

            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """

            transformer_options["original_shape"] = list(x.shape)
            transformer_options["transformer_index"] = 0
            transformer_patches = transformer_options.get("patches", {})
            block_modifiers = transformer_options.get("block_modifiers", [])

            # This code must be kept in sync.
            # https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/29be1da7cf2b5dccfc70fbdd33eb35c56a31ffb7/ldm_patched/ldm/modules/diffusionmodules/openaimodel.py#L831
            assert (y is not None) == (
                unet.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(x.dtype)
            emb = unet.time_embed(t_emb)

            if unet.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + unet.label_emb(y)

            h = x
            cache_store.setup(h, 1000 - t[0].item())

            for id, module in enumerate(unet.input_blocks):
                if cache_store.skip_unet_block("input", id):
                    break

                transformer_options["block"] = ("input", id)

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'before', transformer_options)

                h = forward_timestep_embed(module, h, emb, context, transformer_options)
                h = apply_control(h, control, 'input')

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'after', transformer_options)

                if "input_block_patch" in transformer_patches:
                    patch = transformer_patches["input_block_patch"]
                    for p in patch:
                        h = p(h, transformer_options)

                hs.append(h)
                if "input_block_patch_after_skip" in transformer_patches:
                    patch = transformer_patches["input_block_patch_after_skip"]
                    for p in patch:
                        h = p(h, transformer_options)

            if not cache_store.skip_unet_block("middle"):
                transformer_options["block"] = ("middle", 0)

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'before', transformer_options)

                h = forward_timestep_embed(unet.middle_block, h, emb, context, transformer_options)
                h = apply_control(h, control, 'middle')

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'after', transformer_options)

            block_count = len(unet.output_blocks)
            for id, module in enumerate(unet.output_blocks):
                if cache_store.skip_unet_block("output", id, block_count):
                    continue

                is_active, cache_step = cache_store.cache_unet_block(id, block_count)

                if is_active:
                    if cache_step:
                        cache_store.set_cache(h)
                    else:
                        h = cache_store.get_cache()

                transformer_options["block"] = ("output", id)
                hsp = hs.pop()
                hsp = apply_control(hsp, control, 'output')

                if "output_block_patch" in transformer_patches:
                    patch = transformer_patches["output_block_patch"]
                    for p in patch:
                        h, hsp = p(h, hsp, transformer_options)

                h = torch.cat([h, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'before', transformer_options)

                h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'after', transformer_options)

            transformer_options["block"] = ("last", 0)

            for block_modifier in block_modifiers:
                h = block_modifier(h, 'before', transformer_options)

            h = h.type(x.dtype)
            if unet.predict_codebook_ids:
                model_output =  unet.id_predictor(h)
            else:
                model_output =  unet.out(h)
            
            for block_modifier in block_modifiers:
                h = block_modifier(h, 'after', transformer_options)

            cache_store.update_step()

            return m.model_sampling.calculate_denoised(sigma, model_output, xa)

        new_model.set_model_unet_function_wrapper(apply_model)

        return (new_model, )