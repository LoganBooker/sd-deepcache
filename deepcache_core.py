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
import types

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
    ORIGINAL_FORWARD_ATTRIBUTE = "_deepcache_original_forward"

    def try_remove(self, model):
        # Remove the patched method, and use the original stored forward method.
        # If a new model is loaded, this won't be present, so we automatically handle that case.
        if hasattr(model.model.diffusion_model.forward, DeepCache.ORIGINAL_FORWARD_ATTRIBUTE):
            model.model.diffusion_model.forward = getattr(model.model.diffusion_model.forward, DeepCache.ORIGINAL_FORWARD_ATTRIBUTE)

    def apply(self, model, arg_cache_interval, cache_depth, start_step, end_step, force_step, pow_curve):

        # Capture the original forward method for the decorator. Once the method has been patched, this will
        # be the patched method, but that's OK as we never repatch and won't overwrite the method.
        original_forward = model.model.diffusion_model.forward

        def deepcache_patched_decorator(func):
            setattr(func, DeepCache.ORIGINAL_FORWARD_ATTRIBUTE, original_forward)
            return func

        @deepcache_patched_decorator
        def forward_deepcache_patched(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """

            # Fetch the store from the model.
            cache_store = self._deepcache_store

            transformer_options["original_shape"] = list(x.shape)
            transformer_options["transformer_index"] = 0
            transformer_patches = transformer_options.get("patches", {})
            block_modifiers = transformer_options.get("block_modifiers", [])

            # This code must be kept in sync.
            # https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/29be1da7cf2b5dccfc70fbdd33eb35c56a31ffb7/ldm_patched/ldm/modules/diffusionmodules/openaimodel.py#L831
            assert (y is not None) == (
                self.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
            emb = self.time_embed(t_emb)

            if self.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            h = x

            # DeepCache
            cache_store.setup(h, 1000 - timesteps[0].item())

            for id, module in enumerate(self.input_blocks):
                # DeepCache
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

            # DeepCache
            if not cache_store.skip_unet_block("middle"):
                transformer_options["block"] = ("middle", 0)

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'before', transformer_options)

                h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)
                h = apply_control(h, control, 'middle')

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'after', transformer_options)

            block_count = len(self.output_blocks)
            for id, module in enumerate(self.output_blocks):
                # DeepCache
                if cache_store.skip_unet_block("output", id, block_count):
                    continue

                # DeepCache - START
                is_active, cache_step = cache_store.cache_unet_block(id, block_count)

                if is_active:
                    if cache_step:
                        cache_store.set_cache(h)
                    else:
                        h = cache_store.get_cache()
                # DeepCache - END

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

            # DeepCache
            cache_store.update_step()

            transformer_options["block"] = ("last", 0)

            for block_modifier in block_modifiers:
                h = block_modifier(h, 'before', transformer_options)

            if self.predict_codebook_ids:
                h = self.id_predictor(h)
            else:
                h = self.out(h)
            
            for block_modifier in block_modifiers:
                h = block_modifier(h, 'after', transformer_options)

            return h.type(x.dtype)

        new_model = model.clone()

        # Patch the forward method of the model. If we haven't patched it already, we use a decorator to
        # store the original forward pass on our patched method.
        if not hasattr(new_model.model.diffusion_model.forward, DeepCache.ORIGINAL_FORWARD_ATTRIBUTE):
            method_name = new_model.model.diffusion_model.forward.__qualname__
            if (method_name != 'UNetModel.forward'):
                print(f"\n[DeepCache] Warning: Unet forward pass appears to be patched by another extension ({method_name})! DeepCache disabled.")
                return
            
            new_model.model.diffusion_model.forward = types.MethodType(forward_deepcache_patched, new_model.model.diffusion_model)

        # Store this run's cache on the model itself.
        new_model.model.diffusion_model._deepcache_store = DeepCacheStore(cache_depth, arg_cache_interval, start_step, end_step, force_step, pow_curve)

        return (new_model, )