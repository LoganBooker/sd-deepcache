# https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86
'''
https://arxiv.org/abs/2312.00858
start_step, end_step: apply this method when the timestep is between start_step and end_step
cache_interval: interval of caching (1 means no caching)
cache_depth: depth of caching
force_step: after this step, ignore the interval and always use the cache
'''

import torch
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, timestep_embedding, apply_control
import types
import sys, os

class DeepCacheStore:
    def __init__(self, depth, interval, start_step, end_step, force_step, smart_cache):
        self.last_timestep = -1
        self.current_timestep = -1

        self.cache_key = None
        self.sub_cache = {}
        self.cache_sub_key = None

        self.step = 0
        self.cache_activated_step = -1

        self.depth = depth
        self.interval = interval
        self.start_step = start_step
        self.end_step = end_step
        self.force_step = force_step
        self.smart_cache = smart_cache

    def __get_cache_interval(self):
        return self.interval

    def __have_cache(self):
        return self.sub_cache.get(self.cache_sub_key, None) is not None

    def __is_cache_active(self):
        return self.start_step <= self.current_timestep <= self.end_step

    def __is_caching_step(self):
        if self.force_step > 0 and self.current_timestep >= self.force_step and self.__have_cache():
            return False

        interval = self.__get_cache_interval()
        return (self.current_timestep != self.cache_activated_step and self.step % interval == 0) or self.sub_cache.get(self.cache_sub_key, None) is None

    def __clear_caches(self):
        for k in list(self.sub_cache.keys()):
            v = self.sub_cache[k]
            if v is not None:
                del self.sub_cache[k]

        self.sub_cache = {}
        self.cache_key = None
        self.cache_sub_key = None
        self.step = 0

    def setup(self, value, step):
        key, sub_key = DeepCache.fast_context_after_common_call(debug=False)
        sub_key = f"{sub_key}-{value.shape[0]}"
        
        # When smart caching is disabled, clear the cache the moment we believe an extension has
        # made a sampling call or batch size has changed.
        if (self.cache_key is None or self.cache_key != key) or \
           (not self.smart_cache and self.cache_sub_key is not None and self.cache_sub_key != sub_key):
            self.__clear_caches()

        self.cache_key = key
        self.cache_sub_key = sub_key
        
        self.last_timestep = self.current_timestep
        self.current_timestep = step

        if self.__is_cache_active() and self.cache_activated_step < 0:
            self.cache_activated_step = step

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
        return (is_active, is_cache_level, is_cache_step)
    
    def get_cache(self):
        return self.sub_cache.get(self.cache_sub_key, None)
    
    def set_cache(self, value):
        cached_value = self.sub_cache.get(self.cache_sub_key, None)
        if cached_value is not None:
            del cached_value
        self.sub_cache[self.cache_sub_key] = value

    def update_step(self):
        if self.__is_cache_active():
            if self.current_timestep != self.last_timestep:
                self.step += 1

class DeepCache:
    ORIGINAL_FORWARD_ATTRIBUTE = "_deepcache_original_forward"
    enabled = True

    @staticmethod
    def fast_context_after_common_call(max_depth=100, debug=False):
        wrapper_funcs = {"_wrapped_call_impl", "_call_impl", "decorate_context", "<lambda>"}
        target_suffix, target_func = os.path.normpath("modules/processing.py"), "process_images"
        sample_suffix, sample_func = os.path.normpath("modules/samplers.py"), "sampling_function"

        last_func = sub_func = None
        found_common = False

        for i in range(max_depth):
            try:
                frame = sys._getframe(i)
            except ValueError:
                break
            
            filename, func = frame.f_code.co_filename, frame.f_code.co_name
            if func in wrapper_funcs:
                continue
            if not found_common:
                if filename.endswith(target_suffix) and func == target_func:
                    found_common = True
                elif filename.endswith(sample_suffix) and func == sample_func:
                    sub_func = last_func
            else:
                return (f"{target_suffix}:{func}", sub_func)
            last_func = func
        return None

    def try_remove(self, model):
        # Remove the patched method, and use the original stored forward method.
        # If a new model is loaded, this won't be present, so we automatically handle that case.
        if hasattr(model.model.diffusion_model.forward, DeepCache.ORIGINAL_FORWARD_ATTRIBUTE):
            model.model.diffusion_model.forward = getattr(model.model.diffusion_model.forward, DeepCache.ORIGINAL_FORWARD_ATTRIBUTE)

    def apply(self, model, arg_cache_interval, cache_depth, start_step, end_step, force_step, smart_cache):

        # Capture the original forward method for the decorator. Once the method has been patched, this will
        # be the patched method, but that's OK as we never repatch and won't overwrite the method.
        original_forward = model.model.diffusion_model.forward

        def deepcache_patched_decorator(func):
            setattr(func, DeepCache.ORIGINAL_FORWARD_ATTRIBUTE, original_forward)
            return func

        @deepcache_patched_decorator
        @torch.inference_mode()
        def forward_deepcache_patched(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            
            # Use original forward pass when disabled.
            if not DeepCache.enabled:
                return original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)

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
                is_active, is_level, cache_step = cache_store.cache_unet_block(id, block_count)

                if is_level:
                    if cache_step:
                        cache_store.set_cache(h)
                    elif is_active:
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
            if method_name != 'UNetModel.forward':
                print(f"[-] DeepCache: \033[91mCaching disabled\033[0m - UNet forward pass already patched by another extension ({method_name}).")
                return
            
            new_model.model.diffusion_model.forward = types.MethodType(forward_deepcache_patched, new_model.model.diffusion_model)

        # Store this run's cache on the model itself.
        new_model.model.diffusion_model._deepcache_store = DeepCacheStore(cache_depth, arg_cache_interval, start_step, end_step, force_step, smart_cache)

        return (new_model, )