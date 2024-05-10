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

class DeepCache:
    def apply(self, model, arg_cache_interval, cache_depth, start_step, end_step, force_step, pow_curve):
        new_model = model.clone()
        m = new_model.model
        unet = m.diffusion_model

        current_t = -1
        current_step = -1
        cache_h = {}

        original_cache_interval = arg_cache_interval

        def apply_model(model_function, kwargs):

            nonlocal current_t, current_step, cache_h, unet, m, original_cache_interval
            
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

            # unet次回実行はtimestepが上がってると仮定・・Refiner等でエラーが起きるかも
            active_t = t[0].item()

            if active_t > current_t:
                current_step = -1

            current_t = active_t

            cache_interval = original_cache_interval

            real_step = 1000 - current_t
            apply = False

            if end_step > start_step:
                apply = real_step >= start_step and real_step <= end_step

                # Apply a power curve to the interval. We start from 1 and slower increase the interval
                # to the user-provided cache_interval. Cosine/smoothstep might be better here, so we spend
                # at least some time at the higher cache interval.
                
                # This is not used by default, as it's a hard value to tune, though 4 seems to work well.
                if pow_curve > 0 and apply and real_step >= start_step:
                    step_mod = real_step - start_step
                    step_mod_end = 1000 - start_step
                    step_frac = (step_mod / step_mod_end) ** pow_curve

                    cache_interval = 1 + round((original_cache_interval - 1) * step_frac)
            else:
                # If the start step is greater than end, we treat it as the interval in which caching
                # should be disabled, rather than enabled. Can't see a situation where this would be
                # better than the default behaviour, but we provide the option anyway.
                apply = real_step >= start_step or real_step <= end_step

            if apply:
                current_step += 1
            else:
                current_step = -1

            # Allow caching of different batch sizes. Batch sizes are usually changed by other extensions,
            # or when guidance is disabled.
            cache_shape_key = x.shape[0]
            is_caching_step = current_step % cache_interval == 0
            have_cache_value = cache_shape_key in cache_h

            # Don't bother caching on the last few steps.
            if force_step > 0 and real_step >= force_step and have_cache_value:
                apply = True
                is_caching_step = False
            elif apply and not is_caching_step and not have_cache_value:
                print(f'\n[DEEPCACHE] Shape {cache_shape_key} not found in cache. Forcing cache step.\n')
                current_step = 0
                is_caching_step = True

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
            for id, module in enumerate(unet.input_blocks):
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
                
                if id == cache_depth and apply:
                    if not is_caching_step:
                        break # cache位置以降はスキップ

            if is_caching_step or not apply:
                transformer_options["block"] = ("middle", 0)

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'before', transformer_options)

                h = forward_timestep_embed(unet.middle_block, h, emb, context, transformer_options)
                h = apply_control(h, control, 'middle')

                for block_modifier in block_modifiers:
                    h = block_modifier(h, 'after', transformer_options)

            for id, module in enumerate(unet.output_blocks):
                if id < len(unet.output_blocks) - cache_depth - 1 and apply:
                    if not is_caching_step:
                        continue # cache位置以前はスキップ
                
                if id == len(unet.output_blocks) - cache_depth - 1 and apply:
                    if is_caching_step:
                        cache_h[cache_shape_key] = h # cache
                    else:
                        h = cache_h[cache_shape_key] # load cache
                
                transformer_options["block"] = ("output", id)
                hsp = hs.pop()
                hsp = apply_control(hsp, control, 'output')

                if "output_block_patch" in transformer_patches:
                    patch = transformer_patches["output_block_patch"]
                    for p in patch:
                        h, hsp = p(h, hsp, transformer_options)

                h = th.cat([h, hsp], dim=1)
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

            return m.model_sampling.calculate_denoised(sigma, model_output, xa)

        new_model.set_model_unet_function_wrapper(apply_model)

        return (new_model, )