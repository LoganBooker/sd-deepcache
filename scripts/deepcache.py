import torch
import gradio as gr

from modules import scripts
from deepcache_core import DeepCache

deepcache_helper = DeepCache()

class DeepCacheScript(scripts.Script):
    sorting_priority = 0
    use_with_adetailer = False
    use_with_img2img = False
    initialised = False

    def title(self):
        return "DeepCache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row(equal_height=True):
                deepcache_mode = gr.Radio(['None', 'First', 'Hires', 'Both'], label='Enable DeepCache for passes', value='None')
                force_step = gr.Slider(label='Force cache after sigma', minimum=0, maximum=80, step=0.01, value=0.1)

            gr.HTML(value='<div style="height:8px"></div>')

            with gr.Row(equal_height=True):
                cache_interval = gr.Slider(label='Cache interval', minimum=1, maximum=1000, step=1, value=3)
                cache_depth = gr.Slider(label='Cache depth', minimum=0, maximum=12, step=1, value=4)
                
            with gr.Row(equal_height=True):
                start_step = gr.Slider(label='Cache start sigma', minimum=0, maximum=80, step=0.01, value=1)
                end_step = gr.Slider(label='Cache end sigma', minimum=0, maximum=1, step=0.01, value=0)

            gr.HTML(value='<div style="height:8px"></div>')

            with gr.Accordion(open=False, label='Hires pass settings'):
                use_first_pass_settings = gr.Checkbox(label='Use first pass settings', value=False)
                
                gr.HTML(value='<div style="height:8px"></div>')

                with gr.Row(equal_height=True):
                    hr_cache_interval = gr.Slider(label='HR Cache interval', minimum=1, maximum=1000, step=1, value=3)
                    hr_cache_depth = gr.Slider(label='HR Cache depth', minimum=0, maximum=12, step=1, value=4)

                with gr.Row(equal_height=True):
                    hr_start_step = gr.Slider(label='HR Cache start sigma', minimum=0, maximum=80, step=0.01, value=80)
                    hr_end_step = gr.Slider(label='HR Cache end sigma', minimum=0, maximum=1, step=0.01, value=0)

                def toggle_hr_controls(use_first):
                    state = use_first
                    return (
                        gr.update(interactive=not state),
                        gr.update(interactive=not state),
                        gr.update(interactive=not state),
                        gr.update(interactive=not state)
                    )

                use_first_pass_settings.change(
                    fn=toggle_hr_controls,
                    inputs=use_first_pass_settings,
                    outputs=[hr_cache_interval, hr_cache_depth, hr_start_step, hr_end_step]
                )

            with gr.Accordion(open=False, label='Compatibility settings'):
                with gr.Row(equal_height=True):
                    smart_cache = gr.Checkbox(label='Smart sub-caching', value=True, info='Disable if you get errors or strange results with other extensions active.')
                    apply_to_adetailer_pass = gr.Checkbox(label='Enable for ADetailer pass', value=True)
                    apply_to_img2img_pass = gr.Checkbox(label='Enable for img2img', value=False)

        return deepcache_mode, cache_interval, cache_depth, start_step, end_step, force_step, \
            hr_cache_interval, hr_cache_depth, hr_start_step, hr_end_step, \
            use_first_pass_settings, apply_to_adetailer_pass, apply_to_img2img_pass, \
            smart_cache

    def after_extra_networks_activate(self, p, enable, *args, **kwargs):
        is_ad_pass = getattr(p, "_ad_inner", False)
        p_type = type(p).__name__

        if is_ad_pass and not self.use_with_adetailer:
            print("[-] DeepCache: \033[91mCaching disabled\033[0m - Adetailer pass detected.")
            DeepCache.enabled = False
        elif not is_ad_pass and not self.use_with_img2img and "StableDiffusionProcessingImg2Img" in p_type:
            print("[-] DeepCache: \033[91mCaching disabled\033[0m - img2img operation detected.")
            DeepCache.enabled = False
        else:
            DeepCache.enabled = True

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        deepcache_mode, cache_interval, cache_depth, start_step, end_step, force_step, \
            hr_cache_interval, hr_cache_depth, hr_start_step, hr_end_step, \
            use_first_pass_settings, apply_to_adetailer_pass, apply_to_img2img_pass, \
            smart_cache = script_args

        unet = p.sd_model.forge_objects.unet
        
        self.use_with_adetailer = apply_to_adetailer_pass
        self.use_with_img2img = apply_to_img2img_pass

        if (deepcache_mode == 'None') or (deepcache_mode == 'First' and p.is_hr_pass) or (deepcache_mode == 'Hires' and not p.is_hr_pass):
            deepcache_helper.try_remove(unet)
            return

        # Set infotext before we modify the values.
        p.extra_generation_params.update(dict(
            deepcache_mode=deepcache_mode,
            cache_interval=cache_interval,
            cache_depth=cache_depth,
            start_step=start_step,
            end_step=end_step,
            force_step=force_step,
            use_first_pass_settings=use_first_pass_settings,
            hr_cache_interval=hr_cache_interval,
            hr_cache_depth=hr_cache_depth,
            hr_start_step=hr_start_step,
            hr_end_step=hr_end_step,
            smart_cache=smart_cache
        ))

        if p.is_hr_pass and not use_first_pass_settings:
            cache_interval, cache_depth = hr_cache_interval, hr_cache_depth
            start_step, end_step = hr_start_step, hr_end_step

        start_step = 0 if start_step == 0 else 1000 - unet.model.model_sampling.timestep(torch.tensor(start_step)).item()
        end_step = 1000 if end_step == 0 else 1000 - unet.model.model_sampling.timestep(torch.tensor(end_step)).item()
        force_step = 0 if force_step == 0 else 1000 - unet.model.model_sampling.timestep(torch.tensor(force_step)).item()

        unet = deepcache_helper.apply(unet, cache_interval, cache_depth, start_step, end_step, force_step, smart_cache)[0]
        p.sd_model.forge_objects.unet = unet

        return
