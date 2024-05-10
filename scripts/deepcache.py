import torch
import gradio as gr

from modules import scripts
from deepcache_core import DeepCache

deepcache_helper = DeepCache()

class DeepCacheScript(scripts.Script):
    sorting_priority = 0

    def title(self):
        return "DeepCache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row(equal_height=True):
                deepcache_mode = gr.Radio(['None', 'First', 'Hires', 'Both'], label='Enable DeepCache for passes', value='None')
                pow_curve = gr.Slider(label='Cache interval power curve', minimum=0, maximum=10, step=0.1, value=0)
                force_step = gr.Slider(label='Force cache after sigma', minimum=0, maximum=80, step=0.01, value=0.1)

            gr.HTML(value='<div style="height:8px"></div>')

            with gr.Row(equal_height=True):
                cache_interval = gr.Slider(label='Cache interval', minimum=1, maximum=1000, step=1, value=3)
                cache_depth = gr.Slider(label='Cache depth', minimum=0, maximum=12, step=1, value=5)
                
            with gr.Row(equal_height=True):
                start_step = gr.Slider(label='Cache start sigma', minimum=0, maximum=80, step=0.01, value=2)
                end_step = gr.Slider(label='Cache end sigma', minimum=0, maximum=1, step=0.01, value=0)

            gr.HTML(value='<div style="height:8px"></div>')

            with gr.Accordion(open=False, label='Hires pass settings'):
                use_first_pass_settings = gr.Checkbox(label='Use first pass settings', value=False)
                
                gr.HTML(value='<div style="height:8px"></div>')

                with gr.Row(equal_height=True):
                    hr_cache_interval = gr.Slider(label='HR Cache interval', minimum=1, maximum=1000, step=1, value=3)
                    hr_cache_depth = gr.Slider(label='HR Cache depth', minimum=0, maximum=12, step=1, value=5)

        return deepcache_mode, cache_interval, cache_depth, start_step, end_step, force_step, hr_cache_interval, hr_cache_depth, use_first_pass_settings, pow_curve

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        deepcache_mode, cache_interval, cache_depth, start_step, end_step, force_step, hr_cache_interval, hr_cache_depth, use_first_pass_settings, pow_curve = script_args

        if deepcache_mode == 'None':
            return
        elif deepcache_mode == 'First' and p.is_hr_pass:
            return
        elif deepcache_mode == 'Hires' and not p.is_hr_pass:
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
            pow_curve=pow_curve
        ))

        unet = p.sd_model.forge_objects.unet

        if p.is_hr_pass: 
            if not use_first_pass_settings:
                cache_interval = hr_cache_interval
                cache_depth = hr_cache_depth

            # For the hires pass, ignore various settings that we can't
            # apply consistently. Essentially, we use DeepCache for the entire
            # hires process (if enabled)
            use_optimisations = 0                
            start_step = 0
            end_step = 999999999
            force_step = 0
        else:
            start_step = 999999998 if start_step == 0 else 1000 - unet.model.model_sampling.timestep(torch.tensor(start_step)).item()
            end_step = 999999999 if end_step == 0 else 1000 - unet.model.model_sampling.timestep(torch.tensor(end_step)).item()
            force_step = 0 if force_step == 0 else 1000 - unet.model.model_sampling.timestep(torch.tensor(force_step)).item()

        unet = deepcache_helper.apply(unet, cache_interval, cache_depth, start_step, end_step, force_step, pow_curve)[0]
        p.sd_model.forge_objects.unet = unet

        return
