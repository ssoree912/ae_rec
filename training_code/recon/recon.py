from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusionXLImg2ImgPipeline, DiffusionPipeline, DDIMScheduler, AutoPipelineForText2Image, AutoencoderKL, AutoPipelineForImage2Image
from diffusers import StableDiffusion3Pipeline
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def retrieve_latents_compat(encoder_output, generator=None):
    # Compatibility helper for multiple diffusers versions.
    if hasattr(encoder_output, "latent_dist"):
        latent_dist = encoder_output.latent_dist
        if hasattr(latent_dist, "sample"):
            try:
                return latent_dist.sample(generator=generator)
            except TypeError:
                return latent_dist.sample()
        if hasattr(latent_dist, "mode"):
            return latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    if isinstance(encoder_output, (tuple, list)):
        return encoder_output[0]
    return encoder_output


def reconstruct_simple(x, ae, seed, steps=None, tools=None):
    decode_dtype = ae.dtype
    generator = torch.Generator().manual_seed(seed)
    x = x.to(dtype=ae.dtype) * 2.0 - 1.0
    latents = retrieve_latents_compat(ae.encode(x), generator=generator)

    reconstructions = ae.decode(
                        latents.to(decode_dtype), return_dict=False
                    )[0]
    reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)
    return reconstructions


def get_vae(repo_id, return_full=False):
    if 'ldm' in repo_id:
        pipe = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256", cache_dir="weights")
        return pipe.vqvae
    elif 'stable-diffusion-3' in repo_id:
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16,cache_dir='weights')
        return pipe.vae
        #pipe = pipe.to("cuda")




@torch.no_grad()
def ddim_inversion(unet, cond, latent, scheduler, steps=None):
    
    timesteps = reversed(scheduler.timesteps)
    if steps is not None:
        timesteps = timesteps[:steps]
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        for i, t in enumerate(tqdm(timesteps)):
            cond_batch = cond.repeat(latent.shape[0], 1, 1)

            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                    scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else scheduler.final_alpha_cumprod
                )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = unet(latent, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (latent - sigma_prev * eps) / mu_prev
            latent = mu * pred_x0 + sigma * eps
    return latent

@torch.no_grad()
def ddim_sample(x, cond, unet, scheduler, steps=None):
    timesteps = scheduler.timesteps
    if steps is not None:
        timesteps = timesteps[-steps:]
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        for i, t in enumerate(tqdm(timesteps)):
            cond_batch = cond.repeat(x.shape[0], 1, 1)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                        scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else scheduler.final_alpha_cumprod
                    )
            mu = alpha_prod_t ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = unet(x, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (x - sigma * eps) / mu
            x = mu_prev * pred_x0 + sigma_prev * eps

    return x
