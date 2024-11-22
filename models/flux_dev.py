import torch
import math
import warnings

from torch import Tensor

from diffusers import FluxPipeline

class FluxWrapper:
    def __init__(
        self, 
        pipeline: FluxPipeline,
        latent_image_ids: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Initializes the FluxWrapper with the necessary components.

        Args:
            pipeline: The diffusion pipeline object.
            latents_shape: The shape of the latent variables (packed_latents).
            dtype: The data type for computations.
            device: The device on which computations are performed.
        """
        self.pipeline = pipeline
        self.dtype = dtype
        self.device = device
        self.latent_image_ids = latent_image_ids

        # Initialize cache for prompt embeddings
        self.cached_prompt = None
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None

    def get_flux_velocity(
        self, 
        X_t: Tensor,
        t: float,
        prompt: str | None = None,
        guidance_scale: float = 3.5,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
    ):
        """
        Computes the flux velocity for a given latent state and time.

        Args:
            X_t: The packed latent variables at time t.
            t (float): The time in your ODE, which will be converted to Flux's time (1 - t).
            prompt (str, optional): The text prompt. Defaults to empty string.
            guidance_scale (float, optional): The guidance scale. Defaults to 0.0.
            prompt_embeds (Tensor, optional): Precomputed prompt embeddings. If provided,
                'prompt' is ignored for embeddings.
            pooled_prompt_embeds (Tensor, optional): Precomputed pooled prompt embeddings.

        Returns:
            Tensor: The negative flux velocity.
        """
        if X_t.device != self.device:
            X_t = X_t.to(device=self.device)
            warnings.warn(f"X_t was moved to the device {self.device} of the FluxWrapper.")

        if X_t.dtype != self.dtype:
            X_t = X_t.to(dtype=self.dtype)
            warnings.warn(f"X_t was casted to the dtype {self.dtype} of the FluxWrapper.")

        # Convert ODE time t to Flux time 1 - t
        t_flux = 1.0 - t
        t_vec = torch.full(
            (X_t.shape[0],),
            t_flux,
            dtype=X_t.dtype,
            device=X_t.device
        )

        # Prepare guidance vector
        guidance_vec = torch.full(
            (X_t.shape[0],),
            guidance_scale,
            device=self.device,
            dtype=self.dtype
        )

        with torch.inference_mode():
            if prompt_embeds is not None and pooled_prompt_embeds is not None:
                self.prompt_embeds = prompt_embeds
                self.pooled_prompt_embeds = pooled_prompt_embeds
                _, _, self.text_ids = self.pipeline.encode_prompt(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )
            else:
                if prompt is None:
                    assert self.cached_prompt is not None, "Prompt must be provided if not cached."
                elif prompt != self.cached_prompt: # Encode prompt if it has changed
                    self.cached_prompt = prompt
                    self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = self.pipeline.encode_prompt(
                        prompt=prompt,
                        prompt_2=prompt,
                    )
                    print(f"Prompt {prompt} encoded.")

        flux_velocity = self.pipeline.transformer(
            hidden_states=X_t,
            timestep=t_vec,
            guidance=guidance_vec,
            pooled_projections=self.pooled_prompt_embeds,
            encoder_hidden_states=self.prompt_embeds,
            txt_ids=self.text_ids,
            img_ids=self.latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=self.pipeline,
        )[0]

        flux_velocity = -flux_velocity

        return flux_velocity
    

@torch.inference_mode()
def decode_imgs(latents, pipeline):
    imgs = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    imgs = pipeline.vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(imgs, output_type="pil")
    return imgs


@torch.inference_mode()
def encode_imgs(imgs, pipeline, dtype):
    latents = pipeline.vae.encode(imgs).latent_dist.sample()
    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    latents = latents.to(dtype=dtype)
    return latents


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)


def get_time_grid(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list:
    def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b
    def time_shift(mu: float, sigma: float, t: Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float32)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def prepare_packed_latents(
    batch_size,
    height,
    width,
    dtype,
    device,
    generator=None,
    latents=None,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (8 * 2))
    width = 2 * (int(width) // (8 * 2))

    # VAE latent channels = 16, VAE latent resolution = resolution // 8
    # shape [num_samples, 16, resolution // 8, resolution // 8]
    shape = (batch_size, 16, height, width)

    if latents is not None:
        # num of tokens = (resolution // 8 * resolution // 8) / 4
        # shape [num_samples, (resolution // 8 * resolution // 8) / 4,  3]
        latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        return latents.to(device=device, dtype=dtype), latent_image_ids

    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # After packing, shape [num_samples, (resolution // 16 * resolution // 16), 16 * 2 * 2]
    latents = _pack_latents(latents, batch_size, 16, height, width)

    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

    return latents, latent_image_ids