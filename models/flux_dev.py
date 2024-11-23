import torch
import math
import warnings

from torch import Tensor

from diffusers import FluxPipeline

class FluxWrapper:
    def __init__(
        self, 
        pipeline: FluxPipeline,
        height: int,
        width: int,
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

        if height % 16 != 0 or width % 16 != 0:
            height = 16 * (height // 16)
            width = 16 * (width // 16)
            warnings.warn(f"Height and width must be divisible by 16. Adjusted to {height}x{width}.")
        self.height, self.width = height, width

        self.dtype = dtype
        self.device = device

        # Initialize cache for prompt embeddings
        self.cached_prompt = None
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None

    def prepare_packed_latents(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
        vae_latents: torch.Tensor | None = None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = self.height // 8
        width = self.width // 8

        # VAE latent channels = 16, VAE latent resolution = resolution // 8
        # shape [num_samples, 16, resolution // 8, resolution // 8]
        shape = (batch_size, 16, height, width)
        print("VAE compressed shape = ", shape)

        # num of tokens = (resolution // 8 * resolution // 8) / 4
        # shape [num_samples, (resolution // 8 * resolution // 8) / 4,  3]
        if vae_latents is not None:
            assert vae_latents.shape[0] == batch_size, "Batch size must match the number of samples."
            assert vae_latents.shape[1] == shape[1], "Number of channels must match the VAE latent channels."
            assert vae_latents.shape[2] == shape[2], "Height must match the VAE latent height."
            assert vae_latents.shape[3] == shape[3], "Width must match the VAE latent width."
        else:
            vae_latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)

        # After packing, shape [num_samples, (resolution // 16 * resolution // 16), 16 * 2 * 2]
        packed_latents = _pack_latents(vae_latents, batch_size, 16, height, width)
        self.image_seq_len = packed_latents.shape[1]
        print("Packed latents shape = ", packed_latents.shape)

        latent_image_ids = _prepare_latent_image_ids(batch_size, height, width, self.device, self.dtype)
        self.latent_image_ids = latent_image_ids
        print("Latent image ids shape = ", latent_image_ids.shape)

        return packed_latents
    
    def prepare_time_grid(
        self,
        num_steps: int,
        shift: bool = True,
    ):
        time_grid = get_timesteps_flux(
            num_steps=num_steps,
            image_seq_len=self.image_seq_len,
            shift=shift,
        )
        time_grid = [1.0 - t for t in time_grid]
        
        return time_grid
    
    def __call__(
        self, 
        X_t: Tensor,
        t: torch.Tensor,
        prompt: str | None = None,
        guidance_scale: float = 3.5,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
    ):
        """
        Computes the flux velocity for a given latent state and time.

        Args:
            X_t: The packed latent variables at time t.
            t: The time in RF ODE, which will be converted to Flux's time (1 - t).
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
        t_vec = 1.0 - t 
        assert isinstance(t_vec, torch.Tensor) and t_vec.ndim == 1 and t.shape[0] == X_t.shape[0], \
            "Time vector must be a 1D tensor with the same length as the batch size."

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

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def get_timesteps_flux(
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


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents


def unpack_and_decode(
    pipeline,
    packed_latents: Tensor, 
    height: int,
    width: int,  
):
    latents = _unpack_latents(packed_latents, height, width, pipeline.vae_scale_factor)
    imgs = decode_imgs(latents, pipeline)[0]
    return imgs