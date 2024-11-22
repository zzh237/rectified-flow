import torch
from diffusers import FluxPipeline

class FluxWrapper:
    def __init__(
        self, 
        pipeline: FluxPipeline,
        latents_shape: tuple[int],
        DTYPE: torch.dtype,
        device: torch.device,
    ):
        """
        Initializes the FluxWrapper with the necessary components.

        Args:
            pipeline: The diffusion pipeline object.
            latents_shape: The shape of the latent variables (packed_latents).
            DTYPE: The data type for computations.
            device: The device on which computations are performed.
        """
        self.pipeline = pipeline
        self.DTYPE = DTYPE
        self.device = device
        self.latents_shape = latents_shape

        # Prepare latent image IDs (fixed upon initialization)
        self.latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            latents_shape[0],
            latents_shape[2],
            latents_shape[3],
            device,
            DTYPE,
        )

        # Initialize cache for prompt embeddings
        self.cached_prompt = None
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None

    def get_flux_velocity(self, X_t, t, prompt="", guidance_scale=0.0,
                          prompt_embeds=None, pooled_prompt_embeds=None):
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
            dtype=self.DTYPE
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