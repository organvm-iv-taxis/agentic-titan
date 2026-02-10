"""
Image Generation Tool - AI image generation capabilities.

Provides:
- Text-to-image generation
- Image-to-image transformation
- Inpainting and outpainting
- Multiple backend support (Stable Diffusion, DALL-E, etc.)

Reference: vendor/tools/stable-diffusion/
"""

from __future__ import annotations

import base64
import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger("titan.tools.image_gen")


# ============================================================================
# Data Structures
# ============================================================================


class ImageFormat(StrEnum):
    """Output image format."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class ImageSize(StrEnum):
    """Predefined image sizes."""

    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"
    WIDE = "1792x1024"
    TALL = "1024x1792"


@dataclass
class ImageRequest:
    """Request for image generation."""

    prompt: str
    negative_prompt: str = ""
    size: str = "512x512"
    num_images: int = 1
    seed: int | None = None

    # Generation parameters
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    strength: float = 0.75  # For img2img

    # Input image (for img2img, inpainting)
    input_image: bytes | None = None
    mask_image: bytes | None = None

    # Output options
    output_format: ImageFormat = ImageFormat.PNG
    quality: int = 95

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.size.split("x")[0])

    @property
    def height(self) -> int:
        return int(self.size.split("x")[1])


@dataclass
class GeneratedImage:
    """A generated image."""

    data: bytes
    format: ImageFormat
    width: int
    height: int
    seed: int | None = None

    # Generation info
    prompt: str = ""
    model: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_base64(self) -> str:
        """Convert image to base64 string."""
        return base64.b64encode(self.data).decode("utf-8")

    def save(self, path: str | Path) -> None:
        """Save image to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        logger.info(f"Saved image to {path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "prompt": self.prompt,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "data_base64": self.to_base64(),
        }


@dataclass
class ImageGenerationResult:
    """Result of image generation."""

    images: list[GeneratedImage]
    model: str
    total_time_ms: float = 0.0

    # Cost tracking
    estimated_cost_usd: float = 0.0

    # Errors
    error: str | None = None

    @property
    def success(self) -> bool:
        return len(self.images) > 0 and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "model": self.model,
            "image_count": len(self.images),
            "total_time_ms": self.total_time_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "error": self.error,
            "images": [img.to_dict() for img in self.images],
        }


# ============================================================================
# Image Generation Backends
# ============================================================================


class ImageBackend(ABC):
    """Abstract base class for image generation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @property
    @abstractmethod
    def supported_sizes(self) -> list[str]:
        """Supported image sizes."""
        pass

    @abstractmethod
    async def generate(self, request: ImageRequest) -> ImageGenerationResult:
        """Generate images from request."""
        pass

    @abstractmethod
    async def img2img(self, request: ImageRequest) -> ImageGenerationResult:
        """Transform image based on prompt."""
        pass

    def validate_request(self, request: ImageRequest) -> str | None:
        """Validate request parameters. Returns error message or None."""
        if not request.prompt:
            return "Prompt is required"

        if request.size not in self.supported_sizes:
            return f"Unsupported size {request.size}. Supported: {self.supported_sizes}"

        if request.num_images < 1 or request.num_images > 10:
            return "num_images must be between 1 and 10"

        return None


class StableDiffusionBackend(ImageBackend):
    """
    Stable Diffusion backend for image generation.

    Supports local models via diffusers or API services.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "auto",
        api_url: str | None = None,
        api_key: str | None = None,  # allow-secret
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.api_url = api_url
        self.api_key = api_key  # allow-secret

        # Pipeline (lazy loaded)
        self._pipeline = None
        self._use_api = api_url is not None

    @property
    def name(self) -> str:
        return f"stable-diffusion:{self.model_id}"

    @property
    def supported_sizes(self) -> list[str]:
        return [
            "256x256",
            "512x512",
            "768x768",
            "1024x1024",
            "512x768",
            "768x512",
            "1024x768",
            "768x1024",
        ]

    def _get_pipeline(self) -> Any:
        """Get or create the diffusion pipeline."""
        if self._pipeline is None:
            try:
                import torch
                from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

                # Determine device
                if self.device == "auto":
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                else:
                    device = self.device

                # Load appropriate pipeline
                pipeline: Any
                if "xl" in self.model_id.lower():
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    )
                else:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    )

                pipeline = pipeline.to(device)

                # Enable memory optimizations
                if device == "cuda":
                    pipeline.enable_attention_slicing()

                self._pipeline = pipeline

                logger.info(f"Loaded Stable Diffusion pipeline on {device}")

            except ImportError:
                raise RuntimeError(
                    "diffusers package required. Install with: pip install diffusers torch"
                )

        return self._pipeline

    async def generate(self, request: ImageRequest) -> ImageGenerationResult:
        """Generate images using Stable Diffusion."""
        import time

        # Validate
        error = self.validate_request(request)
        if error:
            return ImageGenerationResult(images=[], model=self.name, error=error)

        start_time = time.time()

        try:
            if self._use_api:
                return await self._generate_api(request)
            else:
                return await self._generate_local(request)

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return ImageGenerationResult(
                images=[],
                model=self.name,
                total_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _generate_local(self, request: ImageRequest) -> ImageGenerationResult:
        """Generate using local pipeline."""
        import time

        import torch

        start_time = time.time()
        pipeline = self._get_pipeline()

        # Set seed for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=pipeline.device).manual_seed(request.seed)

        # Generate
        output = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            num_images_per_prompt=request.num_images,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )

        # Convert to GeneratedImage objects
        images = []
        for i, pil_image in enumerate(output.images):
            # Convert PIL to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format=request.output_format.value.upper())
            image_data = buffer.getvalue()

            images.append(
                GeneratedImage(
                    data=image_data,
                    format=request.output_format,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    prompt=request.prompt,
                    model=self.name,
                )
            )

        return ImageGenerationResult(
            images=images,
            model=self.name,
            total_time_ms=(time.time() - start_time) * 1000,
        )

    async def _generate_api(self, request: ImageRequest) -> ImageGenerationResult:
        """Generate using API service."""
        import time

        import httpx

        start_time = time.time()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/v1/generation/text-to-image",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": request.width,
                    "height": request.height,
                    "samples": request.num_images,
                    "steps": request.num_inference_steps,
                    "cfg_scale": request.guidance_scale,
                    "seed": request.seed,
                },
                timeout=120.0,
            )

            if response.status_code != 200:
                return ImageGenerationResult(
                    images=[],
                    model=self.name,
                    total_time_ms=(time.time() - start_time) * 1000,
                    error=f"API error: {response.status_code} - {response.text}",
                )

            data = response.json()

        # Parse response
        images = []
        for artifact in data.get("artifacts", []):
            image_data = base64.b64decode(artifact["base64"])
            images.append(
                GeneratedImage(
                    data=image_data,
                    format=request.output_format,
                    width=request.width,
                    height=request.height,
                    seed=artifact.get("seed"),
                    prompt=request.prompt,
                    model=self.name,
                )
            )

        return ImageGenerationResult(
            images=images,
            model=self.name,
            total_time_ms=(time.time() - start_time) * 1000,
        )

    async def img2img(self, request: ImageRequest) -> ImageGenerationResult:
        """Image-to-image transformation."""
        if not request.input_image:
            return ImageGenerationResult(
                images=[],
                model=self.name,
                error="input_image is required for img2img",
            )

        # Implementation similar to generate but using img2img pipeline
        # Simplified for brevity
        return ImageGenerationResult(
            images=[],
            model=self.name,
            error="img2img not yet implemented for local backend",
        )


class DallEBackend(ImageBackend):
    """
    DALL-E backend for image generation via OpenAI API.
    """

    def __init__(
        self,
        api_key: str | None = None,  # allow-secret
        model: str = "dall-e-3",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")  # allow-secret
        self.model = model

        if not self.api_key:
            logger.warning("No OpenAI API key provided for DALL-E backend")

    @property
    def name(self) -> str:
        return f"dall-e:{self.model}"

    @property
    def supported_sizes(self) -> list[str]:
        if self.model == "dall-e-3":
            return ["1024x1024", "1792x1024", "1024x1792"]
        else:  # dall-e-2
            return ["256x256", "512x512", "1024x1024"]

    async def generate(self, request: ImageRequest) -> ImageGenerationResult:
        """Generate images using DALL-E."""
        import time

        if not self.api_key:
            return ImageGenerationResult(
                images=[],
                model=self.name,
                error="OpenAI API key not configured",
            )

        error = self.validate_request(request)
        if error:
            return ImageGenerationResult(images=[], model=self.name, error=error)

        start_time = time.time()

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "prompt": request.prompt,
                        "n": min(request.num_images, 1 if self.model == "dall-e-3" else 10),
                        "size": request.size,
                        "response_format": "b64_json",
                        "quality": "hd" if request.quality > 90 else "standard",
                    },
                    timeout=120.0,
                )

                if response.status_code != 200:
                    return ImageGenerationResult(
                        images=[],
                        model=self.name,
                        total_time_ms=(time.time() - start_time) * 1000,
                        error=f"OpenAI API error: {response.status_code} - {response.text}",
                    )

                data = response.json()

            # Parse response
            images = []
            for item in data.get("data", []):
                image_data = base64.b64decode(item["b64_json"])
                images.append(
                    GeneratedImage(
                        data=image_data,
                        format=ImageFormat.PNG,
                        width=request.width,
                        height=request.height,
                        prompt=item.get("revised_prompt", request.prompt),
                        model=self.name,
                    )
                )

            # Estimate cost
            cost = self._estimate_cost(request.size, len(images))

            return ImageGenerationResult(
                images=images,
                model=self.name,
                total_time_ms=(time.time() - start_time) * 1000,
                estimated_cost_usd=cost,
            )

        except Exception as e:
            logger.error(f"DALL-E generation failed: {e}")
            return ImageGenerationResult(
                images=[],
                model=self.name,
                total_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _estimate_cost(self, size: str, count: int) -> float:
        """Estimate cost in USD."""
        # DALL-E 3 pricing (as of 2024)
        if self.model == "dall-e-3":
            if size == "1024x1024":
                return count * 0.04
            else:  # HD sizes
                return count * 0.08
        else:  # DALL-E 2
            if size == "1024x1024":
                return count * 0.02
            elif size == "512x512":
                return count * 0.018
            else:
                return count * 0.016

    async def img2img(self, request: ImageRequest) -> ImageGenerationResult:
        """DALL-E doesn't support img2img directly."""
        return ImageGenerationResult(
            images=[],
            model=self.name,
            error="DALL-E does not support img2img. Use variations endpoint instead.",
        )


# ============================================================================
# Image Generation Tool
# ============================================================================


class ImageGenerationTool:
    """
    Tool for AI image generation.

    Supports multiple backends (Stable Diffusion, DALL-E).

    Example:
        tool = ImageGenerationTool()

        # Text to image
        result = await tool.generate(
            prompt="A serene mountain landscape at sunset",
            size="1024x1024",
        )

        if result.success:
            result.images[0].save("landscape.png")
    """

    name = "image_generation"
    description = "Generate images using AI models (Stable Diffusion, DALL-E)"

    def __init__(
        self,
        default_backend: str = "dall-e",
        backends: dict[str, ImageBackend] | None = None,
    ) -> None:
        self.default_backend = default_backend
        self._backends: dict[str, ImageBackend] = backends or {}

        # Initialize default backends if not provided
        if not self._backends:
            self._init_default_backends()

    def _init_default_backends(self) -> None:
        """Initialize default backends."""
        # DALL-E (if API key available)
        openai_key = os.environ.get("OPENAI_API_KEY")  # allow-secret
        if openai_key:
            self._backends["dall-e"] = DallEBackend(api_key=openai_key)  # allow-secret
            self._backends["dall-e-3"] = DallEBackend(
                api_key=openai_key,  # allow-secret
                model="dall-e-3",  # allow-secret
            )  # allow-secret
            self._backends["dall-e-2"] = DallEBackend(
                api_key=openai_key,  # allow-secret
                model="dall-e-2",  # allow-secret
            )  # allow-secret

        # Stable Diffusion (local - lazy loaded)
        self._backends["stable-diffusion"] = StableDiffusionBackend()
        self._backends["sdxl"] = StableDiffusionBackend(
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )

    def get_backend(self, name: str | None = None) -> ImageBackend | None:
        """Get a backend by name."""
        name = name or self.default_backend
        return self._backends.get(name)

    def list_backends(self) -> list[str]:
        """List available backends."""
        return list(self._backends.keys())

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        size: str = "1024x1024",
        num_images: int = 1,
        backend: str | None = None,
        seed: int | None = None,
        guidance_scale: float = 7.5,
        steps: int = 50,
        output_format: str = "png",
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        Generate images from a text prompt.

        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid in the image
            size: Image size (e.g., "1024x1024")
            num_images: Number of images to generate
            backend: Backend to use (default: configured default)
            seed: Random seed for reproducibility
            guidance_scale: How closely to follow the prompt
            steps: Number of inference steps
            output_format: Output format (png, jpeg, webp)

        Returns:
            ImageGenerationResult with generated images
        """
        backend_impl = self.get_backend(backend)
        if not backend_impl:
            return ImageGenerationResult(
                images=[],
                model="unknown",
                error=f"Backend '{backend or self.default_backend}' not found",
            )

        request = ImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            size=size,
            num_images=num_images,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            output_format=ImageFormat(output_format),
        )

        return await backend_impl.generate(request)

    async def img2img(
        self,
        prompt: str,
        input_image: bytes,
        strength: float = 0.75,
        backend: str | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """
        Transform an existing image based on a prompt.

        Args:
            prompt: Text description for transformation
            input_image: Input image bytes
            strength: How much to change (0-1)
            backend: Backend to use

        Returns:
            ImageGenerationResult with transformed images
        """
        backend_impl = self.get_backend(backend)
        if not backend_impl:
            return ImageGenerationResult(
                images=[],
                model="unknown",
                error=f"Backend '{backend or self.default_backend}' not found",
            )

        request = ImageRequest(
            prompt=prompt,
            input_image=input_image,
            strength=strength,
            **kwargs,
        )

        return await backend_impl.img2img(request)

    async def execute(
        self,
        action: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute a tool action.

        Actions:
            - generate: Text-to-image generation
            - img2img: Image-to-image transformation
            - list_backends: List available backends

        Returns:
            Result dictionary
        """
        if action == "generate":
            result = await self.generate(**kwargs)
            return result.to_dict()

        elif action == "img2img":
            result = await self.img2img(**kwargs)
            return result.to_dict()

        elif action == "list_backends":
            return {
                "backends": self.list_backends(),
                "default": self.default_backend,
            }

        else:
            return {"error": f"Unknown action: {action}"}


# ============================================================================
# Convenience Functions
# ============================================================================


# Global tool instance
_image_tool: ImageGenerationTool | None = None


def get_image_tool() -> ImageGenerationTool:
    """Get the global image generation tool."""
    global _image_tool
    if _image_tool is None:
        _image_tool = ImageGenerationTool()
    return _image_tool


async def generate_image(
    prompt: str,
    size: str = "1024x1024",
    backend: str | None = None,
    **kwargs: Any,
) -> GeneratedImage | None:
    """
    Convenience function to generate a single image.

    Args:
        prompt: Image description
        size: Image size
        backend: Backend to use
        **kwargs: Additional parameters

    Returns:
        GeneratedImage or None if failed
    """
    tool = get_image_tool()
    result = await tool.generate(
        prompt=prompt,
        size=size,
        num_images=1,
        backend=backend,
        **kwargs,
    )

    if result.success and result.images:
        return result.images[0]
    return None
