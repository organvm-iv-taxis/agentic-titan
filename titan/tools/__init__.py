"""
Titan Tools - Advanced tools for AI agents.

Provides:
- ImageGenerationTool: AI image generation (Stable Diffusion, DALL-E)
- Microsoft365Tool: Microsoft 365 Graph API integration
"""

from titan.tools.image_gen import (
    ImageGenerationTool,
    ImageRequest,
    GeneratedImage,
    ImageGenerationResult,
    StableDiffusionBackend,
    DallEBackend,
    generate_image,
    get_image_tool,
)

from titan.tools.m365 import (
    Microsoft365Tool,
    GraphClient,
    M365User,
    EmailMessage,
    CalendarEvent,
    DriveItem,
    TeamsMessage,
    get_m365_tool,
)

__all__ = [
    # Image Generation
    "ImageGenerationTool",
    "ImageRequest",
    "GeneratedImage",
    "ImageGenerationResult",
    "StableDiffusionBackend",
    "DallEBackend",
    "generate_image",
    "get_image_tool",
    # Microsoft 365
    "Microsoft365Tool",
    "GraphClient",
    "M365User",
    "EmailMessage",
    "CalendarEvent",
    "DriveItem",
    "TeamsMessage",
    "get_m365_tool",
]
