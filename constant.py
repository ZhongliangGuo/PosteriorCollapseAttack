from loss import PosteriorCollapseLoss
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline

X_ADV_FOLDER_NAME = "attacked"
ADV_RECONSTRUCT_FOLDER_NAME = "adv_reconstruct"
CLEAN_RECONSTRUCT_FOLDER_NAME = "clean_reconstruct"

IMPLEMENTED_MODEL_ID = {
    "sd14": {
        "pipe": StableDiffusionImg2ImgPipeline,
        "model_id": "CompVis/stable-diffusion-v1-4",
    },
    "sd15": {
        "pipe": StableDiffusionImg2ImgPipeline,
        "model_id": "runwayml/stable-diffusion-v1-5",
    },
    "sdxl": {
        "pipe": StableDiffusionXLImg2ImgPipeline,
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    },
    "sdxlr": {
        "pipe": StableDiffusionXLImg2ImgPipeline,
        "model_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
    },
    "sd20": {
        "pipe": StableDiffusionImg2ImgPipeline,
        "model_id": "stabilityai/stable-diffusion-2"
    }
}

IMPLEMENTED_LOSS = {
    'ReverseKL': PosteriorCollapseLoss
}

PRESET_PROMPT = {
    -1: "",
    0: None,
    1: "add some snow",
    2: "apply sunset lighting",
    3: "make it like a watercolor painting"
}
