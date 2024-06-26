#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate

# import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from egoexo_dataset import EgoExoDataset
from utils import render_forward_splat, get_camera_params


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": (
        "input_image",
        "edit_prompt",
        "edited_image",
    ),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


# def convert_to_np(image, resolution):
#     image = image.convert("RGB").resize((resolution, resolution))
#     return np.array(image).transpose(2, 0, 1)
def convert_to_np(image):
    image = image.convert("RGB")
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the Inpainting UNet from the pretrained UNet.")
    in_channels = 13
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels,
            out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :9, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # unet.down_blocks.requires_grad_(False)
    # # unet.mid_block.requires_grad_(False)
    # unet.up_blocks.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    train_dataset = EgoExoDataset(
        data_dir=args.train_data_dir,
        split="train",
        img_size=args.resolution,
        random_split=True,
        subsampling_rate=1.0,
        train_val_split=0.8,
    )
    print("length of train_dataset: ", len(train_dataset))

    test_dataset = EgoExoDataset(
        data_dir=args.train_data_dir,
        split="test",
        img_size=args.resolution,
        random_split=True,
        subsampling_rate=1.0,
        train_val_split=0.8,
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            (
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution)
            ),
            # (
            #     transforms.RandomHorizontalFlip()
            #     if args.random_flip
            #     else transforms.Lambda(lambda x: x)
            # ),
        ]
    )

    def preprocess_images(examples):
        input_pose_images = convert_to_np(examples["input_pose_image"])
        input_timestep_images = convert_to_np(examples["input_timestep_image"])
        edited_images = convert_to_np(examples["edited_image"])
        input_pose_depth_images = examples["input_pose_depth"][None,]
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate(
            [
                input_pose_images,
                input_timestep_images,
                edited_images,
                input_pose_depth_images,
            ],
        )
        images = torch.tensor(images)
        preprocessed = train_transforms(images)

        preprocessed_images = preprocessed[:-1]
        preprocessed_images = 2 * (preprocessed_images / 255) - 1

        preprocessed_depth = preprocessed[-1:]

        # visualizer
        # tensor_to_pil_rgb = transforms.ToPILImage(mode="RGB")
        # tensor_to_pil_rgb(preprocessed_images[:3].cpu() / 2 + 0.5).show()

        return preprocessed_images, preprocessed_depth

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images, preprocessed_depth = preprocess_images(examples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        input_pose_images, input_timestep_images, edited_images = (
            preprocessed_images.chunk(3)
        )
        input_pose_images = input_pose_images.reshape(
            -1, 3, args.resolution, args.resolution
        )
        input_timestep_images = input_timestep_images.reshape(
            -1, 3, args.resolution, args.resolution
        )
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        # clip all the images to [-1, 1]
        input_pose_images = torch.clamp(input_pose_images, -1, 1)
        input_timestep_images = torch.clamp(input_timestep_images, -1, 1)
        edited_images = torch.clamp(edited_images, -1, 1)

        depth_images = preprocessed_depth.reshape(args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["input_pose_pixel_values"] = input_pose_images
        examples["depth_pixel_values"] = depth_images
        examples["input_timestep_pixel_values"] = input_timestep_images
        examples["edited_pixel_values"] = edited_images
        examples["input_pose_extrinsics"] = torch.tensor(
            examples["input_pose_extrinsics"]
        )
        examples["input_pose_intrinsics"] = torch.tensor(
            examples["input_pose_intrinsics"]
        )
        examples["input_timestep_extrinsics"] = torch.tensor(
            examples["input_timestep_extrinsics"]
        )
        examples["input_timestep_intrinsics"] = torch.tensor(
            examples["input_timestep_intrinsics"]
        )

        # Preprocess the captions.
        captions = [
            "as photorealistic as possible"
            for _ in range(len(examples["input_pose_pixel_values"]))
        ]
        examples["input_ids"] = tokenize_captions(captions)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(
                range(args.max_train_samples)
            )
        # Set the training transforms
        train_dataset.with_transform(preprocess_train)
        test_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        input_pose_pixel_values = torch.stack(
            [example["input_pose_pixel_values"] for example in examples]
        )
        input_pose_pixel_values = input_pose_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        input_timestep_pixel_values = torch.stack(
            [example["input_timestep_pixel_values"] for example in examples]
        )
        input_timestep_pixel_values = input_timestep_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        edited_pixel_values = torch.stack(
            [example["edited_pixel_values"] for example in examples]
        )
        edited_pixel_values = edited_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        depth_pixel_values = torch.stack(
            [example["depth_pixel_values"] for example in examples]
        )
        depth_pixel_values = depth_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        source_intrinsics = torch.stack(
            [example["input_pose_intrinsics"] for example in examples]
        )
        source_extrinsics = torch.stack(
            [example["input_pose_extrinsics"] for example in examples]
        )
        target_intrinsics = torch.stack(
            [example["input_timestep_intrinsics"] for example in examples]
        )
        target_extrinsics = torch.stack(
            [example["input_timestep_extrinsics"] for example in examples]
        )

        input_ids = torch.stack([example["input_ids"] for example in examples])

        original_pixel_values = torch.cat(
            [input_pose_pixel_values, input_timestep_pixel_values], dim=1
        )

        return {
            "original_pixel_values": original_pixel_values,
            "depth_pixel_values": depth_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "source_intrinsics": source_intrinsics,
            "source_extrinsics": source_extrinsics,
            "target_intrinsics": target_intrinsics,
            "target_extrinsics": target_extrinsics,
            "input_ids": input_ids,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    test_dataloader = accelerator.prepare(test_dataloader)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):

                source_intrinsics = batch["source_intrinsics"].to(weight_dtype)
                source_extrinsics = batch["source_extrinsics"].to(weight_dtype)
                target_intrinsics = batch["target_intrinsics"].to(weight_dtype)
                target_extrinsics = batch["target_extrinsics"].to(weight_dtype)

                R_source, T_source, K_source = get_camera_params(
                    source_extrinsics, source_intrinsics, args.resolution
                )
                R_target, T_target, K_target = get_camera_params(
                    target_extrinsics, target_intrinsics, args.resolution
                )
                warp_feature, warp_disp, warp_mask = render_forward_splat(
                    batch["original_pixel_values"][:, 0] / 2 + 0.5,
                    batch["depth_pixel_values"],
                    R_source.to(torch.float32),
                    T_source.to(torch.float32),
                    K_source.to(torch.float32),
                    R_target.to(torch.float32),
                    T_target.to(torch.float32),
                    K_target.to(torch.float32),
                )
                tensor_to_pil_rgb = transforms.ToPILImage(mode="RGB")
                # tensor_to_pil_rgb(warp_feature[0].cpu()).show()
                batch["original_pixel_values"][:, 0] = warp_feature * 2 - 1

                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(
                    batch["edited_pixel_values"]
                    .view(-1, 3, args.resolution, args.resolution)
                    .to(weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(
                    batch["original_pixel_values"]
                    .view(-1, 3, args.resolution, args.resolution)
                    .to(weight_dtype)
                ).latent_dist.mode()
                feature_size = original_image_embeds.shape[-1]
                original_image_embeds = original_image_embeds.reshape(
                    bsz, -1, feature_size, feature_size
                )

                # Initialize an all one mask with the same size as original_pixel_values
                mask = (1 - warp_mask).to(weight_dtype)
                # mask = torch.ones_like(batch["original_pixel_values"][:, :, 0, :, :])
                mask = torch.nn.functional.interpolate(
                    mask, size=(feature_size, feature_size)
                )
                mask = mask.to(device=original_image_embeds.device)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator
                    )
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(
                        tokenize_captions([""]).to(accelerator.device)
                    )[0]
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states
                    )

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(
                            image_mask_dtype
                        )
                        * (random_p < 3 * args.conditioning_dropout_prob).to(
                            image_mask_dtype
                        )
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat(
                    [
                        noisy_latents,
                        mask,
                        original_image_embeds,
                    ],
                    dim=1,
                )

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                logger.info(f"Running validation...")
                # create pipeline
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                def prepare_mask_latents(
                    mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
                ):
                    # resize the mask to latents shape as we concatenate the mask to the latents
                    # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
                    # and half precision
                    mask = torch.nn.functional.interpolate(
                        mask,
                        size=(
                            height // pipeline.vae_scale_factor,
                            width // pipeline.vae_scale_factor,
                        ),
                    )
                    mask = mask.to(device=device, dtype=dtype)

                    masked_image = masked_image.to(device=device, dtype=dtype)

                    if masked_image.shape[1] == 4:
                        masked_image_latents = masked_image
                    else:
                        original_image_embeds = pipeline.vae.encode(
                            masked_image
                        ).latent_dist.mode()
                        masked_image_latents = original_image_embeds
                        feature_size = masked_image_latents.shape[-1]
                        masked_image_latents = masked_image_latents.reshape(
                            1, -1, feature_size, feature_size
                        )

                    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
                    if mask.shape[0] < batch_size:
                        if not batch_size % mask.shape[0] == 0:
                            raise ValueError(
                                "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                                f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                                " of masks that you pass is divisible by the total requested batch size."
                            )
                        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
                    if masked_image_latents.shape[0] < batch_size:
                        if not batch_size % masked_image_latents.shape[0] == 0:
                            raise ValueError(
                                "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                                f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                                " Make sure the number of images that you pass is divisible by the total requested batch size."
                            )
                        masked_image_latents = masked_image_latents.repeat(
                            batch_size // masked_image_latents.shape[0], 1, 1, 1
                        )

                    mask = (
                        torch.cat([mask] * 2) if do_classifier_free_guidance else mask
                    )
                    masked_image_latents = (
                        torch.cat([masked_image_latents] * 2)
                        if do_classifier_free_guidance
                        else masked_image_latents
                    )

                    # aligning device to prevent device errors when concating it with the latent model input
                    masked_image_latents = masked_image_latents.to(
                        device=device, dtype=dtype
                    )
                    return mask, masked_image_latents

                pipeline.prepare_mask_latents = prepare_mask_latents

                # run inference
                validation_iter = iter(test_dataloader)
                edited_images = []
                gt_images = []
                conditioning_images = []
                original_images = []
                transformed_images = []
                with torch.autocast(
                    str(accelerator.device).replace(":0", ""),
                    enabled=accelerator.mixed_precision == "fp16",
                ):
                    for _ in range(args.num_validation_images):
                        try:
                            batch = next(validation_iter)
                        except StopIteration:
                            validation_iter = iter(test_dataloader)
                            batch = next(validation_iter)

                        original_image = transforms.ToPILImage(mode="RGB")(
                            batch["original_pixel_values"][:, 0]
                            .squeeze()
                            .to(accelerator.device)
                            / 2
                            + 0.5
                        )
                        original_images.append(original_image)

                        source_intrinsics = batch["source_intrinsics"].to(weight_dtype)
                        source_extrinsics = batch["source_extrinsics"].to(weight_dtype)
                        target_intrinsics = batch["target_intrinsics"].to(weight_dtype)
                        target_extrinsics = batch["target_extrinsics"].to(weight_dtype)

                        R_source, T_source, K_source = get_camera_params(
                            source_extrinsics, source_intrinsics, args.resolution
                        )
                        R_target, T_target, K_target = get_camera_params(
                            target_extrinsics, target_intrinsics, args.resolution
                        )
                        warp_feature, warp_disp, warp_mask = render_forward_splat(
                            batch["original_pixel_values"][:, 0].squeeze(1) / 2 + 0.5,
                            batch["depth_pixel_values"],
                            R_source.to(torch.float32),
                            T_source.to(torch.float32),
                            K_source.to(torch.float32),
                            R_target.to(torch.float32),
                            T_target.to(torch.float32),
                            K_target.to(torch.float32),
                        )
                        tensor_to_pil_rgb = transforms.ToPILImage(mode="RGB")
                        batch["original_pixel_values"][:, 0] = (
                            warp_feature.unsqueeze(1) * 2 - 1
                        )

                        transformed_image = transforms.ToPILImage(mode="RGB")(
                            batch["original_pixel_values"][:, 0]
                            .squeeze()
                            .to(accelerator.device)
                            / 2
                            + 0.5
                        )
                        transformed_images.append(transformed_image)
                        conditioning_image = transforms.ToPILImage(mode="RGB")(
                            batch["original_pixel_values"][:, 1]
                            .squeeze()
                            .to(accelerator.device)
                            / 2
                            + 0.5
                        )
                        conditioning_images.append(conditioning_image)
                        gt_image = transforms.ToPILImage(mode="RGB")(
                            batch["edited_pixel_values"]
                            .squeeze()
                            .to(accelerator.device)
                            / 2
                            + 0.5
                        )
                        # mask_image = (
                        #     torch.ones_like(
                        #         batch["original_pixel_values"][:, :, 0, :, :]
                        #     )
                        #     .to(accelerator.device)
                        #     .squeeze(0)
                        # )
                        mask_image = (1 - warp_mask).to(accelerator.device).squeeze(0)
                        input_image = (
                            batch["original_pixel_values"]
                            .squeeze()
                            .to(accelerator.device)
                            / 2
                            + 0.5
                        )
                        edited_images.append(
                            pipeline(
                                "as photorealistic as possible",
                                image=input_image,
                                mask_image=mask_image,
                                num_inference_steps=50,
                                # image_guidance_scale=1.5,
                                guidance_scale=1.0,
                                generator=generator,
                            ).images[0]
                        )
                        gt_images.append(gt_image)

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                        for edited_image in edited_images:
                            wandb_table.add_data(
                                wandb.Image(original_image),
                                wandb.Image(edited_image),
                                args.validation_prompt,
                            )
                        tracker.log({"validation": wandb_table})
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

                visualization_path = os.path.join(args.output_dir, "visualization")
                os.makedirs(visualization_path, exist_ok=True)
                epoch_visualization_path = os.path.join(
                    visualization_path, f"epoch_{epoch}"
                )
                os.makedirs(epoch_visualization_path, exist_ok=True)
                for i, (
                    edited_image,
                    gt_image,
                    conditioning_image,
                    original_image,
                    transformed_image,
                ) in enumerate(
                    zip(
                        edited_images,
                        gt_images,
                        conditioning_images,
                        original_images,
                        transformed_images,
                    )
                ):
                    edited_image.save(
                        os.path.join(epoch_visualization_path, f"edited_image_{i}.png")
                    )
                    gt_image.save(
                        os.path.join(epoch_visualization_path, f"gt_image_{i}.png")
                    )
                    conditioning_image.save(
                        os.path.join(
                            epoch_visualization_path, f"conditioning_image_{i}.png"
                        )
                    )
                    original_image.save(
                        os.path.join(
                            epoch_visualization_path, f"original_image_{i}.png"
                        )
                    )
                    transformed_image.save(
                        os.path.join(
                            epoch_visualization_path, f"transformed_image_{i}.png"
                        )
                    )

                del pipeline
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if args.validation_prompt is not None:
            edited_images = []
            pipeline = pipeline.to(accelerator.device)
            with torch.autocast(str(accelerator.device).replace(":0", "")):
                for _ in range(args.num_validation_images):
                    edited_images.append(
                        pipeline(
                            args.validation_prompt,
                            image=original_image,
                            num_inference_steps=50,
                            image_guidance_scale=1.5,
                            guidance_scale=2.5,
                            generator=generator,
                        ).images[0]
                    )

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                    for edited_image in edited_images:
                        wandb_table.add_data(
                            wandb.Image(original_image),
                            wandb.Image(edited_image),
                            args.validation_prompt,
                        )
                    tracker.log({"test": wandb_table})

    accelerator.end_training()


if __name__ == "__main__":
    main()
