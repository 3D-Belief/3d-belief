import sys
import os
import wandb
import hydra
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion.diffusion_temporal import (
    DiffusionTemporal,
    Trainer,
)
import data_io
from einops import rearrange

from splat import SplatBeliefState
from splat.ply_export import export_ply, export_gaussians_to_ply
from splat.decoder import get_decoder
from splat.encoder import get_encoder
from embodied.semantic_mapper import SemanticMapper
import torch
import random
import imageio
import copy
import numpy as np
from utils.vision_utils import *
from torchvision.utils import make_grid
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from configurations.inference import temporal_indices
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from utils.model_utils import build_2d_model

import clip
import open_clip


@hydra.main(
    version_base=None, config_path="../configurations/", config_name="config",
)

def train(cfg: DictConfig):
    run_dir = cfg.results_folder
    print(f"run dir: {run_dir}")
    adjacent_angle = cfg.adjacent_angle
    adjacent_distance = cfg.adjacent_distance
    normalize = normalize_to_neg_one_to_one
    # initialize the accelerator at the beginning
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
    )

    # dataset
    train_batch_size = cfg.batch_size
    dataset = data_io.get_dataset(cfg)
    dl = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    seed = cfg.seed
    torch.manual_seed(seed)

    depth_mode = cfg.depth_mode
    extended_visualization = cfg.extended_visualization
    use_history = cfg.use_history

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    use_semantic = cfg.model.encoder.use_semantic
    use_reg_model = cfg.model.encoder.use_reg_model
    reg_model_name = cfg.model.encoder.reg_model_name
    use_depth_mask = cfg.model.encoder.use_depth_mask
    semantic_config = cfg.semantic_config
    semantic_mode = cfg.semantic_mode
    semantic_viz = cfg.semantic_viz

    inference_save_scene = cfg.inference_save_scene

    clip_model = None
    if use_semantic:
        # clip_model, _ = clip.load("ViT-B/16", device="cuda")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            precision="fp16",
        )
        clip_model = clip_model.to("cuda")

        clip_model.eval()

    dino_model = None
    if use_semantic and use_reg_model: 
        dino_model = build_2d_model(model_name=reg_model_name)

    text_encoder = None
    text_tokenizer = None
    if use_semantic and semantic_mode == "embed":
        text_encoder = clip_model.encode_text
        # text_tokenizer = clip.tokenize
        text_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    
    semantic_mapper = None
    if use_semantic:
        semantic_mapper = SemanticMapper(
            config_path=semantic_config, 
            mode=semantic_mode, 
            text_encoder=text_encoder, 
            text_tokenizer=text_tokenizer, 
            semantic_viz=semantic_viz
        )

    model = SplatBeliefState(
        encoder,
        encoder_visualizer,
        decoder,
        semantic_mapper=semantic_mapper,
        depth_mode=depth_mode,
        extended_visualization=extended_visualization,
        use_history=use_history,
        use_semantic=use_semantic,
        semantic_mode=semantic_mode,
        inference_mode=cfg.model.encoder.inference_mode,
        clip_model=clip_model,
        dino_model=dino_model,
        use_depth_mask=use_depth_mask,
    ).cuda()

    diffusion = DiffusionTemporal(
        model,
        image_size=dataset.image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=cfg.sampling_steps,
        loss_type="l2",  # L1 or L2
        objective="pred_x0",
        beta_schedule="cosine",
        use_guidance=cfg.use_guidance,
        guidance_scale=cfg.guidance_scale,
        temperature=cfg.temperature,
        clean_target=cfg.clean_target,
        use_semantic=use_semantic,
        use_reg_model=use_reg_model,
        use_depth_mask=use_depth_mask,
    ).cuda()

    print(f"using lr {cfg.lr}")
    trainer = Trainer(
        diffusion,
        dataloader=dl,
        train_batch_size=train_batch_size,
        train_lr=cfg.lr,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        sample_every=1000,
        wandb_every=50,
        save_every=5000,
        num_samples=1,
        warmup_period=1_000,
        checkpoint_path=cfg.checkpoint_path,
        wandb_config=cfg.wandb,
        run_name=cfg.name,
        accelerator=accelerator,
        cfg=cfg,
        use_semantic=use_semantic,
    )

    if cfg.inference_sample_from_dataset:
        interesting_indices = sample_video_idx_from_dataset(
            dataset, num_samples=cfg.inference_num_samples,
            min_frames=cfg.inference_min_frames, max_frames=cfg.inference_max_frames, 
        )
    else:
        interesting_indices = temporal_indices.interesting_indices

    with torch.no_grad():
        for i in interesting_indices:
            # import ipdb; ipdb.set_trace()
            print(f"Starting rendering sample {i}")
            video_idx = i[0]

            if len(i)==2:
                num_frames = i[1]

                data = dataset.data_for_temporal(
                    video_idx=video_idx, frames_render=num_frames
                )
            elif len(i)==3:
                start_end = [i[1], i[2]]
                data = dataset.data_for_temporal(
                    video_idx=video_idx, frames_render=start_end
                )
                num_frames = i[2]-i[1]+1
            else:
                raise ValueError("len(interesting_indices) should be 2 or 3")

            rgb_frames = data[1]
            rgbs = [(frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8') for frame in rgb_frames]

            start_idx = data[2]
            end_idx = data[3]

            print(
                f"start_idx: {start_idx}, end_idx: {end_idx}, num_frames_render: {num_frames}"
            )

            ## create save folder for the sample
            save_folder_sample = os.path.join(run_dir, f"visuals_{video_idx}_{start_idx}_{end_idx}_{seed}")
            os.makedirs(
                save_folder_sample, exist_ok=True,
            )

            save_folder_gt_frames = os.path.join(run_dir, f"visuals_{video_idx}_{start_idx}_{end_idx}_{seed}", "GT_frames")
            os.makedirs(
                save_folder_gt_frames, exist_ok=True,
            )

            GT_f = os.path.join(save_folder_sample, f"GT_exploration.mp4")
            imageio.mimwrite(GT_f, rgbs, fps=10, quality=10)

            # save all frames
            for p, frame in enumerate(rgbs):
                Image.fromarray(frame).save(
                    os.path.join(save_folder_gt_frames, f"GT_{p}.png")
                )
        
            ## load data
            video_dict = data[0]
            data_rgbs = video_dict["rgbs"]
            render_poses = video_dict["render_poses"]
            abs_camera_poses = video_dict["abs_camera_poses"]
            intrinsics = video_dict["intrinsics"]
            image_shape = video_dict["image_shape"]
            near = video_dict["near"]
            far = video_dict["far"]
            lang = video_dict["lang"]

            video_length = len(render_poses)
            assert video_length==num_frames

            ## compute key frame indices based on rotation difference
            key_frame_indices = []
            z_start = render_poses[0][0][:, 2][:3]
            t_start = render_poses[0][0][:, 3][:3]
            z_previous = z_start
            t_previous = t_start
            for idx in range(1, num_frames):
                current_pose = render_poses[idx][0]
                z_idx = current_pose[:, 2][:3]  # current forward vector
                t_idx = current_pose[:, 3][:3]  # current translation
                # import ipdb; ipdb.set_trace()
                angle = rotation_angle(z_previous, z_idx)
                distance = torch.norm(t_idx - t_previous)
                if angle > adjacent_angle or distance > adjacent_distance or idx==num_frames-1: # must include the last
                    key_frame_indices.append(idx)
                    z_previous = z_idx
                    t_previous = t_idx

            trainer.model.model.reset_timestep()
            trainer.ema.ema_model.model.reset_timestep()
            previous_t = 0
            inp = {}
            inp["ctxt_c2w"] = torch.cat(render_poses[:1], dim=0)
            inp["ctxt_rgb"] = torch.cat(data_rgbs[:1], dim=0)
            inp["ctxt_abs_camera_poses"] = torch.cat(abs_camera_poses[:1], dim=0)
            
            # list to store generated frames for all timesteps visualization
            all_generated_frames_history = []  # list of lists: [timestep][frame_predictions]
            observed_keyframes = [] # track which keyframes have been observed at each timestep
            
            for state_t, update_t in enumerate(key_frame_indices):
                ## construct inp dict
                inp["trgt_c2w"] = render_poses[update_t]
                inp["trgt_rgb"] = data_rgbs[update_t]
                inp["trgt_abs_camera_poses"] = abs_camera_poses[update_t]
                inp["intrinsics"] = intrinsics[0]
                inp["image_shape"] = image_shape
                inp["render_poses"] = torch.cat(render_poses, dim=0)
                inp["near"] = torch.tensor(near)
                inp["far"] = torch.tensor(far)
                inp["lang"] = lang
                # import ipdb; ipdb.set_trace()

                inp = to_gpu(inp, "cuda")
                for k in inp.keys():
                    if not k=="num_frames_render":
                        inp[k] = inp[k].unsqueeze(0)
                
                inp["num_frames_render"] = num_frames

                out = trainer.ema.ema_model.sample(batch_size=1, inp=inp, state_t=state_t)
                
                ## autoregressively extend the scene
                if not state_t==len(key_frame_indices)-1:
                    # input
                    imagine_input = copy.deepcopy(inp) 
                    imagine_input.update({
                        "ctxt_c2w": torch.cat(render_poses[update_t:update_t+1], dim=0),
                        "ctxt_rgb": normalize(out["images"]),
                        "ctxt_abs_camera_poses": torch.cat(abs_camera_poses[update_t:update_t+1], dim=0)
                    })
                    # copy current model
                    imagine_model = copy.deepcopy(trainer.ema.ema_model)
                    for imagine_t in range(state_t+1, len(key_frame_indices)):
                        imagine_input["trgt_c2w"] = render_poses[key_frame_indices[imagine_t]]
                        imagine_input["trgt_rgb"] = data_rgbs[key_frame_indices[imagine_t]]
                        imagine_input["trgt_abs_camera_poses"] = abs_camera_poses[key_frame_indices[imagine_t]]
                        imagine_input["intrinsics"] = intrinsics[0]
                        imagine_input["image_shape"] = image_shape
                        imagine_input["render_poses"] = torch.cat(render_poses, dim=0)
                        imagine_input["near"] = torch.tensor(near)
                        imagine_input["far"] = torch.tensor(far)
                        imagine_input["lang"] = lang
                        if not imagine_t==len(key_frame_indices)-1:
                            imagine_input.pop("render_poses")
                        imagine_input = to_gpu(imagine_input, "cuda")
                        for k in imagine_input.keys():
                            if not k=="num_frames_render":
                                imagine_input[k] = imagine_input[k].unsqueeze(0)
                        inp["num_frames_render"] = num_frames
                        # import ipdb; ipdb.set_trace()
                        out = imagine_model.sample(batch_size=1, inp=imagine_input, state_t=imagine_t)
                        if not imagine_t==len(key_frame_indices)-1:
                            imagine_input["ctxt_c2w"] = render_poses[key_frame_indices[imagine_t]]
                            imagine_input["ctxt_rgb"] = normalize(out["images"])
                            imagine_input["ctxt_abs_camera_poses"] = abs_camera_poses[key_frame_indices[imagine_t]]

                if use_depth_mask:
                    frames, depth_frames, semantics, depth_masks = prepare_video_viz(out)
                else:
                    frames, depth_frames, semantics = prepare_video_viz(out)
                
                # store generated frames for this timestep's visualization
                all_generated_frames_history.append(frames)
                
                # update observed keyframes list
                observed_keyframes.append(update_t)
                

                # create save folder for this step
                save_folder_step = os.path.join(run_dir, f"visuals_{video_idx}_{start_idx}_{end_idx}_{seed}", f'step_{previous_t}')
                os.makedirs(
                    save_folder_step, exist_ok=True,
                )

                save_folder_rendered_frames = os.path.join(run_dir, f"visuals_{video_idx}_{start_idx}_{end_idx}_{seed}", f'step_{previous_t}', f'frames')
                os.makedirs(
                    save_folder_rendered_frames, exist_ok=True,
                )
                # save all frames
                for p, (gt_frame, denoised_frame) in enumerate(zip(rgbs, frames)):
                    frame = np.vstack((gt_frame, denoised_frame))
                    Image.fromarray(frame).save(
                        os.path.join(save_folder_rendered_frames, f"rendered_{p}.png")
                    )

                denoised_f = os.path.join(save_folder_step, f"denoised_view_circle.mp4")
                imageio.mimwrite(denoised_f, frames, fps=10, quality=10)
                denoised_f_depth = os.path.join(save_folder_step, f"denoised_view_circle_depth.mp4")
                imageio.mimwrite(denoised_f_depth, depth_frames, fps=10, quality=10)
                if use_depth_mask:
                    denoised_f_depth_mask = os.path.join(save_folder_step, f"denoised_view_circle_depth_mask.mp4")
                    imageio.mimwrite(denoised_f_depth_mask, depth_masks, fps=10, quality=10)
                denoised_f_semantic = os.path.join(save_folder_step, f"denoised_view_circle_semantic.mp4")
                imageio.mimwrite(denoised_f_semantic, semantics, fps=10, quality=10)
                
                # concate GT and rendered for visualization
                output_f = os.path.join(save_folder_step, "concatenated_video.mp4")
                assert len(rgbs) == len(frames), "Mismatch in the number of frames between GT and denoised videos."
                output_writer = imageio.get_writer(output_f, fps=10, quality=10)
                for gt_frame, denoised_frame in zip(rgbs, frames):
                    concatenated_frame = np.vstack((gt_frame, denoised_frame))
                    output_writer.append_data(concatenated_frame)
                output_writer.close()

                # save the scene
                ply_path = Path(f"{save_folder_step}/scene_{video_idx}_{start_idx}_{end_idx}.ply")
                vis_model = imagine_model if not state_t==len(key_frame_indices)-1 else trainer.ema.ema_model
                gaussians = vis_model.model.history_gaussians + vis_model.model.belief_gaussians
                gaussians = gaussians.float()
                
                if inference_save_scene:
                    export_gaussians_to_ply(
                        gaussians,
                        render_poses[0].to("cuda"),
                        ply_path
                    )
                
                # create timestep visualization for current step
                create_timestep_visualization(
                    rgbs=rgbs, 
                    frames=frames, 
                    key_frame_indices=key_frame_indices, 
                    previous_t=previous_t, 
                    update_t=update_t, 
                    save_folder_step=save_folder_step,
                    state_t=state_t
                )
                
                # create all-timesteps visualization showing model's beliefs up to this point
                create_all_timesteps_visualization(
                    rgbs=rgbs,
                    frames_history=all_generated_frames_history,
                    key_frame_indices=key_frame_indices,
                    observed_keyframes=observed_keyframes,  # pass the observed keyframes
                    save_folder_sample=save_folder_step,  # save in the step folder
                    current_state_t=state_t  # pass the current state_t
                )

                # update obs
                previous_t = update_t
                inp["ctxt_c2w"] = torch.cat(render_poses[previous_t:previous_t+1], dim=0)
                inp["ctxt_rgb"] = torch.cat(data_rgbs[previous_t:previous_t+1], dim=0)
                inp["ctxt_abs_camera_poses"] = torch.cat(abs_camera_poses[previous_t:previous_t+1], dim=0)
            
            
def create_timestep_visualization(
    rgbs, 
    frames, 
    key_frame_indices, 
    previous_t, 
    update_t, 
    save_folder_step,
    state_t
):
    """
    Create a visualization showing observations, belief, and ground truth at each timestep.
    
    Args:
        rgbs: List of ground truth frames (numpy arrays)
        frames: List of predicted frames from the model (numpy arrays)
        key_frame_indices: List of key frame indices
        previous_t: Previous timestep
        update_t: Current update timestep
        save_folder_step: Folder to save the visualization
        state_t: Current state timestep
    """
    
    # create figure with three rows (Obs, Belief @ Image view, GT obs @ Image view)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f"t={state_t}", fontsize=24, x=0.95, y=0.98)
    
    # row 1: Observation up to now (previous timestep)
    axes[0].set_title(f"Observation Frame (t={previous_t})")
    axes[0].imshow(rgbs[previous_t])
    axes[0].set_ylabel("Obs", fontsize=24)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # row 2: Belief at image view (predicted frame at current timestep)
    axes[1].set_title(f"Predicted Frame (t={update_t})")
    axes[1].imshow(frames[update_t])
    axes[1].set_ylabel("Belief\n@ Imag. view", fontsize=24)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # row 3: Ground truth at image view
    axes[2].set_title(f"Ground Truth Frame (t={update_t})")
    axes[2].imshow(rgbs[update_t])
    axes[2].set_ylabel("GT obs\n@ Imag. view", fontsize=24)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    # adjust layout
    plt.tight_layout()
    
    # Save figure
    visualization_path = os.path.join(save_folder_step, f"timestep_visualization.png")
    plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved timestep visualization to {visualization_path}")

def create_all_timesteps_visualization(
    rgbs, 
    frames_history,
    key_frame_indices, 
    save_folder_sample,
    observed_keyframes=None,
    current_state_t=None
):
    """
    create a visualization showing all timesteps together.
    args:
        rgbs: List of ground truth frames (numpy arrays)
        frames_history: List of lists of frames generated at each timestep
        key_frame_indices: List of key frame indices
        save_folder_sample: Base folder to save the visualization
        observed_keyframes: List of keyframes that have been observed (optional)
        current_state_t: Current state timestep index (optional)
    """    
    # make sure the first frame (t=0) is included in the visualization
    full_key_frame_indices = [0]  # Start with the initial frame (t=0)
    for idx in key_frame_indices:
        if idx not in full_key_frame_indices:  # Avoid duplicates
            full_key_frame_indices.append(idx)
    
    num_timesteps = len(full_key_frame_indices)
    
    # use latest state if current_state_t not provided
    if current_state_t is None:
        current_state_t = len(frames_history) - 1
    
    # if observed_keyframes is not provided, create it based on current_state_t
    if observed_keyframes is None:
        if current_state_t < len(key_frame_indices):
            observed_keyframes = [0] + key_frame_indices[:current_state_t+1]
        else:
            observed_keyframes = [0] + key_frame_indices
    
    # create figure with three rows and num_timesteps columns
    fig, axes = plt.subplots(3, num_timesteps, figsize=(4*num_timesteps, 12))
    
    # if there's only one timestep, make axes 2D
    if num_timesteps == 1:
        axes = axes.reshape(3, 1)
    
    # add row labels
    row_labels = ["Obs", "Belief\n@ Imag. view", "GT obs\n@ Imag. view"]
    for i, label in enumerate(row_labels):
        fig.text(0.01, 0.75 - i*0.25, label, fontsize=24, ha='left', va='center', rotation=90)
    
    for col, keyframe_idx in enumerate(full_key_frame_indices):
        # add column header
        axes[0, col].set_title(f"t={keyframe_idx}", fontsize=24)
        
        # row 1: Observation - shows the currently observed frame at this specific column/timestep
        if col == 0:
            axes[0, col].imshow(rgbs[0])
        else:
            if col <= current_state_t and col < len(full_key_frame_indices):
                prev_keyframe_idx = full_key_frame_indices[col]
                axes[0, col].imshow(rgbs[prev_keyframe_idx])
            else:
                # create a black image to indicate not yet observed
                black_img = np.zeros_like(rgbs[0])
                axes[0, col].imshow(black_img)
        
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        
        # row 2: Belief at image view - current model's prediction for this keyframe
        axes[1, col].imshow(frames_history[current_state_t][keyframe_idx])
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        
        # row 3: Ground truth for this keyframe
        axes[2, col].imshow(rgbs[keyframe_idx])
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])
    
    # add horizontal separator lines
    for i in range(1, 3):
        fig.add_artist(plt.Line2D([0.05, 0.99], [0.67 - (i-1)*0.33, 0.67 - (i-1)*0.33], 
                                 color='blue', alpha=0, linewidth=2, transform=fig.transFigure))
    
    # add a vertical line for each timestep
    for i in range(1, num_timesteps):
        x_pos = 0.03 + i * (0.97 / num_timesteps)
        fig.add_artist(plt.Line2D([x_pos, x_pos], [0.05, 0.95], 
                                 color='blue', alpha=0, linewidth=2, transform=fig.transFigure))
    
    # adjust layout
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # make space for row labels
    
    # save figure
    if current_state_t is not None:
        filename = f"all_timesteps_visualization_state_{current_state_t}.png"
    else:
        filename = f"all_timesteps_visualization.png"
        
    visualization_path = os.path.join(save_folder_sample, filename)
    plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved all timesteps visualization to {visualization_path}")
      
def sample_image_idx_from_dataset(dataset, num_samples=15):
    sampled_indices = []
    for i in range(num_samples):
        scene_idx = random.randint(0, len(dataset.scene_path_list) - 1)
        num_ctxt_frames = len(dataset.all_ctxt_rgb_files[scene_idx])
        num_trgt_frames = len(dataset.all_trgt_rgb_files[scene_idx])
        print(f"num ctxt frames: {num_ctxt_frames}, num trgt frames: {num_trgt_frames}")
        # randomly sample target and context frames from the corresponding lists
        if num_ctxt_frames == 1:
            ctxt_idx = [0]
        else:
            ctxt_idx = dataset.rng.choice(
                np.arange(0, num_ctxt_frames - 1), dataset.num_context, replace=False
            )
        if num_trgt_frames == 1:
            trgt_idx = [0]
        else:
            trgt_idx = dataset.rng.choice(
                np.arange(0, num_trgt_frames - 1), dataset.num_target, replace=False
            )
        sampled_indices.append((scene_idx, ctxt_idx[0], trgt_idx[0]))
    return sampled_indices

def sample_video_idx_from_dataset(dataset, num_samples=15, min_frames=20, max_frames=30):
    sampled_indices = []
    for i in range(num_samples):
        scene_idx = random.randint(0, len(dataset.scene_path_list) - 1)
        video_length = len(dataset.all_rgb_files[scene_idx])
        assert video_length>0
        num_frames = dataset.rng.choice(
            np.arange(min(min_frames, video_length-1), min(max_frames, video_length)), 1, replace=False
        )
        sampled_indices.append((scene_idx, num_frames[0]))
    return sampled_indices

def prepare_video_viz(out):
    frames = out["videos"]
    for f in range(len(frames)):
        frames[f] = rearrange(frames[f], "b h w c -> h (b w) c")

    depth_frames = out["depth_videos"]
    for f in range(len(depth_frames)):
        depth_frames[f] = rearrange(depth_frames[f], "(n b) h w -> n h (b w)", n=1)

    depth = torch.cat(depth_frames, dim=0)
    print(f"depth shape: {depth.shape}")

    depth = (
        torch.from_numpy(
            jet_depth(depth[:].cpu().detach().view(depth.shape[0], depth.shape[-1], -1))
        )
        * 255
    )
    # convert depth to list of images
    depth_frames = []
    for i in range(depth.shape[0]):
        depth_frames.append(depth[i].cpu().detach().numpy().astype(np.uint8))

    # depth masks
    if out["depth_masks"] is not None:
        depth_masks = out["depth_masks"]
        for f in range(len(depth_masks)):
            depth_masks[f] = rearrange(depth_masks[f], "b h w c -> (b h) w c")
            depth_masks[f] = np.repeat(depth_masks[f], 3, axis=2) 
        depth_masks = [
            depth_mask.astype(np.uint8)
            for depth_mask in depth_masks
        ]
    else:
        depth_masks = None

    # semantic
    semantics = out["semantic_videos"]
    
    if out["depth_masks"] is not None:
        ret = (
            frames,
            depth_frames,
            semantics,
            depth_masks
        )
    else:
        ret = (
            frames,
            depth_frames,
            semantics
        )

    return ret

if __name__ == "__main__":
    train()
