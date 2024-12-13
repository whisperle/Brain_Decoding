import itertools
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn

from Train import (
    parse_arguments,
    build_model,
    prepare_data,
)
import utils
from matplotlib import pyplot as plt
from torchvision import transforms
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2


CACHE_DIR = os.environ.get("CACHE_DIR", "/scratch/yz10381/CODES/IVP/inference/Brain_Decoding/cache_dir")


def validation_test(
    args,
    model, clip_img_embedder, diffusion_prior, # for brain recon
    autoenc, cnx, mean, std, blur_augs, # for blurry recon
    vector_suffix, clip_text_model, clip_convert, processor, diffusion_engine, # for text generation
    test_dl,
    epoch,
    device,
    save_dir="test",
    num_eval_samples=None,
):
    os.makedirs(save_dir, exist_ok=True)
    plotting = True
    num_samples_per_image = 1
    assert num_samples_per_image == 1


    clip_scale = args.clip_scale
    use_prior = args.use_prior
    prior_scale = args.prior_scale
    blurry_recon = args.blurry_recon
    mse = nn.MSELoss()
    

    test_recon_cossim, test_loss_prior_total, test_recon_mse, test_loss_clip_total, test_blurry_pixcorr = 0., 0., 0., 0., 0.
    test_fwd_percent_correct, test_bwd_percent_correct = 0., 0.
    test_losses = []
    test_image, test_voxel, test_coords = None, None, None
    all_clipvoxels, all_predcaptions, all_recons = None, None, None
    with torch.no_grad(), torch.amp.autocast('cuda'): 
        # Add progress bar for test dataloader
        if num_eval_samples is not None:
            test_dl = itertools.islice(test_dl, num_eval_samples)
            num_test = num_eval_samples
        else:
            num_test = len(test_dl)

        test_progress = tqdm(test_dl, desc=f'Testing epoch {epoch}', leave=False, total=num_test)
        for test_i, (images, voxels, subj_idx, coords, image_idx) in enumerate(test_progress):
            images = images.to(device)
            voxels = voxels.to(device)
            coords = coords.to(device)
            image_idx = image_idx.to(device)
            # all test samples should be loaded per batch such that test_i should never exceed 0
            if len(images) != args.batch_size:
                print(f"Warning: Batch size mismatch. Expected {args.batch_size}, got {len(images)}")
                continue

            # Update progress bar description with current metrics
            if test_i > 0:  # Only update if we have accumulated some metrics
                test_progress.set_postfix({
                    'loss': f"{np.mean(test_losses[-(test_i+1):]):.4f}",
                    'fwd_acc': f"{test_fwd_percent_correct/(test_i+1):.4f}",
                    'bwd_acc': f"{test_bwd_percent_correct/(test_i+1):.4f}"
                })

            ## Average same-image repeats ##
            if test_image is None:
                voxel = voxels
                image = image_idx
                unique_image, sort_indices = torch.unique(image, return_inverse=True) # this will break multi gpu inference if wanting to do all clip
                for im in unique_image:
                    locs = torch.where(im == image_idx)[0]
                    if len(locs)==1:
                        locs = locs.repeat(3)
                    elif len(locs)==2:
                        locs = locs.repeat(2)[:3]
                    assert len(locs)==3
                    if test_image is None:
                        test_image = torch.Tensor(images[locs,0][None])
                        test_voxel = voxels[locs][None]
                        test_coords = coords[locs][None]
                    else:
                        test_image = torch.vstack((test_image, torch.Tensor(images[locs,0][None])))
                        test_voxel = torch.vstack((test_voxel, voxels[locs][None]))
                        test_coords = torch.vstack((test_coords, coords[locs][None]))
            loss=0.
                        
            test_indices = torch.arange(len(test_voxel))
            voxel = test_voxel[test_indices]
            coords = test_coords[test_indices]
            image = test_image[test_indices]

            clip_target = clip_img_embedder(image)
            for rep in range(3):
                backbone0, clip_voxels0, blurry_image_enc_ = model(voxel[:,rep], coords[:,rep])
                if rep==0:
                    clip_voxels = clip_voxels0
                    backbone = backbone0
                else:
                    clip_voxels += clip_voxels0
                    backbone += backbone0
            clip_voxels /= 3
            backbone /= 3

            # EVAL: Save retrieval submodule outputs
            if all_clipvoxels is None:
                all_clipvoxels = clip_voxels.cpu()
            else:
                all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.cpu()))

            # VALIDATION
            if clip_scale>0:
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    accelerator=None,
                    temp=.006)

                test_loss_clip_total += loss_clip.item()
                loss_clip = loss_clip * clip_scale
                loss += loss_clip
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
            
            
            # VALIDATION
            # for some evals, only doing a subset of the samples per batch because of computational cost
            # random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
            random_samps = np.arange(len(image))
            if use_prior:
                loss_prior, prior_out = diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                test_loss_prior_total += loss_prior.item()
                loss_prior *= prior_scale
                loss += loss_prior
                # TODO: this two line was not tested
                test_recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target[random_samps]).mean().item()
                test_recon_mse += mse(prior_out, clip_target[random_samps]).item()

            # EVAL: Feed voxels through OpenCLIP-bigG diffusion prior
            prior_out = diffusion_prior.p_sample_loop(backbone.shape, 
                        text_cond = dict(text_embed = backbone), 
                        cond_scale = 1., timesteps = 20)
            pred_caption_emb = clip_convert(prior_out)
            generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(generated_caption)
            all_predcaptions = np.hstack((all_predcaptions, generated_caption))
        
            # Feed diffusion prior outputs through unCLIP
            for i in range(len(voxel)):
                samples = utils.unclip_recon(prior_out[[i]],
                                diffusion_engine,
                                vector_suffix,
                                num_samples=num_samples_per_image)
                if all_recons is None:
                    all_recons = samples.cpu()
                else:
                    all_recons = torch.vstack((all_recons, samples.cpu()))
                if plotting:
                    plot_dirname = os.environ.get("PLOT_DIRNAME", f"plots")
                    plt_dir = osp.join(save_dir, plot_dirname)
                    os.makedirs(plt_dir, exist_ok=True)
                    for s in range(num_samples_per_image):
                        img = transforms.ToPILImage()(samples[s])
                        img.save(osp.join(plt_dir, f"recon_sample{i:05}_{s:05}.png"))
                    
                if blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    test_blurry_pixcorr += pixcorr.item()


            utils.check_loss(loss)                
            test_losses.append(loss.item())
        # END OF TEST LOOP
        # assert (test_i+1) == 1
        logs = {
            "epoch/epoch": epoch,
            "epoch/test_loss": np.mean(test_losses[-num_eval_samples:]),  # Only average losses from current test run
            "epoch/lr": lrs[-1],
            "epoch/test_fwd_acc": test_fwd_percent_correct / (test_i + 1),
            "epoch/test_bwd_acc": test_bwd_percent_correct / (test_i + 1),
        }

        if clip_scale > 0:
            logs.update({
                "epoch/test_loss_clip": test_loss_clip_total / (test_i + 1),
            })

        if blurry_recon:
            logs.update({
                "epoch/test_blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
            })

        if use_prior:
            logs.update({
                "epoch/test_loss_prior": test_loss_prior_total / (test_i + 1),
                "epoch/test_recon_cossim": test_recon_cossim / (test_i + 1),
                "epoch/test_recon_mse": test_recon_mse / (test_i + 1),
            })
        
        for k, v in logs.items():
            print(f"{k}: {v}")



def load_validation_components():
    # prep unCLIP
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
    offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38

    diffusion_engine = DiffusionEngine(network_config=network_config,
                        denoiser_config=denoiser_config,
                        first_stage_config=first_stage_config,
                        conditioner_config=conditioner_config,
                        sampler_config=sampler_config,
                        scale_factor=scale_factor,
                        disable_first_stage_autocast=disable_first_stage_autocast)
    # set to inference
    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)

    ckpt_path = f'{CACHE_DIR}/unclip6_epoch0_step110000.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])

    batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    print("vector_suffix", vector_suffix.shape)

    # setup text caption networks
    from transformers import AutoProcessor, AutoModelForCausalLM
    from modeling_git import GitForCausalLMClipEmb
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    clip_text_model.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
    clip_text_model.eval().requires_grad_(False)
    clip_text_seq_dim = 257
    clip_text_emb_dim = 1024
    clip_seq_dim = 256
    clip_emb_dim = 1664
    
    class CLIPConverter(torch.nn.Module):
        def __init__(self):
            super(CLIPConverter, self).__init__()
            self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
            self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
        def forward(self, x):
            x = x.permute(0,2,1)
            x = self.linear1(x)
            x = self.linear2(x.permute(0,2,1))
            return x
            
    clip_convert = CLIPConverter()
    state_dict = torch.load(f"{CACHE_DIR}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
    clip_convert.load_state_dict(state_dict, strict=True)
    clip_convert.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
    del state_dict

    return vector_suffix, clip_text_model, clip_convert, processor, diffusion_engine, # for text generation
    

if __name__ == "__main__":
    # Load the model
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_type = torch.float16  # Change depending on your mixed_precision

    clip_img_embedder, model, diffusion_prior, autoenc, cnx, mean, std, blur_augs, param_count_dict = build_model(args, device, data_type)
    
    diffusion_prior = diffusion_prior.to(device)

    # print(model)

    epoch_start, losses, test_losses, lrs, resumed = utils.load_ckpt(args, model, tag="iter_135000")
    print(f"model_name: {args.model_name}")    
    print(f"epoch_start: {epoch_start}")
    print(f"len(losses): {len(losses)}, mean(losses): {np.mean(losses)}")
    print(f"len(test_losses): {len(test_losses)}, mean(test_losses): {np.mean(test_losses)}")
    print(f"len(lrs): {len(lrs)}")
    print(f"resumed: {resumed}")
    train_dl, test_dl, num_test, num_iterations_per_epoch = prepare_data(args, data_type)
    print(f"train_dl: {train_dl}, len(train_dl): {len(train_dl)}")
    print(f"test_dl: {test_dl}, len(test_dl): {len(test_dl)}")

    vector_suffix, clip_text_model, clip_convert, processor, diffusion_engine = load_validation_components()
    # vector_suffix, clip_text_model, clip_convert, processor, diffusion_engine = None, None, None, None, None
    validation_test(
        args,
        model, clip_img_embedder, diffusion_prior,
        autoenc, cnx, mean, std, blur_augs, # these are for blurry recon
        vector_suffix, clip_text_model, clip_convert, processor, diffusion_engine, # these are for text generation
        test_dl,
        epoch_start,
        device,
        # num_eval_samples=1,
    )