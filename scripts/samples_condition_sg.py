import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
import torchvision
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch_size",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='0,',
        help="gpus,support ddp",
    )
    parser.add_argument(
        "--outdir",
        required=False,
        default='',
        type=str,
        help="Where to write outputs to.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Sample from among top-k predictions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=200,
        help="Sample image numbers",
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        distributed_backend='ddp'
        gpus='0,1,2,3'
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data

    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None

    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step

class ImageLogger():
    def __init__(self, clamp=True):
        super().__init__()
        self.clamp = clamp


    def log_local(self, save_dir, images, start_index):
        #root = os.path.join(save_dir, "images", split)
        for k in images:
            for img_index in range(images[k].shape[0]):
                img=images[k][img_index]
                img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                img = img.transpose(0, 1).transpose(1, 2).squeeze(-1)
                img = img.numpy()
                img = (img * 255).astype(np.uint8)
                filename = "{}.png".format(start_index+img_index)
                path = os.path.join(save_dir,k, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(img).save(path)

            # grid = torchvision.utils.make_grid(images[k], nrow=4)
            #
            # grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            # grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            # grid = grid.numpy()
            # grid = (grid*255).astype(np.uint8)
            # filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
            #     k,
            #     global_step,
            #     current_epoch,
            #     batch_idx)
            # path = os.path.join(root, filename)
            # os.makedirs(os.path.split(path)[0], exist_ok=True)
            # Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, start_index, split="train",save_dir=None):
        pl_module.eval()
        with torch.no_grad():
            images = pl_module.log_images(batch, split=split)

        for k in images:
            os.makedirs(os.path.join(save_dir,k),exist_ok=True)
        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            # images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)

        self.log_local(save_dir, images, start_index)


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)

    print(ckpt)
    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(config))

    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)

    print(f"Global step: {global_step}")

    outdir = os.path.join(opt.resume,'samples',opt.outdir, "{}".format(int(time.time())))
    os.makedirs(outdir, exist_ok=True)
    # print("Writing samples to ", outdir)
    # for k in ["originals", "reconstructions", "samples"]:
    #     os.makedirs(os.path.join(outdir, k), exist_ok=True)

    imageLogger=ImageLogger()
    #pl_module, batch, batch_idx, split = "train"
    val_dataloader=dsets._val_dataloader()
    #iter=iter(val_dataloader)
    batch_size=config["data"]["params"]["batch_size"]
    total_length=len(val_dataloader)
    for batch_idx,batch in enumerate(val_dataloader):
        start_index = batch_size*batch_idx
        imageLogger.log_img(model, batch, start_index, split="test",save_dir=outdir)
        print("{}/{}".format(batch_idx,total_length))
    print("OK!")


