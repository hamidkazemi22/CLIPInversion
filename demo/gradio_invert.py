import argparse
import os
import clip
import kornia.augmentation as kaugs
import torch
import torch.nn as nn
import torchvision
from helpers.augmentations import ColorJitter, RepeatBatch, Jitter, TotalVariation
from helpers.utils import Normalization, Scale, freeze_module
from torch.nn.utils import clip_grad_norm_

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='inverting clip!')
parser.add_argument('--num_iters', default=3400, type=int)
parser.add_argument('--save_every', default=200, type=int)
parser.add_argument('--print_every', default=1, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('-p', '--prompt', action='append', type=str, default=[])
parser.add_argument('-e', '--extra_prompts', action='append', type=str, default=[])
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--tv', default=0.005, type=float)
parser.add_argument('--jitter', action='store_true')
parser.add_argument('--color', action='store_true')
parser.add_argument('--img_size', default=64, type=int)
parser.add_argument('--eps', default=2 / 255)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--bri', type=float, default=0.1)
parser.add_argument('--con', type=float, default=0.1)
parser.add_argument('--sat', type=float, default=0.1)
parser.add_argument('--l1', type=float, default=0.)
parser.add_argument('--trial', type=int, default=1)
parser.add_argument('--cg_std', type=float, default=0.)
parser.add_argument('--cg_mean', type=float, default=0.)
parser.add_argument('--model_name', default='ViT-B/16')
parser.add_argument('--prompt_id', type=int, default=0)
parser.add_argument('--add_noise', type=int, default=1)
args = parser.parse_args()
args.prompt = ' '.join(args.prompt)
print(f'prompt: <{args.prompt}>')
print(f'extra prompts are: {args.extra_prompts}')
device = "cuda" if torch.cuda.is_available() else "cpu"
model_names = [args.model_name]
models = []
for model_name in model_names:
    model, preprocess = clip.load(model_name, device)
    model.eval()
    model = model.float()
    model = model.to(device)
    models.append(model)

normalizer = Normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).to(device)


def run(prompt, tv):
    prompts = [prompt]
    args.tv = tv
    text_inputs = torch.cat([clip.tokenize(f"{c}") for c in prompts]).to(device)

    for model in models:
        freeze_module(model)
    image = torch.rand((1, 3, args.img_size, args.img_size)).to(device)
    image.requires_grad_()

    def get_optimizer(image):
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam([image], lr=args.lr)
        else:
            optimizer = torch.optim.LBFGS([image], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

        return optimizer, scheduler

    optimizer, scheduler = get_optimizer(image)

    text_features_map = {}
    for model in models:
        text_feature = model.encode_text(text_inputs)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        text_features_map[model] = text_feature

    save_path = f'images/gradio/{args.prompt}/{args.trial}/{args.lr}_{args.tv}_{args.cg_std}_{args.cg_mean}'
    os.makedirs(save_path, exist_ok=True)

    seq = []
    if args.jitter:
        jitter = Jitter()
        seq.append(jitter)
    seq.append(RepeatBatch(args.batch_size))
    pre_aug = nn.Sequential(*seq)
    aug = kaugs.AugmentationSequential(
        kaugs.ColorJitter(args.bri, args.con, args.sat, 0.1, p=1.0),
        kaugs.RandomAffine(30, [0.1, 0.1], [0.7, 1.2], p=.5, padding_mode='border'),
        same_on_batch=False,
    )
    tv_module = TotalVariation()

    color_jitter = ColorJitter(args.batch_size, True, mean=args.cg_mean, std=args.cg_std)

    def forward(image, model):
        image_input = pre_aug(image)
        image_input = aug(image_input)
        scale = Scale(model.visual.input_resolution)
        image_input = scale(image_input)
        image_input = color_jitter(image_input)
        epsilon = torch.rand_like(image_input) * 0.007
        image_input = image_input + epsilon
        image_input = normalizer(image_input)
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        l2_loss = torch.norm(image_features - text_features_map[model], dim=1)
        loss = torch.mean(l2_loss)
        return loss, l2_loss

    change_scale_schedule = [900, 1800]

    for i in range(args.num_iters):
        max_grad_norm = 1.
        if i in change_scale_schedule:
            new_res = image.shape[2] * 2
            if args.jitter:
                jitter.lim = jitter.lim * 2
            if new_res >= 224:
                new_res = 224
            up_sample = Scale(new_res)
            image = up_sample(image.detach())
            image.requires_grad_(True)
            optimizer, scheduler = get_optimizer(image)
        yield image

        def closure():
            optimizer.zero_grad()
            other_loss = tv_module(image)
            loss = args.tv * other_loss
            image_input = image
            l1_loss = torch.norm(image_input, p=1)
            loss = loss + args.l1 * l1_loss
            for model in models:
                xent_loss, scores = forward(image_input, model)
                loss = loss + xent_loss * (1 / len(models))
            loss.backward()
            clip_grad_norm_([image], max_grad_norm)
            image.data = torch.clip(image.data, 0, 1)
            if i % args.print_every == 0:
                print(f'{i:04d}: loss is {loss:.4f}, xent: {xent_loss:.4f}, tv: {other_loss:.4f}, l1: {l1_loss:.4f}')
            return loss

        optimizer.step(closure)
        if i >= 3400:
            scheduler.step()
