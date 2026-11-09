import os
import copy

import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model.diffusion_lib import DiffusionMLP, GaussianDiffusion, VaeHiddenLayer
from model.ddpm import LatentDiffusion

_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class build_CLIFF(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_CLIFF, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        # self.denoiser = LatentDiffusion(cfg=cfg, device='cuda')
        # self.condition_mapping = nn.Linear(768, 512)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.diffusion_classifier = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.diffusion_classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        self.diffusion_bn = nn.BatchNorm1d(self.in_planes_proj)
        self.diffusion_bn.bias.requires_grad_(False)
        self.diffusion_bn.apply(weights_init_kaiming)

        self.num_ddpm_steps = cfg.MODEL.DDPM_STEPS
        self.diffusion_model = DiffusionMLP(512, 512*2, 512, 2, self.num_ddpm_steps,timesteps=self.num_ddpm_steps)
        self.gaussian_diffusion = GaussianDiffusion(timesteps=self.num_ddpm_steps, beta_schedule='cosine')

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.size = self.h_resolution * self.w_resolution
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size,
                                     model_path=cfg.MODEL.CLIP_PATH)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.diffusion_vaehiddenlayer = VaeHiddenLayer(512, 512, self.num_ddpm_steps)
        self.dropout = nn.Dropout(p=0.2)
        self.num_feat = cfg.MODEL.DDPM_FEATS

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        self.logger.info('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None, target_feat=None, layer=None, target=None):
        if get_text == True:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_7, image_features_8, image_features_9, image_features_10, image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0], image_features_last[:, 0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_7, image_features_8, image_features_9, image_features_10, image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        # mu, logvar = self.diffusion_vaehiddenlayer(img_feature)
        mu, logvar = self.diffusion_vaehiddenlayer(img_feature_proj)
        # mu = mu.unsqueeze(1).repeat(1, self.num_feat, 1).reshape(-1, 768)
        mu = mu.unsqueeze(1).repeat(1, self.num_feat, 1).reshape(-1, 512)
        # logvar = logvar.unsqueeze(1).repeat(1, self.num_feat, 1).reshape(-1, 768)
        logvar = logvar.unsqueeze(1).repeat(1, self.num_feat, 1).reshape(-1, 512)
        noise = torch.randn(self.num_feat*x.shape[0], self.num_ddpm_steps, 512).cuda() + self.diffusion_vaehiddenlayer.reparameterize(mu, logvar)
        repeat_img_feature_proj = image_features_last.unsqueeze(1).repeat(1, self.num_feat, 1, 1).reshape(-1, self.size+1, 768) + image_features.unsqueeze(1).repeat(1, self.num_feat, 1, 1).reshape(-1, self.size+1, 768) + image_features_10.unsqueeze(1).repeat(1, self.num_feat, 1, 1).reshape(-1, self.size+1, 768) + image_features_9.unsqueeze(1).repeat(1, self.num_feat, 1, 1).reshape(-1, self.size+1, 768) + image_features_8.unsqueeze(1).repeat(1, self.num_feat, 1, 1).reshape(-1, self.size+1, 768) + image_features_7.unsqueeze(1).repeat(1, self.num_feat, 1, 1).reshape(-1, self.size+1, 768)
        # repeat_img_feature_proj = img_feature.unsqueeze(1).repeat(1, self.num_feat, 1).reshape(-1, 768) + img_feature_last.unsqueeze(1).repeat(1, self.num_feat, 1).reshape(-1, 768)
        # noise = self.diffusion_noise.repeat(feat.shape[0], 1)
        # repeat_img_feature_proj = (img_feature + img_feature_last)
        repeat_img_feature_proj = self.dropout(repeat_img_feature_proj)
        diff_feat, diff_feat_list = self.gaussian_diffusion.sample(self.diffusion_model, repeat_img_feature_proj, clip_denoised='clamp', noise=noise[:, 0])
        # diff_feat, diff_feat_list = self.gaussian_diffusion.sample(self.diffusion_model, repeat_img_feature_proj, clip_denoised='clamp', noise=noise)
        # mean_diff_feat = diff_feat.reshape(-1, self.num_feat, 768).mean(1)
        mean_diff_feat = diff_feat.reshape(-1, self.num_feat, 512).mean(1)
        # mean_diff_feat = diff_feat
        bn_diff_feat = self.diffusion_bn(mean_diff_feat)

        if self.training:
            loss_kl = self.diffusion_vaehiddenlayer.kld_loss(mu, logvar)
            repeat_target_feat = target_feat.unsqueeze(1).repeat(1, self.num_feat, 1).reshape(-1, 512)
            # repeat_target_feat = target_feat.unsqueeze(1).repeat(1, self.num_feat, 1).reshape(-1, 768)
            # repeat_target_feat = target_feat
            loss, preicted_noise_list = self.gaussian_diffusion.ddpm_forward_oversample(self.diffusion_model, repeat_target_feat, repeat_img_feature_proj, noise=noise)
            # classifier_loss = self.gaussian_diffusion.classifier_noise(self.diffusion_model, target_feat, img_feature+img_feature_last, noise[:,0], self.num_feat, target)
            # preicted_noise_list = preicted_noise_list.reshape(-1, self.num_ddp, 768)
            preicted_noise_list = preicted_noise_list.reshape(-1, self.num_ddpm_steps, 512)
            cls_score = self.classifier(feat)
            cls_score3 = self.diffusion_classifier(bn_diff_feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            feature_list = [img_feature_last, img_feature, img_feature_proj, diff_feat]
            for i in range(layer):
                feature_list.append(preicted_noise_list[:, -i-2])

            return [cls_score, cls_score_proj, cls_score3], feature_list, img_feature_proj, loss + loss_kl * 0.3

        else:
            # Inference: the final feature is the average of the generated guided features
            return mean_diff_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'prompt_learner' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size, model_path=''):
    url = clip._MODELS[backbone_name]
    if model_path and os.path.isfile(model_path):
        pass
    else:
        model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

