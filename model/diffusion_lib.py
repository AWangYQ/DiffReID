import ipdb
import numpy as np
import torch
from torch import nn, einsum
from torch.nn import functional as F
from torch.nn import init
import math
from loss.triplet_loss import hard_example_mining
from einops import rearrange, repeat
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=1, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        # project_in = nn.Sequential(
        #     nn.Linear(dim, inner_dim),
        #     nn.GELU()
        # ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # import ipdb; ipdb.set_trace()
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.true_divide(torch.arange(0, d_model, step=2), d_model) * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BottleNeck(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.):
        super().__init__()
        # nn.LeakyReLU(0.2, inplace=True),
        self.hidden_dim = hidden_dim
        self.blcok_in = nn.Sequential(
            nn.LayerNorm(input_dim),
            # Swish(),
            nn.Linear(input_dim, hidden_dim),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
        )
        self.short_cut = nn.Linear(input_dim,output_dim)

        self.blcok_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            # Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            # nn.LeakyReLU()
        )

        self.temb_proj = nn.Sequential(
            # nn.LayerNorm(hidden_dim*2),
            # nn.Linear(hidden_dim*2, hidden_dim),
            # Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU()
        )
        # self.cond_proj = nn.Sequential(
        #     # nn.LayerNorm(hidden_dim * 2),
        #     nn.Linear(hidden_dim*2, hidden_dim),
        #     # Swish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     # nn.LeakyReLU()
        # )
        self.attn = CrossAttention(query_dim=hidden_dim, heads=8, dim_head=64, dropout=0., context_dim=hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)


    def forward(self, x, t, cond, timestep_specific_cond):
        h = self.blcok_in(x)
        h += self.temb_proj(t)
        
        timestep_specific_cond = timestep_specific_cond + h.unsqueeze(1)
        timestep_specific_cond  = self.condition_crossattention(timestep_specific_cond, cond)
        h = timestep_specific_cond.mean(1) + h
        # h = F.leaky_relu(h)
        h = self.blcok_out(h)
        h = h + self.short_cut(x)
        return h, timestep_specific_cond

    def condition_crossattention(self, x, cond):
        x = self.attn(self.norm1(x), context=cond) + x
        return x

class DiffusionMLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, T=100, timesteps=10, queries=16):
        super().__init__()

        self.block1 = BottleNeck(input_dim, hidden_dim, input_dim)
        self.block2 = BottleNeck(input_dim, hidden_dim, input_dim)
        self.block3 = BottleNeck(input_dim, hidden_dim, output_dim)
        # self.block4 = BottleNeck(input_dim, hidden_dim, output_dim)

        # self.temb_proj = nn.ModuleList(nn.Linear(n, k) for n, k in zip([hidden_dim] + h, h + [hidden_dim]))
        self.time_embedding = TimeEmbedding(T, input_dim, hidden_dim)

        self.cond_embedding = nn.Sequential(
            nn.LayerNorm(768),
            # Swish(),
            nn.Linear(768, hidden_dim),
        )

        self.query_embed = nn.Embedding(timesteps*queries, hidden_dim)
        self.timestep = timesteps
        self.queries = queries


    def forward(self, x, t, cond):
        # cond = F.normalize(cond, p=2, dim=-1)
        # temb = self.temb_proj()
        query_cond = self.query_embed.weight.reshape(self.timestep, self.queries, -1)
        timestep_specific_cond = query_cond[t[0]].unsqueeze(0).repeat(x.size(0), 1, 1)
        temb = self.time_embedding(t)
        cond = self.cond_embedding(cond)

        x, timestep_specific_cond = self.block1(x, temb, cond, timestep_specific_cond)
        x, timestep_specific_cond = self.block2(x, temb, cond, timestep_specific_cond)
        x, timestep_specific_cond = self.block3(x, temb, cond, timestep_specific_cond)

        # attn = (attn1[:, 0] + attn2[:, 0] + attn3[:, 0])/3
        return x


# beta schedule
def linear_beta_schedule(timesteps):
    # scale = 100 / timesteps
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# def linear_beta_schedule(timesteps):
#     # scale = 1000 / timesteps
#     # beta_start = scale * 0.0001
#     # beta_end = scale * 0.02
#     return torch.linspace(0.01, 0.2, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion:
    def __init__(
            self,
            timesteps=1000,
            beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        # self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )
        self.timesteps = timesteps
        self.ranking_loss = nn.MarginRankingLoss(margin=1.0)
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        self.loss_weight = snr

        # learnable noise
        # fix initial noise matrix
        # noise = torch.randn(1, 512)
        # self.diffusion_noise = nn.Parameter(noise)

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # predict x_0
    def p_mean_variance(self, model, x_t, cond, t, clip_denoised='clamp'):
        # predict noise using model
        model_out = model(x_t, t, cond)
        # x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        x_recon = model_out
        pred_noise = self.predict_noise_from_start(x_t, t, x_recon)

        if clip_denoised == 'tanh':
            # x_recon = torch.clamp(x_recon, min=-1., max=1.)
            x_recon = x_recon.tanh()
        elif clip_denoised == 'clamp':
            x_recon = torch.clamp(x_recon, min=-1., max=1.)


        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    # @torch.no_grad()
    def p_sample(self, model, x_t, cond, t, clip_denoised='clamp'):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, cond, t,
                                                                 clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    # denoise: reverse diffusion
    # @torch.no_grad()
    def p_sample_loop(self, model, cond, shape, clip_denoised='clamp', noise=None):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        # img = torch.randn(shape, device=device)
        img = noise
        # img = noise.to(device)
        img_list = []
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, cond, torch.full((batch_size,), i, device=device, dtype=torch.long),
                                clip_denoised=clip_denoised)
            img_list.append(img)
        return img, img_list

    # sample new images
    # @torch.no_grad()
    def sample(self, model, cond=None, clip_denoised='clamp', noise=None):
        batch_size = cond.size(0)
        channels = cond.size(1)
        return self.p_sample_loop(model, cond, shape=(batch_size, channels), clip_denoised=clip_denoised, noise=noise)

    # use ddim to sample
    # @torch.no_grad()
    def ddim_sample(
            self,
            model,
            input,
            ddim_timesteps=50,
            ddim_discr_method="uniform",
            ddim_eta=0.0,
            clip_denoised=True):

        batch_size, channel = input.size()

        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # import ipdb; ipdb.set_trace()
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        # device = next(model.parameters()).device
        device = input.device

        sample_img = torch.randn(input.size(), device=device)
        # sample_img = input
        # for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)

            # 2. predict noise using model
            pred_noise = model(sample_img, t, input)

            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
                # pred_x0 = torch.clamp(pred_x0, min=-5., max=5.)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)
            # x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev

        return sample_img

    # compute train losses
    # def train_losses(self, model, x_start, t):
    def ddpm_forward(self, model, x_start, cond):
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device='cuda').long()
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, cond)
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    # pred_x0
    def ddpm_forward_oversample(self, model, x_start, cond, noise):
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device='cuda').long()
        # generate random noise
        # noise = noise.repeat(x_start.size(0), 1)
        # noise = torch.randn_like(x_start)
        predicted_noise_list = []
        # noise_list = []
        loss_list = []
        for i in range(self.timesteps):
            t = i * torch.ones((x_start.size(0),), device='cuda').long()
            # get x_t
            x_noisy = self.q_sample(x_start, t, noise=noise[:, i])
            # x_noisy = self.q_sample(x_start, t, noise)
            predicted_noise = model(x_noisy, t, cond)
            loss = F.mse_loss(x_start, predicted_noise)
            loss_list.append(loss)
            predicted_noise_list.append(predicted_noise)
            # noise_list.append(noise[:, i])
            # noise_list.append(noise)
        # noise_list = torch.cat(noise_list,dim=0)
        loss = torch.stack(loss_list) * self.loss_weight.cuda()
        loss = loss.mean()
        predicted_noise_list = torch.cat(predicted_noise_list,dim=0)
        return loss, predicted_noise_list

    def ddpm_forward_denoise(self, model, x_start, cond):
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device='cuda').long()
        # generate random noise
        noise = torch.randn_like(x_start)
        x_1_mask = t == 0
        x_gt = x_start.new_zeros(x_start.size())
        x_noisy = self.q_sample(x_start, t, noise=noise)
    
        x_gt[~x_1_mask] = self.q_sample(x_start[~x_1_mask], t[~x_1_mask]-1, noise=noise[~x_1_mask])
        x_gt[x_1_mask] = x_start[x_1_mask]
        
        # x_denoised = self.q_sample(x_start, t-1, noise=noise)
        # x_denoised[x_1_mask] = x_start[x_1_mask]
        predicted_mean = model(x_noisy, t, cond)
        loss = F.mse_loss(x_gt.detach(), predicted_mean)
        return loss

    def sample_denoise(self, model, cond=None, clip_denoised='clamp'):
        '''
        cond: num_rois, 512
        '''
        batch_size, channel = cond.size()

        device = next(model.parameters()).device

        x_t = torch.randn(cond.size(), device=device) # noise


        for i in reversed(range(0, self.timesteps)):
            t =  torch.full((batch_size,), i, device=device, dtype=torch.long)
  
            model_mean = model(x_t, t, cond)
            model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

            noise = torch.randn_like(x_t)
            # no noise when t == 0
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
            # compute x_{t-1}
            x_t = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return x_t

class VaeHiddenLayer(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, step=10):
        super().__init__()
        self.h2mu = nn.Linear(input_dim, output_dim)
        self.h2logvar = nn.Linear(input_dim, output_dim)
        self.step = step

    def reparameterize(self, mu, logvar, deterministic=False):
        std = torch.exp(0.5 * logvar)
        std = std.unsqueeze(1).repeat(1, self.step, 1)
        mu = mu.unsqueeze(1).repeat(1, self.step, 1)
        eps = torch.randn_like(std)
        z = mu + (std * eps if not deterministic else 0)
        return z

    def kld_loss(self, mu, logvar):
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                logvar.exp()) / mu.shape[0]
        return KLD

    def forward(self, h):
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        return mu, logvar