import torch
import math

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.000001 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits

class AdaFace(torch.nn.Module):
    def __init__(self,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=0.01,
                 ):
        super(AdaFace, self).__init__()

        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, cosine: torch.Tensor, label, norms=None):

        #print(cosine.shape, label.shape, norms.shape)
        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():

            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()

            if self.batch_mean.device != mean.device:
                self.batch_mean = self.batch_mean.to(mean.device)
                self.batch_std = self.batch_std.to(std.device)
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        index = torch.where(label != -1)[0]

        # g_angular
        m_arc = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label[index, None], 1.0)
        g_angular = self.m * margin_scaler[index] * -1
        m_arc = m_arc * g_angular
        cosine.acos_()
        cosine[index] += m_arc
        cosine.cos_()

        # g_additive
        m_cos = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label[index, None], 1.0)
        g_add = self.m + (self.m * margin_scaler[index])
        m_cos = m_cos * g_add
        cosine[index] -= m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
    

class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits
