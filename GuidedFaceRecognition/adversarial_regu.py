
import numpy as np  
import torch 
from torch.autograd import Variable


def adversarial_img_augmentation(cfg, synthesis, backbone, module_partial_fc, img, local_labels, opt):

    # Projected Gradient Descent (PGD) is a multi-step multi-step variant Fast Gradient Sign Method (FGSM)
    # when k=1, 
    sampled_o = Variable(torch.FloatTensor(np.random.normal(0, 1, (cfg.batch_size, 10))))
    sampled_o = sampled_o.cuda(non_blocking=True)
    o = sampled_o.detach()
    for it in range(cfg.k):
        o.requires_grad_()
        _,updated_img,_ = synthesis(img, o)
        with torch.enable_grad():
            local_embeddings = backbone(updated_img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)
            grad = torch.autograd.grad(loss, [o])[0]
            o = o.detach() + cfg.alpha * torch.sign(grad.detach())
            o = torch.min(torch.max(o, sampled_o - cfg.epsilon), sampled_o + cfg.epsilon)

    _,syn_img,_ = synthesis(img, o)

    ## the ratio of real and synthetic images in a mini-batch
    batch_size = img.shape[0]
    idx1 = torch.randperm(batch_size)
    idx1 = idx1[:int(batch_size*cfg.rs_ratio)] 
    idx2 = torch.randperm(batch_size)
    idx2 = idx2[:int(batch_size*(1-cfg.rs_ratio))] 

    train_img = torch.cat((img[idx1], syn_img[idx2]), dim=0)
    train_labels = torch.cat((local_labels[idx1],local_labels[idx2]),dim=0)

    return train_img, train_labels