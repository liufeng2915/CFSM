
import numpy as np  
import torch 
from torch.autograd import Variable


def adversarial_img_augmentation(cfg, synthesis, backbone, module_partial_fc, img, local_labels, opt):

    # Projected Gradient Descent (PGD) is a multi-step multi-step variant Fast Gradient Sign Method (FGSM)
    # when k=1, 
    sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (cfg.batch_size, 10))))
    sampled_z = sampled_z.cuda(non_blocking=True)
    z = sampled_z.detach()
    for it in range(cfg.k):
        z.requires_grad_()
        _,updated_img,_ = synthesis(img, z)
        with torch.enable_grad():
            local_lq_embeddings = backbone(updated_img)
            lq_loss: torch.Tensor = module_partial_fc(local_lq_embeddings, local_labels, opt)
            grad = torch.autograd.grad(lq_loss, [z])[0]
            z = z.detach() + cfg.alpha * torch.sign(grad.detach())
            z = torch.min(torch.max(z, sampled_z - cfg.epsilon), sampled_z + cfg.epsilon)

    _,syn_img,_ = synthesis(img, z)

    ## the ratio of real and synthetic images in a mini-batch
    batch_size = img.shape[0]
    idx1 = torch.randperm(batch_size)
    idx1 = idx1[:int(batch_size*cfg.rs_ratio)] 
    idx2 = torch.randperm(batch_size)
    idx2 = idx2[:int(batch_size*(1-cfg.rs_ratio))] 

    train_img = torch.cat((img[idx1], syn_img[idx2]), dim=0)
    train_labels = torch.cat((local_labels[idx1],local_labels[idx2]),dim=0)

    return train_img, train_labels