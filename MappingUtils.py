import torch
from torchvision import transforms
from math import pi as PI


def mapping_quality(reference, tested, image, patch_size):

    p = int(patch_size/2)
    patch_size = p*2
    img_to_tens = transforms.ToTensor()
    padder = transforms.Pad((p, p, p - 1, p - 1), fill=0,
                            padding_mode='constant')
    b = img_to_tens(padder(image)).detach().permute(1, 2, 0)
    b_h, b_w, depth = b.shape
    a_h, a_w = reference.shape[0], reference.shape[1]

    patch = (b.detach().unfold(0, patch_size, 1).unfold(1, patch_size, 1)
              .contiguous().view(-1, depth*((patch_size)**2)))

    ref_v = (reference[:, :, 0] * a_w + reference[:, :, 1]).reshape(-1)
    test_v = (tested[:, :, 0] * a_w + tested[:, :, 1]).reshape(-1)

    ref_p = torch.index_select(patch, 0, ref_v.long())
    test_p = torch.index_select(patch, 0, test_v.long())

    dist_v = torch.norm(test_p - ref_p, dim=1)**2
    return dist_v.contiguous().view(a_h, a_w)


def color_mapping_quality(reference, tested, image, patch_size):
    dist = mapping_quality(reference, tested, image, patch_size)
    dist_norm = dist.unsqueeze(2) / (dist.max()+1e-8)
    img = torch.cat((dist_norm, dist_norm, dist_norm), dim=2).permute(2, 0, 1)
    return transforms.ToPILImage()(img)


def reconstruct(image_b, mapping):
    a_h, a_w = mapping.shape[0], mapping.shape[1]
    img_to_tens = transforms.ToTensor()
    tens_to_img = transforms.ToPILImage()
    b = img_to_tens(image_b)
    b_vector = b.permute(1, 2, 0).contiguous().view(-1, b.shape[0])
    mapping = mapping.contiguous().view(-1, 2)
    mapping = torch.sum(torch.cat(((mapping[:, 0] * b.shape[2]).view(-1, 1),
                                   mapping[:, 1].view(-1, 1)), 1), dim=1)
    rec = torch.index_select(b_vector, 0, mapping)
    rec = rec.contiguous().view(a_h, a_w, b.shape[0]).permute(2, 0, 1)
    return tens_to_img(rec)


def colormap(offsets):
    offsets = offsets.float()
    h = torch.atan2(offsets[:, :, 1], offsets[:, :, 0]).unsqueeze(2)
    h = h/(2*PI)
    s = torch.norm(offsets, dim=2).unsqueeze(2)
    s = s/(torch.max(s)+1e-8)
    v = torch.ones_like(h)
    hsv = torch.cat((h, s, v), dim=2).permute(2, 0, 1)
    return transforms.ToPILImage(mode='HSV')(hsv).convert('RGB')
