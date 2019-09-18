import torch
from torchvision import transforms


def patch_match(image_a, image_b, patch_size=8, verbose=False):
    if patch_size < 2:
        raise ValueError(f'patch size {patch_size} is too low (< 2).')

    p = int(patch_size / 2)
    patch_size = p*2
    img_to_tens = transforms.ToTensor()
    padder = transforms.Pad((p, p, p - 1, p - 1), fill=0,
                            padding_mode='constant')
    depth = 3

    a = img_to_tens(padder(image_a.convert('RGB'))).detach().permute(1, 2, 0)
    b = img_to_tens(padder(image_b.convert('RGB'))).detach().permute(1, 2, 0)

    a_w, a_h = image_a.size[0], image_a.size[1]
    b_w = image_b.size[0]

    patch_a = (a.detach().unfold(0, patch_size, 1).unfold(1, patch_size, 1)
                .contiguous().view(-1, depth*((patch_size)**2)))
    patch_b = (b.detach().unfold(0, patch_size, 1).unfold(1, patch_size, 1)
                .contiguous().view(-1, depth*((patch_size)**2)))

    norm_a = torch.norm(patch_a, dim=1)
    norm_b = torch.norm(patch_b, dim=1)

    mapping = torch.empty(patch_a.shape[0], 2, dtype=torch.int32)
    for x in range(patch_a.shape[0]):
        if verbose:
            print(f"Comparing patch {x+1} of {patch_a.shape[0]}...", end="",
                  flush=True)
        dist = (norm_a[x] ** 2 + norm_b ** 2
                - 2 * torch.mm(patch_a[x].view(1, -1),
                               torch.transpose(patch_b, 0, 1)))
        dist_min = torch.argmin(dist)
        mapping[x] = torch.tensor([int(dist_min / b_w), dist_min % b_w])
        if verbose:
            print(" done.")

    a_index = torch.arange(patch_a.shape[0], dtype=torch.int32)
    offsets = mapping - torch.cat((a_index.unsqueeze(1) / a_w,
                                   a_index.unsqueeze(1) % a_w), dim=1)
    mapping = mapping.view(a_h, a_w, 2)
    offsets = offsets.contiguous().view(a_h, a_w, 2)

    return offsets, mapping
