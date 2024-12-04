import torch
import random

class MaskCollator(object):
    def __init__(self, mask_ratio=0.8, num_masks=4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks

    def __call__(self, batch):
        images, voxels, subjects, coords, image_idx = zip(*batch)
        images = torch.stack(images, dim=0)
        voxels = torch.stack(voxels, dim=0)
        subjects = torch.tensor(subjects)
        coords = torch.stack(coords, dim=0)
        image_idx = torch.tensor(image_idx)

        B, _, sequence_length = voxels.shape
        mask_length = sequence_length * self.mask_ratio
        mask_size = int(mask_length / self.num_masks)  # Fixed mask size per region

        mask_indices = []
        for b in range(B):
            selected_indices = []
            for _ in range(self.num_masks):
                random_idx = random.randint(0, sequence_length - 1)
                random_coord = coords[b, random_idx]

                distances = torch.norm(coords[b] - random_coord, dim=1)
                closest_indices = distances.topk(mask_size, largest=False).indices.tolist()
                selected_indices += closest_indices
            mask_indices.append(torch.tensor(selected_indices))
        
        mask_indices = torch.stack(mask_indices, dim=0)

        return images, voxels, subjects, coords, image_idx, mask_indices
