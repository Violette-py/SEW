import numpy as np
import torch.nn as nn

class SEW_trigger(nn.Module):
    def __init__(self, size=32, a=1.):
        super(SEW_trigger, self).__init__()
        pattern_x, pattern_y = int(size * 0.75), int(size * 0.9375)
        self.mask = np.zeros([size, size, 3])
        self.mask[pattern_x:pattern_y, pattern_x:pattern_y, :] = 1 * a

        # random pixels -- fixed pattern for all poison samples
        np.random.seed(0)
        self.pattern = np.random.rand(size, size, 3)
        self.pattern = np.round(self.pattern * 255.) / 255.
        np.random.seed(None)

        # init standard deviation
        self.scale = 0.1  
        
        self.mask = np.transpose(self.mask, (2, 0, 1))
        self.pattern = np.transpose(self.pattern, (2, 0, 1))
    
    def forward(self, sample, cover=False):
        mask, pattern = self.mask, self.pattern
        if cover:
            # add gaussian noise to trigger pattern for cover samples
            pattern = np.clip(pattern + np.random.normal(loc=0, scale=self.scale, size=pattern.shape), 0, 1)
        sample = sample * (1 - mask) + pattern * mask
        return sample