from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarRenderedDistanceProjection(nn.Module):
    """Store polar prototypes, render poses to Cartesian, then match there."""

    def __init__(self, out_channels=192, bases=48, directions=16, kernel_size=24,
                 radial_bins=12, angular_bins=64, stride=16, prototype_std=.02):
        super().__init__()
        if angular_bins % directions: raise ValueError("angular_bins must divide directions")
        self.out_channels=out_channels; self.bases=bases; self.directions=directions
        self.kernel_size=kernel_size; self.radial_bins=radial_bins; self.angular_bins=angular_bins
        self.stride=stride; self.input_padding=(kernel_size-stride)//2
        self.prototype=nn.Parameter(torch.randn(bases,3,radial_bins,angular_bins)*prototype_std)
        self.log2_scale=nn.Parameter(torch.zeros(bases))
        self.value=nn.Parameter(torch.empty(bases,directions,out_channels)); nn.init.trunc_normal_(self.value,std=.02)

        yy,xx=torch.meshgrid(torch.arange(kernel_size),torch.arange(kernel_size),indexing="ij")
        center=(kernel_size-1)/2; dx=xx-center; dy=yy-center
        radius=torch.sqrt(dx.square()+dy.square()); angle=torch.remainder(torch.atan2(dy,dx),2*math.pi)
        cover=torch.cos(radius*math.pi/kernel_size).clamp_min(0); self.register_buffer("cover",cover)
        # Polar source is circular-padded to width A+2. Original angular index
        # j lives at padded index j+1. Radial slot 0 represents rho=.5.
        angular_index=angle/(2*math.pi)*angular_bins
        gx=2*(angular_index+1.5)/(angular_bins+2)-1
        gy=2*radius/radial_bins-1
        support=radius<kernel_size/2
        self.register_buffer("support_mask",support.flatten())
        self.register_buffer("support_cover",cover.flatten()[support.flatten()])
        self.register_buffer("cartesian_grid",torch.stack((gx[support],gy[support]),-1))
        self.register_buffer("direction_shift",torch.arange(directions)*(angular_bins//directions))
        n_in=float(3*cover.sum()); self.multi=1.0/((n_in/6.0)-.5*(n_in*7.0/180.0)**.5)

    def rendered_prototypes(self):
        poses=torch.stack([torch.roll(self.prototype,shifts=int(s),dims=-1) for s in self.direction_shift],1)
        poses=poses.reshape(self.bases*self.directions,3,self.radial_bins,self.angular_bins)
        poses=F.pad(poses,(1,1,0,0),mode="circular")
        grid=self.cartesian_grid[None,:,None].expand(poses.shape[0],-1,-1,-1)
        rendered=F.grid_sample(poses,grid,mode="bilinear",padding_mode="zeros",align_corners=False)
        return rendered[...,0].view(self.bases,self.directions,3,-1)

    def indexed_patches(self,image):
        if self.input_padding:image=F.pad(image,(self.input_padding,)*4,mode="reflect")
        square=F.unfold(image,self.kernel_size,stride=self.stride)
        batch,_,tokens=square.shape
        square=square.view(batch,3,self.kernel_size*self.kernel_size,tokens)
        return square[:,:,self.support_mask].permute(0,3,1,2).contiguous()

    def pose_scores(self,image):
        with torch.autocast(device_type=image.device.type,enabled=False):
            image=image.float()
            rendered=self.rendered_prototypes().float()
            patch=self.indexed_patches(image)
            weight=self.support_cover[None,None,None,:]
            cross=torch.einsum("qtcm,ndcm->qtnd",patch*weight,rendered)
            patch_energy=(patch.square()*weight).sum((2,3))
            proto_energy=(rendered.square()*self.support_cover[None,None,None,:]).sum((2,3))
            distance=(patch_energy[:,:,None,None]+proto_energy[None,None]-2*cross).clamp_min(0)
            side=int(patch.shape[1]**.5)
            distance=distance.permute(0,2,3,1).reshape(image.shape[0],self.bases,self.directions,side,side)
            return -distance*torch.exp2(self.log2_scale)[None,:,None,None,None]*self.multi

    def forward(self,image):
        scores=self.pose_scores(image); direction_weight=scores.softmax(2)
        base_value=torch.einsum("qbrhw,brc->qbhwc",direction_weight,self.value)
        amplitude=scores.amax(2)-scores.mean(2)
        return (base_value*amplitude[...,None]).sum(1).permute(0,3,1,2).contiguous()
