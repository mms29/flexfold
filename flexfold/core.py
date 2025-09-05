from scipy.spatial.transform import Rotation
from cryodrgn import fft
from cryodrgn.mrcfile import write_mrc, parse_mrc

import torch
import numpy as np

import tqdm

from openfold.np import residue_constants, protein
from openfold.utils.tensor_utils import tensor_tree_map

def output_single_pdb(all_atom_positions, aatype, all_atom_mask, file):

    chain_index = np.zeros_like(aatype)
    b_factors = np.zeros_like(all_atom_mask)
    residue_index = np.arange(len(aatype))+1

    pdb_elem = protein.Protein(
        aatype=aatype,
        atom_positions=all_atom_positions,
        atom_mask=all_atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index,
        remark="",
        parents=None,
        parents_chain_index=None,
    )
    outstring = protein.to_pdb(pdb_elem)
    with open(file, 'w') as fp:
        fp.write(outstring)


def struct_to_pdb(struct, file):
    struct = tensor_tree_map(lambda x: x.detach().cpu().numpy, struct)

    aatype=struct["aatype"],
    atom_positions=struct["final_atom_positions"],
    atom_mask=struct["final_atom_mask"],
    residue_index=struct["residue_index"] if "residue_index" in struct else   np.arange(len(aatype))+1,
    b_factors=struct["plddt"] if "plddt" in struct else np.zeros_like(atom_mask),
    chain_index=struct["chain_index"] if "chain_index" in struct else  np.zeros_like(aatype),

    pdb_elem = protein.Protein(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=struct["chain_index"] if "chain_index" in struct else  np.zeros_like(struct["aatype"]),
        remark="",
        parents=None,
        parents_chain_index=None,
    )
    outstring = protein.to_pdb(pdb_elem)
    with open(file, 'w') as fp:
        fp.write(outstring)



def fourier_corr(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # normalized complex correlation in Fourier domain 
    num = torch.sum(A * torch.conj(B), dim=(-1,-2))
    denom = torch.sqrt(torch.sum(torch.abs(A) ** 2, dim=(-1,-2)) * torch.sum(torch.abs(B) ** 2, dim=(-1,-2)))

    return (num / denom).real  # Return real part only


def get_cc(im1, im2):
    """
    im1 : [B, N, N]
    im2 : [B, N, N]
    output : [B]
    """
    
    return torch.sum( im1 * im2, dim=(-1,-2)) / (
        torch.sqrt(
            torch.sum(
                torch.square(im1), dim=(-1,-2)
            ) *
            torch.sum(
                torch.square(im2), dim=(-1,-2)
            )
        )
    )

def euler2matrix(angles):
    return Rotation.from_euler("zyz", np.array(angles), degrees=True).as_matrix()

def matrix2euler(A):
    return Rotation.from_matrix(A).as_euler("zyz", degrees=True)

def get_sphere(angular_dist):
    num_pts = int(np.pi * 10000 * 1 / (angular_dist ** 2))
    angles = np.zeros((num_pts, 3))
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = (np.arccos(1 - 2.0 * indices / num_pts))
    theta = (np.pi * (1 + 5 ** 0.5) * indices)
    for i in range(num_pts):
        R = Rotation.from_euler("yz", np.array([phi[i], theta[i]]), degrees=False).as_matrix()
        angles[i] = matrix2euler(R)
    return angles

def get_sphere_full(angular_dist):
    n_zviews = 360 // angular_dist
    angles = get_sphere(angular_dist)
    num_pts = len(angles)
    new_angles = np.zeros((num_pts * n_zviews, 3))
    for i in range(num_pts):
        for j in range(n_zviews):
            new_angles[i*n_zviews+ j, 0] =j * angular_dist
            new_angles[i*n_zviews+ j, 1] =angles[i,1]
            new_angles[i*n_zviews+ j, 2] =angles[i,2]
    return new_angles

def get_angular_distance(a1, a2):
    R1 = euler2matrix(a1)
    R2 = euler2matrix(a2)
    R = np.dot(R1, R2.T)
    cosTheta = (np.trace(R) - 1) / 2
    cosTheta = np.clip(cosTheta, -1.0, 1.0) 
    return    np.rad2deg(np.arccos(cosTheta))

def get_corr_ft(v1, v2):
    corr = fft.ifftn_center(v1 * v2.conj()).real
    flat_index = corr.argmax()
    index = np.unravel_index(flat_index.item(), v1.shape)
    return corr.max(), index


def register_crd_to_vol(vol, crd, grid_size, sigma, pixel_size, dist_search = 15, real_space=True, quality_ratio=5.0):
    angles = get_sphere_full(int(dist_search))
    cc=torch.zeros(len(angles), dtype=crd.dtype, device=crd.device)
    shifts=torch.zeros((len(angles),3), dtype=crd.dtype, device=crd.device)

    n_pix_cutoff=int(np.ceil(quality_ratio * sigma / pixel_size) * 2 + 1)    

    for i,a in tqdm.tqdm(enumerate(angles), "Angular search", len(angles)) :
        R = euler2matrix(a)
        c_search = crd @ torch.tensor(R, device=crd.device, dtype=crd.dtype)

        if real_space:
            vox_loc, vox_mask = get_voxel_mask(c_search, grid_size, pixel_size,  n_pix_cutoff)
            vol_search_real = vol_real_mask(
                c_search[None], 
                vox_loc[None], 
                vox_mask[None], 
                grid_size, 
                sigma, 
                pixel_size
            )[-1]
            vol_search = fft.fftshift(fft.fftn(fft.fftshift(vol_search_real), dim=(-1,-2,-3)))
        else:
            vol_search = vol_ft(c_search, grid_size, sigma, pixel_size)
        corr, shift = get_corr_ft(vol_search, vol)
        shifts[i] = torch.tensor(shift, device=crd.device, dtype=crd.dtype)
        cc[i] = corr

    angles_final = angles[cc.argmax()]
    shift_final = (shifts[cc.argmax()] - grid_size/2 + .5)* pixel_size
    R_final = torch.tensor(euler2matrix(angles_final), device=crd.device, dtype=crd.dtype)

    return angles_final, R_final, shift_final



def lattice_ft_2D(device, grid_size = 128, sigma = 1.0, pixel_size=1.0):
    freqs = torch.fft.fftfreq(grid_size, d=pixel_size, device=device)
    u, v = torch.meshgrid(freqs, freqs)
    u = torch.fft.fftshift(u).reshape(grid_size**2)
    v = torch.fft.fftshift(v).reshape(grid_size**2)
    return torch.stack((u,v)).T

def lattice_ft_3D(device, grid_size = 128, sigma = 1.0, pixel_size=1.0):
    freqs = torch.fft.fftfreq(grid_size, d=pixel_size, device=device)
    u, v, w = torch.meshgrid(freqs, freqs, freqs)
    u = torch.fft.fftshift(u).reshape(grid_size**3)
    v = torch.fft.fftshift(v).reshape(grid_size**3)
    w = torch.fft.fftshift(w).reshape(grid_size**3)
    return torch.stack((u,v, w)).T


def img_ft_lattice(crd, crd_lattice, sigma = 1.0, pixel_size=1.0, crd_mask=None):
    # crd 
    #   [batch_dim, N_atoms, 3]
    # crd_mask  
    #   [batch_dim, N_atoms]
    # crd_lattice  
    #   [lattice_size, 3]
    # -> Output  
    #   [batch_dim, crd_lattice]

    batch_dim = crd.shape[:-2]
    lattice_size = crd_lattice.shape[-2]

    crd_lattice /= pixel_size

    gaussian_envelope = torch.exp(-2 * (torch.pi**2) * sigma**2 * torch.sum(crd_lattice**2, dim=-1))

    F = torch.exp(-2j * torch.pi * torch.sum(crd[...,None, :] * crd_lattice[..., None, :,:], dim=-1))
    if crd_mask is not None:
        F = torch.sum(crd_mask[..., None] *  F, dim=-2)
    else:
        F = torch.sum(F, dim=-2)

    F /= 2* torch.pi

    return (gaussian_envelope[None] * F).reshape(batch_dim+(lattice_size, ))

def img_ht_lattice(crd, crd_lattice, sigma = 1.0, pixel_size=1.0):
    I = img_ft_lattice(crd, crd_lattice, sigma, pixel_size)
    return  I.real- I.imag

def get_circle(radius,):
    s = radius
    x = torch.arange(s)
    y = torch.arange(s)
    xx, yy = torch.meshgrid(x, y)
    d = (xx - s // 2) ** 2 + (yy - s // 2) ** 2
    mask = d <= (s / 2) ** 2
    circle = torch.stack((xx[mask], yy[mask]), dim=-1)
    return circle


def get_circle_3D(radius):
    s = radius
    x = torch.arange(s)
    y = torch.arange(s)
    z = torch.arange(s)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')  
    center = s // 2
    d = (xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2
    mask = d <= (s / 2) ** 2

    sphere = torch.stack((xx[mask], yy[mask], zz[mask]), dim=-1)
    return sphere


def get_pixel_mask(coord, grid_size, pixel_size, n_pix_cutoff):
    threshold = (n_pix_cutoff - 1) // 2
    circle = get_circle(n_pix_cutoff)  # shape (n_pix, 2)
    circle= circle.to(coord.device)
    
    # Compute base pixel positions for all atoms (broadcasting)
    base_pix = torch.floor(coord[..., :2] / pixel_size - threshold + grid_size / 2)  # shape (n_atoms, 2)
    
    # Add circle offsets to each base pixel position
    pix = base_pix[..., None, :] + circle[None, :, :]  # shape (n_atoms, n_pix, 2)

    # Validity mask: check if x and y are in bounds
    valid_x = (pix[..., 0] >= 0) & (pix[..., 0] < grid_size)
    valid_y = (pix[..., 1] >= 0) & (pix[..., 1] < grid_size)
    valid_mask = valid_x & valid_y

    pix[valid_mask==0.0] = 0.0

    return pix, valid_mask


def get_voxel_mask(coord, grid_size, pixel_size, n_pix_cutoff):
    threshold = (n_pix_cutoff - 1) // 2
    circle = get_circle_3D(n_pix_cutoff)  # shape (n_pix, 3)
    circle= circle.to(coord.device)
    
    # Compute base pixel positions for all atoms (broadcasting)
    base_pix = torch.floor(coord / pixel_size - threshold + grid_size / 2)  # shape (n_atoms, 3)
    
    # Add circle offsets to each base pixel position
    pix = base_pix[..., None, :] + circle[None, :, :]  # shape (n_atoms, n_pix, 2)

    # Validity mask: check if x and y are in bounds
    valid_x = (pix[..., 0] >= 0) & (pix[..., 0] < grid_size)
    valid_y = (pix[..., 1] >= 0) & (pix[..., 1] < grid_size)
    valid_z = (pix[..., 2] >= 0) & (pix[..., 2] < grid_size)
    valid_mask = valid_x & valid_y & valid_z

    pix[valid_mask==0.0] = 0.0

    return pix, valid_mask


def img_real(crd, grid_size = 128, sigma = 1.0, pixel_size=1.0):
    # crd 
    #   [batch_dim, N_atoms, 3]
    # crd_mask  
    #   [batch_dim, N_atoms]

    batch_dim = crd.shape[:-2]

    xx, yy = torch.meshgrid(
        torch.arange(grid_size, device=crd.device), torch.arange(grid_size, device=crd.device),
    )
    xx = (xx.reshape(grid_size**2) - grid_size/2 +.5) * pixel_size
    yy = (yy.reshape(grid_size**2) - grid_size/2 +.5) * pixel_size

    I = torch.exp(-1 / (2 * sigma**2)  * ( 
        (xx - crd[..., 1, None])**2 + \
        (yy - crd[..., 0, None])**2 ) \
    )
    I = torch.sum(I, dim=-2)
    I /= (2 * torch.pi * sigma**2)
    return  I.reshape(batch_dim+(grid_size, grid_size))


def img_real_mask(crd, pix_loc, pix_mask, grid_size=128, sigma=1.0, pixel_size=1.0):
    B, N_atoms, P = pix_mask.shape

    # Compute (x, y) positions in real space
    xy = (pix_loc - grid_size / 2 + 0.5) * pixel_size  # (B, N_atoms, P, 2)
    crd_xy = crd[:, :, None, :2]                       # (B, N_atoms, 1, 2)

    # Compute Gaussian contributions
    sq_dist = torch.sum((xy - crd_xy) ** 2, dim=-1)    # (B, N_atoms, P)
    I = torch.exp(-0.5 * sq_dist / sigma**2) * pix_mask  # (B, N_atoms, P)

    # Flatten for scatter
    flat_I = I.reshape(B, -1)                                # (B, N_atoms * P)
    flat_idx = pix_loc.reshape(B, -1, 2).long()              # (B, N_atoms * P, 2)

    # Build linear indices for scatter_add
    x = flat_idx[..., 0]
    y = flat_idx[..., 1]
    idx = x * grid_size + y                                  # flatten 2D index to 1D

    # Scatter into flat image buffer
    I_out = torch.zeros(B, grid_size * grid_size, device=crd.device, dtype=I.dtype)
    I_out = I_out.scatter_add(1, idx, flat_I)

    # Reshape back to 2D grid
    I_out = I_out.view(B, grid_size, grid_size)

    # Normalize Gaussian
    I_out /= (2 * torch.pi * sigma**2)

    return I_out.transpose(-2,-1)

def vol_real_mask(crd, pix_loc, pix_mask, grid_size=128, sigma=1.0, pixel_size=1.0, amplitude=None):
    B, N_atoms, P = pix_mask.shape

    # Compute (x, y) positions in real space
    xyz = (pix_loc - grid_size / 2 + 0.5) * pixel_size  # (B, N_atoms, P, 3)
    crd_xyz = crd[:, :, None, : ]                       # (B, N_atoms, 1, 3)

    # Compute Gaussian contributions
    sq_dist = torch.sum((xyz - crd_xyz) ** 2, dim=-1)    # (B, N_atoms, P)
    I = torch.exp(-0.5 * sq_dist / sigma**2) * pix_mask  # (B, N_atoms, P)

    if amplitude is not None : 
        I *= amplitude[..., None]

    # Flatten for scatter
    flat_I = I.reshape(B, -1)                                # (B, N_atoms * P)
    flat_idx = pix_loc.reshape(B, -1, 3).long()              # (B, N_atoms * P, 2)

    # Build linear indices for scatter_add
    x = flat_idx[..., 0]
    y = flat_idx[..., 1]
    z = flat_idx[..., 2]
    idx = x * (grid_size**2) + y * grid_size + z             # flatten 2D index to 1D

    # Scatter into flat vol buffer
    I_out = torch.zeros(B, grid_size **3, device=crd.device, dtype=I.dtype)
    I_out = I_out.scatter_add(1, idx, flat_I)

    # Reshape back to 3D grid
    I_out = I_out.view(B, grid_size, grid_size,grid_size)

    # Normalize Gaussian
    I_out /= (2 * torch.pi * sigma**2)

    return torch.permute(I_out, (0,3,2,1))



def vol_ft(crd, grid_size = 128, sigma = 1.0, pixel_size=1.0, crd_mask=None):
    # crd 
    #   [batch_dim, N_atoms, 3]
    # crd_mask  
    #   [batch_dim, N_atoms]

    batch_dim = crd.shape[:-2]

    freqs = torch.fft.fftfreq(grid_size, d=pixel_size, device=crd.device)
    u, v, w = torch.meshgrid(freqs, freqs, freqs)
    u = torch.fft.fftshift(u).reshape(grid_size**3)
    v = torch.fft.fftshift(v).reshape(grid_size**3)
    w = torch.fft.fftshift(w).reshape(grid_size**3)

    gaussian_envelope = torch.exp(-2 * (torch.pi**2) * sigma**2 * (u**2 + v**2 + w**2))

    F = torch.exp(-2j * torch.pi * (
        crd[..., 2, None] * u + \
        crd[..., 1, None] * v + \
        crd[..., 0, None] * w )
    )

    if crd_mask is not None:
        F = torch.sum(crd_mask[..., None] *  F, dim=-2)
    else:
        F = torch.sum(F, dim=-2)

    return (gaussian_envelope[None] * F).reshape(batch_dim + (grid_size, grid_size, grid_size))



def vol_real(crd, grid_size = 128, sigma = 1.0, pixel_size=1.0):
    # crd 
    #   [batch_dim, N_atoms, 3]
    # crd_mask  
    #   [batch_dim, N_atoms]

    batch_dim = crd.shape[:-2]

    grid = torch.arange(grid_size, device=crd.device)
    xx, yy, zz = torch.meshgrid(grid,grid,grid)
    xx = (xx.reshape(grid_size**3) - grid_size/2 +.5) * pixel_size
    yy = (yy.reshape(grid_size**3) - grid_size/2 +.5) * pixel_size
    zz = (zz.reshape(grid_size**3) - grid_size/2 +.5) * pixel_size

    I = torch.exp(-1 / (2 * sigma**2)  * ( 
        (xx - crd[..., 2, None])**2 + \
        (yy - crd[..., 1, None])**2 + \
        (zz - crd[..., 0, None])**2 ) 
    )
    I = torch.sum(I, dim=-2)
    I /= (2 * torch.pi * sigma**2)
    return  I.reshape(batch_dim + (grid_size, grid_size, grid_size))

