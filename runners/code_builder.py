import sys
sys.path.append("..")
from codes import *
from utils.ldpc_gen.Mackay_ldpc import mackay_ldpc_code

def build_code(code_name, L):
    """
    Args:
        code_name (str): 纠错码名称
        L (int): 迭代尺寸

    Returns:
        Hx (ndarray): X 稳定子矩阵
        Hz (ndarray): Z 稳定子矩阵
        Lx (ndarray): X 逻辑操作符矩阵
        Lz (ndarray): Z 逻辑操作符矩阵
        dim (int): 码的维数
        isCSS (bool): 是否为 CSS 码
    """

    if code_name == "Surface":
        surface = SurfaceCode(L)
        return surface.hx, surface.hz, surface.lx, surface.lz, surface.k, True

    elif code_name == "Toric":
        toric = ToricCode(L)
        return toric.hx, toric.hz, toric.lx, toric.lz, toric.k, True
    
    elif code_name == "RotatedSurface" or code_name == "RotatedPlanar":
        rotated_surface = RotatedSurfaceCode(L)
        return rotated_surface.hx, rotated_surface.hz, rotated_surface.lx, rotated_surface.lz, rotated_surface.k, True
    
    elif code_name == "RotatedToric":
        rotated_toric = RotatedToricCode(L)
        return rotated_toric.hx, rotated_toric.hz, rotated_toric.lx, rotated_toric.lz, rotated_toric.k, True

    elif code_name == "XZZX":
        xzzx = XZZXCode(L)
        return xzzx.hx, xzzx.hz, xzzx.lx, xzzx.lz, xzzx.k, False
    
    elif code_name == "XZTGRE":
        xztgre = XZTGRE(L)
        return xztgre.hx, xztgre.hz, xztgre.lx, xztgre.lz, xztgre.k, True
    
    elif code_name == "ZTGRE":
        ztgre = ZTGRE(L)
        return ztgre.hx, ztgre.hz, ztgre.lx, ztgre.lz, ztgre.k, True
    
    elif code_name == "ZTGRE-HP":
        seed = ZTGRE(L)
        ztgre_hp = HGP(seed.hz)
        return ztgre_hp.hx, ztgre_hp.hz, ztgre_hp.lx, ztgre_hp.lz, ztgre_hp.k, True

    elif code_name == "HGP":    
        ldpc = mackay_ldpc_code(int(L*3/4), L, 4, 3)
        hgp = HGP(ldpc)
        return hgp.hx, hgp.hz, hgp.lx, hgp.lz, hgp.k, True
    
    elif code_name == "XYZ3D":
        xyz3d = XYZ3DCode(L)
        return xyz3d.hx, xyz3d.hz, xyz3d.lx, xyz3d.lz, xyz3d.k, False
    
    elif code_name == "XYZ4D":
        xyz4d = XYZ4DCode(L)
        return xyz4d.hx, xyz4d.hz, xyz4d.lx, xyz4d.lz, xyz4d.k, False

    else:
        raise ValueError(f"Unknown code name: {code_name}")
