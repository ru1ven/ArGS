import pickle

import numpy as np
import torch
#from smplx import MANO
from manopth.manolayer import ManoLayer
from scipy.spatial.transform import Rotation

from right_hand_model.body_models import MANO

def vis_ho3d_mesh_manolayer():
    # dexycb 使用主成分分析
    body_model = ManoLayer(flat_hand_mean=False,
                               ncomps=45,
                               side='right',
                               mano_root='/home/cyc/pycharm/lxy/visual/',
                               use_pca=True) # .cuda()

    faces = np.load("/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/misc/faces.npz")['faces']

    file_name = '/home/cyc/pycharm/data/hand/DexYCB/20201022-subject-10/20201022_110806/836212060125//labels_000060.npz'
    anno = np.load(file_name)
    mano_para = np.array(anno['pose_m']).reshape(-1)
    trans = mano_para[48:].reshape(1, -1)
    pose = mano_para[3:48].reshape(1, -1)
    rot = mano_para[:3].reshape(1, -1)
    rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
    new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1, 3]).astype(np.float32)
    new_trans = trans.reshape([1, 3]).astype(np.float32)


    hand_beta = np.array([-0.8101279, 0.77720827, 1.9707527, 0.35107753, 0.64986867, 2.711966,
                1.1160069, 0.6333117, 0.75000185, 0.4505857])
    # new_root_orient = np.array([-5.0082648e-01, 2.8636034e+00, 1.0859632e+00],dtype=np.float32).reshape(1, -1)
    # pose = np.array([ 1.0412924e-01,
    #             - 1.3629951e-01, 8.0177844e-02, 1.8768187e-01, 0.0000000e+00,
    #             - 5.5683022e-03, 0.0000000e+00, 0.0000000e+00, 1.5664257e-01,
    #             0.0000000e+00, 8.5665613e-02, 9.4764054e-02, 1.9926904e-01,
    #             0.0000000e+00, 1.4938904e-01, 0.0000000e+00, 0.0000000e+00,
    #             2.2401641e-01, 2.0685506e-03, 2.7393934e-01, 3.6708921e-01,
    #             0.0000000e+00, 1.8089482e-03, 7.3866338e-02, 0.0000000e+00,
    #             0.0000000e+00, 1.0983388e-01, 3.9926848e-01, 1.1577919e-01,
    #             2.6811796e-01, 2.1300247e-01, 0.0000000e+00, 1.1372937e-02,
    #             0.0000000e+00, 0.0000000e+00, 1.9638607e-01, 7.3896486e-01,
    #             - 1.7991401e-01, 5.0306249e-01, 1.5101859e-01, 0.0000000e+00,
    #             - 5.9254896e-03, 0.0000000e+00, 2.3616867e-03, 4.7636814e-02],dtype=np.float32).reshape(1, -1)

    new_trans = np.array([-0.17006603, 0.03036311, 0.41327086],dtype=np.float32).reshape(1, -1)

    pose_torch = torch.from_numpy(pose)  # .cuda()
    betas_torch = torch.from_numpy(hand_beta).reshape(1, -1).float()
    new_trans_torch = torch.from_numpy(new_trans)  # .cuda()
    new_root_orient_torch = torch.from_numpy(new_root_orient)

    output = body_model(torch.cat([new_root_orient_torch,pose_torch],dim=-1).view(1, 48).float(),
                                        betas_torch.float(),
                                        new_trans_torch.float())
    verts = (output[0].numpy() / 1000).reshape(778, 3)


    path = '/home/cyc/pycharm/lxy/visual/labels_000020.obj'
    with open(path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    return 0

def vis_ho3d_mesh():
    body_model = MANO(model_path='/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/mano/', use_pca=True,
                      num_pca_comps=48, flat_hand_mean=False)  # .cuda()
    print(1)

    faces = np.load("/home/cyc/pycharm/lxy/gs/3dgs-avatar-release/hand_models/misc/faces.npz")['faces']

    file_name = '/home/cyc/pycharm/data/hand/DexYCB/20201022-subject-10/20201022_110806/836212060125//labels_000060.npz'
    anno = np.load(file_name)
    mano_para = np.array(anno['pose_m']).reshape(-1)
    trans = mano_para[48:].reshape(1, -1)
    pose = mano_para[3:48].reshape(1, -1)
    rot = mano_para[:3].reshape(1, -1)
    rot = Rotation.from_rotvec(np.array(rot).reshape([-1])).as_matrix()
    new_root_orient = Rotation.from_matrix(rot).as_rotvec().reshape([1, 3]).astype(np.float32)
    new_trans = trans.reshape([1, 3]).astype(np.float32)


    hand_beta = np.array([-0.8101279, 0.77720827, 1.9707527, 0.35107753, 0.64986867, 2.711966,
                1.1160069, 0.6333117, 0.75000185, 0.4505857])
    # new_root_orient = np.array([-5.0082648e-01, 2.8636034e+00, 1.0859632e+00],dtype=np.float32).reshape(1, -1)
    # pose = np.array([ 1.0412924e-01,
    #             - 1.3629951e-01, 8.0177844e-02, 1.8768187e-01, 0.0000000e+00,
    #             - 5.5683022e-03, 0.0000000e+00, 0.0000000e+00, 1.5664257e-01,
    #             0.0000000e+00, 8.5665613e-02, 9.4764054e-02, 1.9926904e-01,
    #             0.0000000e+00, 1.4938904e-01, 0.0000000e+00, 0.0000000e+00,
    #             2.2401641e-01, 2.0685506e-03, 2.7393934e-01, 3.6708921e-01,
    #             0.0000000e+00, 1.8089482e-03, 7.3866338e-02, 0.0000000e+00,
    #             0.0000000e+00, 1.0983388e-01, 3.9926848e-01, 1.1577919e-01,
    #             2.6811796e-01, 2.1300247e-01, 0.0000000e+00, 1.1372937e-02,
    #             0.0000000e+00, 0.0000000e+00, 1.9638607e-01, 7.3896486e-01,
    #             - 1.7991401e-01, 5.0306249e-01, 1.5101859e-01, 0.0000000e+00,
    #             - 5.9254896e-03, 0.0000000e+00, 2.3616867e-03, 4.7636814e-02],dtype=np.float32).reshape(1, -1)

    new_trans = np.array([-0.17006603, 0.03036311, 0.41327086],dtype=np.float32).reshape(1, -1)

    pose_torch = torch.from_numpy(pose)  # .cuda()
    betas_torch = torch.from_numpy(hand_beta).reshape(1, -1).float()
    new_trans_torch = torch.from_numpy(new_trans)  # .cuda()
    new_root_orient_torch = torch.from_numpy(new_root_orient)

    body = body_model(global_orient=new_root_orient_torch, hand_pose=pose_torch, betas=betas_torch, transl=new_trans_torch)


    verts = body['v'].numpy()

    path = '/home/cyc/pycharm/lxy/visual/labels_000020.obj'
    with open(path, 'w') as fp:
        for v in verts[0]:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    return 0

if __name__ == "__main__":

    vis_ho3d_mesh()
