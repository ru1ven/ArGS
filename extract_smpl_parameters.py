import os
import pickle as pkl
import numpy as np

if __name__ == '__main__':
    leftHand_path = "/mnt/sda2/lxy/arctic/unpack/body_models/mano/MANO_LEFT.pkl"
    rightHand_path = "/mnt/sda2/lxy/arctic/unpack/body_models/mano/MANO_RIGHT.pkl"

    data_r = pkl.load(open(rightHand_path, 'rb'), encoding='latin1')
    data_l = pkl.load(open(leftHand_path, 'rb'), encoding='latin1')

    if not os.path.exists('hand_models/misc'):
        os.makedirs('hand_models/misc')

    np.savez('hand_models/misc/J_regressors.npz', rightHand=data_r['J_regressor'].toarray(),
             leftHand=data_l['J_regressor'].toarray())
    np.savez('hand_models/misc/posedirs_all.npz', rightHand=data_r['posedirs'],
             leftHand=data_l['posedirs'])
    np.savez('hand_models/misc/shapedirs_all.npz', rightHand=data_r['shapedirs'],
             leftHand=data_l['shapedirs'])
    np.savez('hand_models/misc/skinning_weights_all.npz', rightHand=data_r['weights'],
             leftHand=data_l['weights'])
    np.savez('hand_models/misc/v_templates.npz', rightHand=data_r['v_template'],
             leftHand=data_l['v_template'])
    #np.save('hand_models/misc/kintree_table.npy', data_r['kintree_table'].astype(np.int32))


    #
    # #np.savez('hand_models/misc/faces.npz', faces=data_r['f'].astype(np.int64))
    # np.savez('hand_models/misc/J_regressors.npz', rightHand=data_r['J_regressor'].toarray())
    # np.savez('hand_models/misc/posedirs_all.npz', rightHand=data_r['posedirs'])
    # np.savez('hand_models/misc/shapedirs_all.npz', rightHand=data_r['shapedirs'])
    # np.savez('hand_models/misc/skinning_weights_all.npz', rightHand=data_r['weights'])
    # np.savez('hand_models/misc/v_templates.npz', rightHand=data_r['v_template'])
    # np.save('hand_models/misc/kintree_table.npy', data_r['kintree_table'].astype(np.int32))

