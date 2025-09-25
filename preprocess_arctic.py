import os
import shutil

import numpy as np



source_dir = "/mnt/sda2/lxy/arctic/unpack/arctic_data/data/splits"
hands25_seqs = ['s03/box_grab_01_1',
                's03/capsulemachine_grab_01_1',
                's03/espressomachine_grab_01_1',
                's03/ketchup_grab_01_1',
                's03/laptop_grab_01_1',
                's03/microwave_grab_01_1',
                's03/mixer_grab_01_1',
                's03/notebook_grab_01_1',
                's03/waffleiron_grab_01_1',
                's05/espressomachine_grab_01_8']

R9_seqs = ['s01/box_grab_01_1',
                's01/capsulemachine_grab_01_1',
                's01/espressomachine_grab_01_1',
                's01/ketchup_grab_01_1',
                's01/laptop_grab_01_1',
                's01/microwave_grab_01_1',
                's01/mixer_grab_01_1',
                #'s01/notebook_grab_01_1',
                's01/waffleiron_grab_01_1',
                ]

NR9_seqs = ['s01/box_use_01_1',
                's01/capsulemachine_use_01_1',
                's01/espressomachine_use_01_1',
                's01/ketchup_use_01_1',
                's01/laptop_use_01_1',
                's01/microwave_use_01_1',
                's01/mixer_use_01_1',
                #'s01/notebook_grab_01_1',
                's01/waffleiron_use_01_1',
                ]

NR9_box = ['s01/box_use_01_1',
                ]
NR9_ketchup = ['s01/ketchup_use_01_1',
                ]
#
# NR9_seqs = ['s01/box_use_01_1',
#                 's01/capsulemachine_use_01_1',
#                 's01/espressomachine_use_01_1',
#                 's01/ketchup_use_01_1',
#                 's01/laptop_use_01_1',
#                 's01/microwave_use_01_1',
#                 's01/mixer_use_01_1',

#                 #'s01/notebook_grab_01_1',
#                 's01/waffleiron_use_01_1',]

NR_seqs_train = ['s01/box_use_02_1',
            's01/capsulemachine_use_01_1','s01/waffleiron_use_01_1',
            's01/phone_use_01_1','s01/notebook_use_01_1','s01/mixer_use_01_1','s01/laptop_use_01_1',
            's01/scissors_use_01_1','s01/ketchup_use_02_1',
                's01/espressomachine_use_01_3',
                 's01/microwave_use_01_6',]

NR_seqs_test = [ 's01/box_use_02_8',
            's01/capsulemachine_use_01_8','s01/waffleiron_use_01_8',
            's01/phone_use_01_8','s01/notebook_use_01_8','s01/mixer_use_01_8','s01/laptop_use_01_8',
            's01/scissors_use_01_8','s01/ketchup_use_02_8',
            's01/espressomachine_use_01_1',
            's01/microwave_use_01_7',

           ]





if __name__ == '__main__':
    def load_npy(file_path):
        data = np.load(file_path, allow_pickle=True).item()
        return data['data_dict'], data['imgnames']


    # 读取三个 split
    data_dict_train, imgnames_train = load_npy(os.path.join(source_dir, "p1_train.npy"))
    #data_dict_train, imgnames_train = load_npy(os.path.join(source_dir, "p1_val.npy"))
    #data_dict_train, imgnames_train = load_npy(os.path.join(source_dir, "p1_test.npy"))

    # 合并 data_dict （注意 key 是否有重复，若有需要处理）
    data_para = {}
    for d in [data_dict_train]:
        # , data_dict_val, data_dict_test]:
        data_para.update(d)

    # 合并 imgnames
    image_names = imgnames_train
                  # + imgnames_val + imgnames_test


    if data_para is not None:
        print(f"'data_dict' 类型: {type(data_para)}, 长度: {len(data_para)}")
    if image_names is not None:
        print(f"'imgnames' 类型: {type(image_names)}, 长度: {len(image_names)}")
    print("示例序列名:", list(data_para.keys())[0])
    print("序列:", data_para[list(data_para.keys())[0]].keys())

    print("示例name名:", image_names[0])
    for seq_name  in NR_seqs_train:
        target_set = set(seq_name)

        filtered_imgnames = []
        for path in image_names:
            parts = path.split(os.sep)
            seq = os.path.join(parts[-4], parts[-3])
            view = parts[-2]
            seq_view = f"{seq}_{view}"

            if seq_view == seq_name:
                filtered_imgnames.append(path)
                imgname = path.replace('./arctic_data/data/images/', '/mnt/sda2/lxy/arctic/unpack/arctic_data/data/cropped_images/')
                out_path = path.replace('./arctic_data/data/images/',
                                       '/mnt/sda2/lxy/dataset/hand/arctic_seqs/images/')
                if int(path.split('/')[-1].split('.')[0]) % 2 == 0:
                    print(out_path)
                    # 确保目标目录存在
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    # 复制 imgname 到 out_path
                    shutil.copy2(imgname, out_path)
                    print(f"Copied {imgname} -> {out_path}")


        # 过滤 data_dict，只保留相关序列
        filtered_data_dict = {}

        seq, view = seq_name.rsplit('_', 1)
        if seq in data_para:
            filtered_data_dict[seq] = data_para[seq]  # 保留整个序列

        # 构造新的字典结构
        filtered_data = {
            'data_dict': filtered_data_dict,
            'imgnames': filtered_imgnames
        }

        # 保存为 npy
        np.save('/mnt/sda2/lxy/dataset/hand/arctic_seqs/splits/train/{}'.format(seq_name.replace('/', '_')), filtered_data)
        print(filtered_data_dict.keys())

        print(f"过滤后序列数: {len(filtered_data_dict)}")
        print(f"过滤后帧数: {len(filtered_imgnames)}")