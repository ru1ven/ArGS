import os
import json
import pickle
from collections import defaultdict

# HO3D 数据集路径
import numpy as np

ho3d_path = '/mnt/sda1/lxy/3DGS/HO3D_v3/train/'

# 用于存储每个序列的subject_id和object_id
sequence_data = []

# 自动生成hand_beta到subject_id的映射
hand_beta_to_subject_id = {}
current_subject_id = 0

# 遍历所有序列文件夹
for sequence_id in os.listdir(ho3d_path):
    sequence_path = os.path.join(ho3d_path, sequence_id, 'model_HOISDF')
    print(sequence_id)

   # for npz in os.listdir(sequence_path):
    model_file_path = os.path.join(sequence_path, '0000.npz')
    meta_file_path = os.path.join(ho3d_path, sequence_id, 'meta', '0000.pkl')
    #model_file_path = os.path.join(sequence_path, npz)
    #meta_file_path = os.path.join(ho3d_path, sequence_id, 'meta', npz.replace('npz','pkl'))

    if os.path.isfile(model_file_path):
        # 读取meta文件
        with open(meta_file_path, 'rb') as f:
            meta_data = pickle.load(f)

        # 提取object_id
        object_id = meta_data['objLabel']

        # 提取hand_beta并推断subject_id
        hand_beta = meta_data['handBeta']
        if hand_beta is None:
            print(meta_file_path)
        hand_beta_tuple = tuple(hand_beta)[0]

        # 如果hand_beta不在映射字典中，添加新的subject_id
        if hand_beta_tuple not in hand_beta_to_subject_id:
            hand_beta_to_subject_id[hand_beta_tuple] = current_subject_id
            current_subject_id += 1
            print(hand_beta_to_subject_id)

        # 获取subject_id
        subject_id = hand_beta_to_subject_id[hand_beta_tuple]



    if os.path.isfile(model_file_path):
        # 读取meta文件
        meta_data = np.load(model_file_path)  #

        # 提取object_id
        object_id = meta_data['obj_label']
        #subject_id = meta_data['subject_id']

        # 存储结果
        sequence_data.append({
            'sequence_id': sequence_id,
            'subject_id': int(subject_id),
            'object_id': int(object_id)
        })

# 统计有多少人和多少物体
subject_ids = set()
object_ids = set()

# 每个人和每个物体对应的序列数量
subject_sequences = defaultdict(list)
object_sequences = defaultdict(int)

for data in sequence_data:
    subject_ids.add(data['subject_id'])
    object_ids.add(data['object_id'])

    subject_sequences[data['subject_id']].append(data['sequence_id'])
    object_sequences[data['object_id']] += 1

    # 原序列路径
    old_sequence_path = os.path.join(ho3d_path, data['sequence_id'])

    # 新的序列名称
    new_sequence_name = f"{data['subject_id']}-{data['object_id']}-{data['sequence_id']}"
    new_sequence_path = os.path.join(ho3d_path, new_sequence_name)
    #print(new_sequence_path)
    # 重命名文件夹
    #os.rename(old_sequence_path, new_sequence_path)

for data in sequence_data:
    print(f"Sequence ID: {data['sequence_id']}, Subject ID: {data['subject_id']}, Object ID: {data['object_id']}")

# 输出统计结果
print(f"总共的人数: {len(subject_ids)}")
print(f"总共的物体数: {len(object_ids)}")

print("\n每个人对应的序列:")
for subject_id, count in subject_sequences.items():
    print(f"Subject ID: {subject_id}, Sequences: {count}")
log_file = open(os.path.join('/mnt/sda1/lxy/3DGS/HO3D_v3/', "seq2sid.txt"), "w+")
for subject_id, count in subject_sequences.items():
    print(f"{subject_id}: {count}",file=log_file)

print("\n每个物体对应的序列数量:")
for object_id, count in object_sequences.items():
    print(f"Object ID: {object_id}, Sequences: {count}")

print(hand_beta_to_subject_id)