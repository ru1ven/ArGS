import os

path = '/mnt/sda1/lxy/HOGS_results/dexycb-hogs_prerigid_centeredbbox/obj_pose_refine_tanh_1020/mesh/mesh_posed_0.4/sdf_mesh/'


def rename_files(folder_path):
    """
    批量重命名文件夹中的 .ply 文件，去除 {} 内数字的前导零
    示例格式：10_20201022_110947_*_{05}_*.ply → ..._5_...
    """

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if not filename.endswith('.ply'):
            continue  # 跳过非 .ply 文件

        if filename.split('_')[4] in ['05', '00']:
            new_name = filename.split('_')[0]+'_'+filename.split('_')[1]+'_'+filename.split('_')[2]+'_'+filename.split('_')[3]+'_'+\
                       str(int(filename.split('_')[4]))+'_'+filename.split('_')[5]

            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # 避免文件名冲突
            if os.path.exists(new_path):
                print(f"跳过 {filename} → 目标文件已存在")
                continue

            #os.rename(old_path, new_path)
            print(f"重命名: {filename} → {new_name}")


if __name__ == '__main__':

    rename_files(path)
    print("操作完成！")