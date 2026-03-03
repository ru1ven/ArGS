import os
import zipfile

# ✅ 需要打包的路径列表（每个路径都会被单独打包成一个 zip）
tag = '3dgs-avatar'
folders = [
    "/mnt/sda2/lxy/ARGS_results/box/arctic_box-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/mixer/arctic_mixer-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/waffleiron/arctic_waffleiron-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/phone/arctic_phone-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/ketchup/arctic_ketchup-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/espressomachine/arctic_espressomachine-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/microwave/arctic_microwave-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/scissors/arctic_scissors-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/laptop/arctic_laptop-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/capsulemachine/arctic_capsulemachine-{}/vis/".format(tag),
    "/mnt/sda2/lxy/ARGS_results/notebook/arctic_notebook-{}/vis/".format(tag),
]

# 统一输出目录
output_dir = "/mnt/sda2/lxy/ARGS_results/zipped/"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 开始打包
# -----------------------------
for folder in folders:
    folder = folder.rstrip("/")  # 去掉末尾斜杠
    # 生成 zip 文件名（例如 arctic_waffleiron-vis_ours.zip）
    zip_name = os.path.basename(os.path.dirname(folder)) + ".zip"
    zip_path = os.path.join(output_dir, zip_name)

    # 打包 PNG 文件
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".png"):
                    abs_path = os.path.join(root, file)
                    # 使 zip 内部路径相对 vis 目录
                    rel_path = os.path.relpath(abs_path, start=folder)
                    zipf.write(abs_path, rel_path)

    print(f"✅ 打包完成: {zip_path}")

print("\n🎯 所有文件已打包到:", output_dir)
