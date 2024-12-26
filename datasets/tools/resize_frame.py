import os
from PIL import Image
from tqdm import tqdm
def resize_dataset(input_folder, output_folder, target_size=320):
# 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹下的所有子文件夹
    for subdir, dirs, files in os.walk(input_folder):
        # 获取相对路径
        rel_path = os.path.relpath(subdir, input_folder)
        # 创建输出子文件夹
        output_subdir = os.path.join(output_folder, rel_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        # 遍历当前子文件夹下的所有图像文件
        for file in tqdm(files):
            # 判断文件是否为图像文件
            if file.endswith('.jpg') or file.endswith('.png'):
                # 构建输入和输出文件路径
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(output_subdir, file)
                # 打开图像并进行 resize
                img = Image.open(input_path)
                width, height = img.size
                if width < height:
                    new_width = target_size
                    new_height = int(height * target_size / width)
                else:
                    new_height = target_size
                    new_width = int(width * target_size / height)
                size = (new_width, new_height)
                img = img.resize(size, resample=Image.BILINEAR)
                
                # 保存图像到输出文件夹
                img.save(output_path)
        print(f'Processed: {subdir} -> {output_subdir}')
        print(len(os.listdir(subdir)), len(os.listdir(output_subdir)))

if __name__ == "__main__":
    # 定义输入和输出文件夹
    input_folder = '/jhcnas4/syangcw/AutoLaparo/frames'
    output_folder = '/jhcnas4/syangcw/AutoLaparo/frames_resized'
    resize_dataset(input_folder, output_folder, 250)