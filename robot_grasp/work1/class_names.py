import torch

# 读取文件
data = torch.load(r"D:\AI\WRS-6DoF-PE\robot_grasp\best_rack34.pt", map_location="cpu")

model = data['model']  # 这是一个 nn.Module
print(type(model))

# 如果 model 有 names 属性
if hasattr(model, 'names'):
    print("类别信息：", model.names)
else:
    # 如果 names 没有直接暴露，可以查看 __dict__
    print(model.__dict__.keys())

    # 或者直接打印相关参数
    if hasattr(model, 'yaml'):
        print("配置文件：", model.yaml)

    # 检查保存的 opt 参数
    print("opt:", data['opt'])