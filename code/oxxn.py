import torch
from torchvision.models import resnet50
from backbone import resnet50_fpn_backbone, LastLevelP6P7

import netron
modelpath=r"D:/pth/output/resNetFpn-model-14.onnx"
netron.start(modelpath)

import onnx
from onnx import shape_inference

model = r'D:/pth/output/resNetFpn-model-14.onnx'
#上一步保存好的onnx格式的模型路径
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)), model)







device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import train
modelpath=r"D:/pth/output/resNetFpn-model-14.pth"
#pth模型路径
# model = resnet50_fpn_backbone(returned_layers=[1, 2, 3, 4],
#                                      extra_blocks=LastLevelP6P7(256, 256),
#                                      trainable_layers=3).to(device)
model=train.create_model(20)
#加载网络结构
w=torch.load(modelpath)
#torch加载模型
model.load_state_dict(w,strict=False)
#网络加载模型参数
model.to(device)
#模型放到device上
model.eval()
#将模型置于评估模式 不启用 BatchNormalization 和 Dropout,不改变权值
inputs = torch.ones((1, 3, 500,500)).to(device)
#网络输入size 同样放到device上
onnxpath=r"D:/pth/output/resNetFpn-model-14.onnx"
#onnx格式的模型保存路径
torch.onnx.export(model,inputs,onnxpath,export_params=True,verbose=True,input_names=['input'],output_names=['output'],opset_version=12)
#模型格式转换
