import torch
import torch.nn as nn
import argparse
from yolov5s import My_YOLO as my_yolov5s
from yolov5l import My_YOLO as my_yolov5l
from yolov5m import My_YOLO as my_yolov5m
from yolov5x import My_YOLO as my_yolov5x
from best import My_YOLO as best
import operator
import cv2
from common import Conv,Hardswish,SiLU

class My_YOLOv5s_extract(nn.Module):
    def __init__(self, YOLO, num_classes, anchors=()):
        super().__init__()
        self.backbone = YOLO.backbone_head
        self.ch = YOLO.yolo_layers.ch
        self.no = num_classes + 5  # number of outputs per anchor
        self.na = len(anchors[0]) // 2  # number of anchors
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.m0 = nn.Conv2d(self.ch[0], self.no * self.na, 1)
        self.m1 = nn.Conv2d(self.ch[1], self.no * self.na, 1)
        self.m2 = nn.Conv2d(self.ch[2], self.no * self.na, 1)
    def forward(self, x):
        out0, out1, out2 = self.backbone(x)

        out0 = self.m0(out0)
        out1 = self.m1(out1)
        out2 = self.m2(out2)

        h, w = out0.shape[2:]
        out0 = out0.view(self.na, self.no, h, w).permute(0, 2, 3, 1).contiguous()   ###去掉batchs维度
        out0 = out0.view(-1, self.no)
        h, w = out1.shape[2:]
        out1 = out1.view(self.na, self.no, h, w).permute(0, 2, 3, 1).contiguous()
        out1 = out1.view(-1, self.no)
        h, w = out2.shape[2:]
        out2 = out2.view(self.na, self.no, h, w).permute(0, 2, 3, 1).contiguous()
        out2 = out2.view(-1, self.no)
        return torch.cat((out0, out1, out2), 0)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', default='best', choices=['yolov5s', 'yolov5l', 'yolov5m', 'yolov5x'])
    parser.add_argument('--num_classes', default=1, type=int)
    args = parser.parse_args()
    print(args)

    # Set up model
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    if args.net_type == 'yolov5s':
        net = my_yolov5s(args.num_classes, anchors=anchors, training=False)
    elif args.net_type == 'yolov5l':
        net = my_yolov5l(args.num_classes, anchors=anchors, training=False)
    elif args.net_type == 'yolov5m':
        net = my_yolov5m(args.num_classes, anchors=anchors, training=False)
    elif args.net_type == 'yolov5x':
        net = my_yolov5x(args.num_classes, anchors=anchors, training=False)
    elif args.net_type == 'best':
        net = best(args.num_classes, anchors=anchors, training=False)



    net.to(device)
    net.eval()
    own_state = net.state_dict()
    pth = args.net_type+'_param.pth'
    # pth = args.net_type+'.pt'
    utl_param = torch.load(pth, map_location=device)
    del utl_param['24.anchors']
    del utl_param['24.anchor_grid']

    print(len(utl_param), len(own_state))
    for a, b, namea, nameb in zip(utl_param.values(), own_state.values(), utl_param.keys(), own_state.keys()):
        if namea.find('anchor') > -1:
            print('anchor')
            continue
        if not operator.eq(a.shape, b.shape):
            print(namea, nameb, a.shape, b.shape)
        else:
            own_state[nameb].copy_(a)

    onnx_model = My_YOLOv5s_extract(net, args.num_classes, anchors=anchors).to(device).eval()
    onnx_param = onnx_model.state_dict()

    print(len(utl_param), len(onnx_param))
    for a, b, namea, nameb in zip(utl_param.values(), onnx_param.values(), utl_param.keys(), onnx_param.keys()):
        if namea.find('anchor')>-1:
            print('anchor')
            continue
        if not operator.eq(a.shape, b.shape):
            print(namea, nameb, a.shape, b.shape)
        else:
            onnx_param[nameb].copy_(a)

    output_onnx = args.net_type+'.onnx'
    inputs = torch.randn(1, 3, 640, 640).to(device)

    # Update model
    for k, m in onnx_model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    torch.onnx.export(onnx_model, inputs, output_onnx, verbose=False, opset_version=12, input_names=['images'], output_names=['out'])
    print('convert',output_onnx,'to onnx finish!!!')

    try:
        dnnnet = cv2.dnn.readNet(output_onnx)
        print('read sucess')
    except:
        print('read failed')
