先用../pre_export.py将.pt权值文件转换为.pth文件

将.pth文件放到convert-onnx，

然后cd convert-onnx里面

python convert_onnx.py


出现的问题：
    elif args.net_type == 'best':
        net = best(args.num_classes, anchors=anchors, training=False)

要根据自己的best.py配置文件来生成.onnx文件

