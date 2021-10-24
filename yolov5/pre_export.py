import torch
from collections import OrderedDict
import pickle
import os

 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
if __name__=='__main__':
    # choices = ['last', 'yolov5s', 'yolov5l', 'yolov5m', 'yolov5x']
    # modelfile = choices[1]+'.pt'
    modelfile = '/home/yy/project/yolov5/runs/train/exp/weights/best.pt'
    utl_model = torch.load(modelfile, map_location=device)
    utl_param = utl_model['model'].model
    torch.save(utl_param.state_dict(), os.path.splitext(modelfile)[0]+'_param.pth')
    own_state = utl_param.state_dict()
    print(len(own_state))
 
    # numpy_param = OrderedDict()
    # for name in own_state:
    #     numpy_param[name] = own_state[name].data.cpu().numpy()
    # print(len(numpy_param))
    # with open(os.path.splitext(modelfile)[0]+'_numpy_param.pkl', 'wb') as fw:
    #     pickle.dump(numpy_param, fw)
