import torch
from tools.test import parse_args
from lib.config.default import update_config
from lib.config import cfg
from lib.models import get_net

def main(path):
    args = parse_args()
    update_config(cfg, args)

    model = get_net(cfg)
    model_dict = model.state_dict()
    # for k in model_dict.keys():
    #     print(k)

    checkpoint_file = path
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    # for k in checkpoint['state_dict'].keys():
    #     print(k)

    checkpoint_dict = {}
    for k, v in checkpoint['state_dict'].items():
        index = int(k.split('.')[1])
        if index > 25:
            k = k[0: 6] + str(int(k[6: 8]) - 9) + k[8:]
        if 14 < index <= 25:
            continue
        checkpoint_dict[k] = v
    # for k in checkpoint_dict.keys():
    #     print(k)

    print(len(model_dict.keys()))
    print(len(checkpoint['state_dict'].keys()))
    print(len(checkpoint_dict.keys()))

    model_dict.update(checkpoint_dict)

    print(len(model_dict.keys()))
    #
    checkpoint['state_dict'] = checkpoint_dict
    torch.save(checkpoint, 'onlyll.pth')


def test():
    pth = torch.load('new_ckpt.pth')
    print(pth['state_dict']['model.9.m.0.cv2.bn.bias'])


if __name__ == '__main__':
    PATH = './new_ckpt.pth'
    main(PATH)
    # test()
