import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from grad_cam import GradCAM, show_cam_on_image, center_crop_img
from model import efficientnetv2_m as create_model
import json



def main(img_path,weights_path,target_category):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = create_model(num_classes=2).to(device)
    weights_dict = torch.load(weights_path, map_location="cpu")

    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if model.state_dict()[k].numel() == v.numel()}


    model.load_state_dict(load_weights_dict, strict=False)

    target_layers = [model.blocks[-1]]



    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)


    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.imshow(visualization)
    plt.savefig('D:\\1.jpg', bbox_inches='tight', dpi=450)

    plt.show()


def test(img_path,weights_path,real_category):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 2

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "m"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = weights_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # print ( "predict_cla:" +str(predict_cla) )
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    real_class_name = "CAG"

    if real_category == 1:
        real_class_name = "CNAG"

    print( "real:" + real_class_name)
    print(print_res)


    return class_indict[str(predict_cla)]


if __name__ == '__main__':
    img_path = "./images/77.bmp"
    img_path = "./a/77_1.bmp"
    weights_path = "E:\\model\\efficentv2-m\\model-165.pth"
    real_category = 1  # 0 CAG  1 CNAG
    categoryName = test(img_path, weights_path, real_category)
    target_category=0;

    if categoryName == "CNAG":
        target_category=1;
    #target_category=0;

    main(img_path, weights_path, target_category)
