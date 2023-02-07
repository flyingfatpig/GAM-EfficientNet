import os
import json
from os import path
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# from efficientnetV2.model import efficientnetv2_s as create_model
from model import efficientnetv2_m as create_model
import csv
import datetime

from utils import get_train_data

    


def main(root_dir):
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
    model_weight_path = "./weights/model-28.pth"
    #model_weight_path = "./w2/model-199.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    files = get_train_data(root_dir)

    log_path = 'test.csv'
    file = open(log_path, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)

    CAG_count = 0
    CAG_T_count = 0
    CAG_F_count = 0

    CNAG_count = 0
    CNAG_T_count = 0
    CNAG_F_count = 0

    for img_path in files:
        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        classname = class_indict[str(predict_cla)]
        filename = os.path.basename(img_path)
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S:%f')
        dir_full = os.path.dirname(img_path)
        dir_full, real_class_name = os.path.split(dir_full)
        
        csv_writer.writerow([filename,real_class_name, classname,  predict[predict_cla].numpy(), time_str])
  
        print(filename + "real:" + real_class_name)
        print(print_res)

        if real_class_name.upper() == "CAG":
            CAG_count += 1
            if classname == "CAG":
                CAG_T_count += 1
            else:
                CAG_F_count += 1
        else:
            CNAG_count += 1
            if classname == "CNAG":
                CNAG_T_count += 1
            else:
                CNAG_F_count += 1

        print("CAG_count", CAG_count)
        print("CAG_T_count", CAG_T_count)
        print("CAG_F_count", CAG_F_count)
        print("CNAG_count", CNAG_count)
        print("CNAG_T_count", CNAG_T_count)
        print("CNAG_F_count", CNAG_F_count)

    csv_writer.writerow(["CAG_count", CAG_count])
    csv_writer.writerow(["CAG_T_count", CAG_T_count])
    csv_writer.writerow(["CAG_F_count", CAG_F_count])
    csv_writer.writerow(["CNAG_count", CNAG_count])
    csv_writer.writerow(["CNAG_T_count", CNAG_T_count])
    csv_writer.writerow(["CNAG_F_count", CNAG_F_count])
    csv_writer.writerow(["acc", ( CAG_T_count +CNAG_T_count)/(CAG_count+CNAG_count) ])
    print("acc",( CAG_T_count +CNAG_T_count)/(CAG_count+CNAG_count) )
    file.close()
    print("finish")

if __name__ == '__main__':
    main('/home/ubuntu/testdata/test1')
    #main('/home/lvbing/dataset')
