import os
import cv2
import numpy as np
import torch
from PIL import Image

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision import transforms

from coreapi.low.object_segmentation.utils import probs2mask, detect_paper, four_point_transform

transform = transforms.Compose([
    transforms.Resize(256),           # Resize the shorter side to 256 pixels
    transforms.CenterCrop(224),       # Crop the center 224x224 pixels
    transforms.ToTensor(),            # Convert image to PyTorch tensor
    transforms.Normalize(             # Normalize based on ImageNet standards
        mean=[0.4611, 0.4359, 0.3905],   # These are the means for ImageNet images
        std=[0.2193, 0.2150, 0.2109]     # These are the std devs for ImageNet images
    ),
])

class Segmenter(object):
    def __init__(self, num_classes=2, input_size=384) -> None:
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "model_mbv3_iou_mix_2C049.pth")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoints = torch.load(checkpoint_path, map_location=device)
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True).to(device)
        model.load_state_dict(checkpoints)
        model.eval()
        # with torch.no_grad():
        #     _ = model(torch.randn((1, 3, input_size, input_size)))
        self.model = model
        self.input_size = input_size
        self.device = device
        self.transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),           # Resize the shorter side to 256 pixels
            transforms.ToTensor(),            # Convert image to PyTorch tensor
            transforms.Normalize(             # Normalize based on ImageNet standards
                mean=[0.4611, 0.4359, 0.3905],   # These are the means for ImageNet images
                std=[0.2193, 0.2150, 0.2109]     # These are the std devs for ImageNet images
            ),
        ])

    def process_input(self, image):
        # image = Image.fromarray(image).convert('RGB')
        image_tensor = self.transforms(image)
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        image_tensor = image_tensor.to(self.device)
        return image_tensor
    
    def predict(self, image):
        imW, imH = image.size
        scale_x = imW / self.input_size
        scale_y = imH / self.input_size

        image_tensor = self.process_input(image.copy())
        with torch.no_grad():
            out = self.model(image_tensor)["out"]
        
        mask = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].cpu().numpy().squeeze().astype(np.uint8)
        rect = detect_paper(mask.copy())
        rect[:, 0] *= scale_x
        rect[:, 1] *= scale_y
        # rect[[0, -1], 0] -= 0.05*imW
        warped = four_point_transform(np.array(image).astype(np.uint8), rect)
        return Image.fromarray(warped).convert("RGB")


if __name__ == "__main__":
    import glob
    from tqdm import tqdm
    output_dir = r"outputs\segmentor"
    os.makedirs(output_dir, exist_ok=True)
    segmenter = Segmenter()
    image_paths = glob.glob(r"E:\project\ARS\str\images\*.jpg") + glob.glob(r"E:\project\ARS\str\images\*.png")
    for image_path in image_paths:
    # image_path = r"E:\project\ARS\str\images\1730400849733.jpg"
        image = cv2.imread(image_path)
        img_rst = segmenter.predict(image)
        img_rst = cv2.resize(img_rst, (image.shape[1], image.shape[0]))
        # print(img_rst.shape, image.shape)
        img_rst = np.hstack((image, img_rst))
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), img_rst)



