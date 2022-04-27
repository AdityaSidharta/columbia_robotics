from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from pick_labeler import draw_grasp


class ActionRegressionDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for ActionRegressionModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32
            'target': torch.Tensor (3,), torch.float32
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        Note: target: [x, y, angle] scaled to between 0 and 1.
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]

        rgb = np.array(data['rgb'])
        center_point = np.array(data['center_point'])
        angle = np.array(data['angle'])

        x_max = rgb.shape[0]
        y_max = rgb.shape[1]

        # print("x_max : {}".format(x_max))
        # print("y_max : {}".format(y_max))
        x = center_point[0]
        y = center_point[1]

        x_norm = x / x_max
        y_norm = y / y_max

        assert -180. <= angle <= 180.
        angle_norm = (angle + 180.) / 360.

        # print("x : {}, y : {}, angle : {}, x_norm : {}, y_norm : {}, angle_norm : {}".format(x, y, angle, x_norm, y_norm, angle_norm))

        input = torch.from_numpy(rgb).permute(2, 0, 1).type(torch.float32)
        target = torch.from_numpy(np.array([x_norm, y_norm, angle_norm])).type(torch.float32)

        # ===============================================================================
        return {
            'input': input,
            'target': target
        }
        # ===============================================================================


def recover_action(
        action: np.ndarray, 
        shape=(128,128)
        ) -> Tuple[Tuple[int, int], float]:
    """
    :action: np.ndarray([x,y,angle], dtype=np.float32)
    return:
    coord: tuple(x, y) in pixel coordinate between 0 and :shape:
    angle: float in degrees, clockwise
    """
    # ===============================================================================
    x_norm, y_norm, angle_norm = action[0], action[1], action[2]
    x = x_norm * shape[0]
    y = y_norm * shape[1]
    angle = (angle_norm * 360.) - 180.
    coord, angle = (int(x), int(y)), angle
    # ===============================================================================
    return coord, angle


class ActionRegressionModel(nn.Module):
    def __init__(self, pretrained=False, out_channels=3, **kwargs):
        super().__init__()
        # load backbone model
        model = mobilenet_v3_small(pretrained=pretrained)
        # replace the last linear layer to change output dimention to 3
        ic = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features=ic, out_features=out_channels)
        self.model = model
        # normalize RGB input to zero mean and unit variance
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.angle_weight = 0.1

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))

    def predict(self, x):
        """
        Think: Why is this the same as forward 
        (comparing to AffordanceModel.predict)
        """
        return self.forward(x)

    def action_loss(self, output, target):
        x_loss = torch.mean((output[:, 0] - target[:, 0])**2)
        y_loss = torch.mean((output[:, 1] - target[:, 1])**2)
        angle_loss = torch.mean(1 - torch.cos(output[:, 2] - target[:, 2]))
        # print("x_loss : {}, y_loss : {}, angle_loss: {}".format(x_loss, y_loss, angle_loss))
        return x_loss + y_loss + (self.angle_weight * angle_loss)

    def get_criterion(self):
        """
        Return the Loss object needed for training.
        """
        # ===============================================================================
        return nn.MSELoss()
        # ===============================================================================

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """        
        vis_img = (np.moveaxis(input,0,-1).copy() * 255).astype(np.uint8)
        # target
        if target is not None:
            coord, angle = recover_action(target, shape=vis_img.shape[:2])
            draw_grasp(vis_img, coord, angle, color=(255,255,255))
        # pred
        coord, angle = recover_action(output, shape=vis_img.shape[:2])
        draw_grasp(vis_img, coord, angle, color=(0,255,0))
        return vis_img

    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Hint: use recover_action
        """
        device = self.device
        rgb_obs = np.moveaxis(rgb_obs, -1, 0)
        rgb_obs = np.expand_dims(rgb_obs, axis = 0)
        input_value = torch.from_numpy(rgb_obs).type(torch.float32).to(device)
        with torch.no_grad():
            action = self.predict(input_value)[0].cpu().numpy()
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        (coord, angle), action = recover_action(action), action
        # ===============================================================================
        # visualization
        vis_img = self.visualize(rgb_obs[0,...], action)
        print("coord : {}, angle : {}".format(coord, angle))
        return coord, angle, vis_img