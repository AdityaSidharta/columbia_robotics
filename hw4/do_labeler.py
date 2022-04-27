import argparse
import json
import os
import pathlib
import random
from collections import defaultdict

import torch

import affordance_model
from skimage.io import imsave
from common import get_splits
from common import load_chkpt
from env import UR5PickEnviornment
from image import write_rgb


def main():
    parser = argparse.ArgumentParser(description='Model eval script')
    parser.add_argument('--headless', action='store_true',
                        help='launch pybullet GUI or not')
    parser.add_argument('--seed', type=int, default=10000000,
                        help='random seed for empty_bin task')
    args = parser.parse_args()

    model_class = affordance_model.AffordanceModel

    base_seed = args.seed
    model_dir = os.path.join('data', 'affordance')
    chkpt_path = os.path.join(model_dir, 'best.ckpt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class()
    model.to(device)
    load_chkpt(model, chkpt_path, device)
    model.eval()

    # load env
    env = UR5PickEnviornment(gui=not args.headless)
    names = get_splits()['train'] + get_splits()['test']
    n_samples = 25
    n_attempts = 25
    n_points = 150

    out_dir = pathlib.Path('data', 'do_labels')
    vis_out_dir = out_dir.joinpath('do_vis')
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_out_dir.mkdir(parents=True, exist_ok=True)
    label_file = out_dir.joinpath('labels.json')
    labels = defaultdict(list)
    if label_file.is_file():
        labels.update(json.load(label_file.open('r')))

    for name_idx, name in enumerate(names):
        start_idx = len(labels[name])
        env.load_ycb_objects([name], seed=base_seed)
        for i in range(n_samples):
            success = False
            seed = base_seed + (100 * name_idx) + i
            for j in range(n_attempts):
                env.reset_objects(seed)
                print('Self Supervised {} : Sample {}, Attempt {}'.format(name, i, j))
                rgb_obs, depth_obs, _ = env.observe()
                n = random.choice(range(1, n_points))
                print("Selected nth best : {}".format(n))
                coord, angle, vis_img = model.predict_grasp(rgb_obs, n)
                fname = os.path.join(vis_out_dir, '{}_{}_{}.png'.format(name, i, j))
                imsave(fname, vis_img)
                pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
                result = env.execute_grasp(*pick_pose)
                if result:
                    success = True
                    print("Success!")
                    break
                else:
                    print("Failed, retrying")
            if success:
                write_rgb(rgb_obs, str(out_dir.joinpath('{}_{}_rgb.png'.format(name, start_idx))))
                start_idx = start_idx + 1
                labels[name].append([coord[0],coord[1],angle])
                json.dump(labels, label_file.open('w'), indent=4)
        env.remove_objects()


if __name__ == '__main__':
    main()



