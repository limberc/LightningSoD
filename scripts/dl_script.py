import os
from argparse import ArgumentParser

parser = ArgumentParser(description='Download Dataset.')
parser.add_argument('--path', metavar='DIR', default='data',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='DUT-OMRON', )

dataset_dict = {
    "DUT-OMRON": [
        "http://saliencydetection.net/dut-omron/download/DUT-OMRON-image.zip",
        "http://saliencydetection.net/dut-omron/download/DUT-OMRON-bounding-box.zip",
        "http://saliencydetection.net/dut-omron/download/DUT-OMRON-eye-fixations.zip"
        "http://saliencydetection.net/dut-omron/download/DUT-OMRON-gt-pixelwise.zip.zip"
    ],
    "ECSSD": ["https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip",
              "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip",
              "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/our_result_HS.zip",
              "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/our_result_CHS.zip"
              ],
}

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.dataset in dataset_dict, "Dataset not supported"
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    if args.dataset == 'ECSSD':
        for url in dataset_dict[args.dataset]:
            os.system(f"wget {url} --no-check-certificate")
    for url in dataset_dict[args.dataset]:
        os.system(f"aria2c {url}")
