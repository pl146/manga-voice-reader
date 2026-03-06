"""
Fine-tune PaddleOCR detection model on manga text regions.

Prerequisites:
  pip install paddleocr paddlepaddle paddledet

Dataset format: COCO JSON
  training/dataset/
    images/        <- manga page images (PNG/JPG)
    labels/
      train.json   <- COCO format annotations (category: "text_region")
      val.json

Usage:
  python train.py --run_name run_001

The trained model is saved to ../models/runs/<run_name>/
Use ../scripts/promote.sh <run_name> to make it the production model.
"""

import argparse
import os
import shutil
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description='Fine-tune PaddleOCR detector on manga')
    parser.add_argument('--run_name', required=True, help='Name for this run (e.g. run_001)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    output_dir = os.path.join(models_dir, 'runs', args.run_name)
    dataset_dir = os.path.join(script_dir, 'dataset')
    config_path = os.path.join(script_dir, 'config.yml')

    # Validate dataset exists
    train_json = os.path.join(dataset_dir, 'labels', 'train.json')
    val_json = os.path.join(dataset_dir, 'labels', 'val.json')
    images_dir = os.path.join(dataset_dir, 'images')

    if not os.path.isfile(train_json):
        print(f'ERROR: Training annotations not found at {train_json}')
        print('Create COCO-format annotations first. See README for labeling instructions.')
        sys.exit(1)

    if not os.path.isdir(images_dir) or not os.listdir(images_dir):
        print(f'ERROR: No images found in {images_dir}')
        sys.exit(1)

    # Create output directory (never overwrite existing runs)
    if os.path.isdir(output_dir):
        print(f'ERROR: Run directory already exists: {output_dir}')
        print('Choose a different run_name or delete the existing one.')
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Training run: {args.run_name}')
    print(f'Output: {output_dir}')
    print(f'Dataset: {dataset_dir}')
    print(f'Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}')
    print()

    # Load and modify config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['Global']['epoch_num'] = args.epochs
    config['Global']['save_model_dir'] = output_dir
    config['Global']['pretrained_model'] = os.path.join(
        models_dir, 'baseline_detector', 'inference'
    )
    config['Optimizer']['lr']['learning_rate'] = args.lr
    config['Train']['dataset']['data_dir'] = images_dir
    config['Train']['dataset']['label_file_list'] = [train_json]
    config['Train']['loader']['batch_size_per_card'] = args.batch_size
    config['Eval']['dataset']['data_dir'] = images_dir
    config['Eval']['dataset']['label_file_list'] = [val_json]

    # Write runtime config
    runtime_config = os.path.join(output_dir, 'config.yml')
    with open(runtime_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f'Config written to {runtime_config}')
    print()
    print('To start training, run:')
    print(f'  python -m paddle.distributed.launch tools/train.py -c {runtime_config}')
    print()
    print('Or with PaddleOCR CLI:')
    print(f'  python -m paddleocr.tools.train -c {runtime_config}')
    print()
    print(f'After training, promote to production:')
    print(f'  cd ../scripts && ./promote.sh {args.run_name}')


if __name__ == '__main__':
    main()
