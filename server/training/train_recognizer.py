"""
Fine-tune PaddleOCR recognition model on manga font crops.

This trains a text recognition model (not detection) using cropped
text region images paired with their correct transcriptions.

Prerequisites:
  pip install paddleocr paddlepaddle

Dataset format (from collect_dataset.py):
  datasets/cropped_text/
    *.png                   <- cropped text region images
    labels.jsonl            <- {"image": "file.png", "text": "correct text"}

  This script converts labels.jsonl to PaddleOCR rec format:
    image_path\ttext

Usage:
  python train_recognizer.py --run_name fonts_v1
  python train_recognizer.py --run_name fonts_v1 --epochs 50 --batch_size 64

The trained model is saved to: models/runs/<run_name>_rec/
Promote to production:
  cp -r models/runs/<run_name>_rec models/trained_fonts
"""

import argparse
import json
import os
import random
import sys


def build_rec_labels(dataset_dir, output_dir, val_split=0.1):
    """Convert labels.jsonl to PaddleOCR recognition label format."""
    labels_path = os.path.join(dataset_dir, 'labels.jsonl')
    if not os.path.isfile(labels_path):
        print(f'ERROR: Labels file not found: {labels_path}')
        sys.exit(1)

    entries = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            img_path = os.path.join(dataset_dir, entry['image'])
            text = entry['text'].strip()
            if not text or not os.path.isfile(img_path):
                continue
            entries.append((img_path, text))

    if len(entries) < 5:
        print(f'ERROR: Need at least 5 labeled samples, found {len(entries)}')
        sys.exit(1)

    random.shuffle(entries)
    split_idx = max(1, int(len(entries) * (1 - val_split)))
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    os.makedirs(output_dir, exist_ok=True)

    train_label_path = os.path.join(output_dir, 'train_rec.txt')
    val_label_path = os.path.join(output_dir, 'val_rec.txt')

    for path, data in [(train_label_path, train_entries), (val_label_path, val_entries)]:
        with open(path, 'w') as f:
            for img_path, text in data:
                f.write(f'{img_path}\t{text}\n')

    # Build character dictionary from all text
    chars = set()
    for _, text in entries:
        chars.update(text)
    chars = sorted(chars)

    dict_path = os.path.join(output_dir, 'manga_dict.txt')
    with open(dict_path, 'w') as f:
        for ch in chars:
            f.write(ch + '\n')

    print(f'Train: {len(train_entries)} samples')
    print(f'Val:   {len(val_entries)} samples')
    print(f'Dict:  {len(chars)} characters -> {dict_path}')

    return train_label_path, val_label_path, dict_path


def main():
    parser = argparse.ArgumentParser(description='Fine-tune PaddleOCR recognizer on manga fonts')
    parser.add_argument('--run_name', required=True, help='Name for this run (e.g. fonts_v1)')
    parser.add_argument('--dataset_dir', default=None, help='Directory with crops + labels.jsonl')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    output_dir = os.path.join(models_dir, 'runs', f'{args.run_name}_rec')
    dataset_dir = args.dataset_dir or os.path.join(script_dir, '..', 'datasets', 'cropped_text')

    if os.path.isdir(output_dir):
        print(f'ERROR: Run directory already exists: {output_dir}')
        print('Choose a different run_name or delete the existing one.')
        sys.exit(1)

    # Build label files
    train_label, val_label, dict_path = build_rec_labels(dataset_dir, output_dir)

    # Write PaddleOCR recognition config
    config = {
        'Global': {
            'use_gpu': False,
            'epoch_num': args.epochs,
            'log_smooth_window': 20,
            'print_batch_step': 10,
            'save_model_dir': output_dir,
            'save_epoch_step': 10,
            'eval_batch_step': [0, 100],
            'cal_metric_during_train': True,
            'pretrained_model': '',  # Uses PP-OCRv4 rec as base
            'character_dict_path': dict_path,
            'max_text_length': 80,
            'use_space_char': True,
        },
        'Architecture': {
            'model_type': 'rec',
            'algorithm': 'SVTR_LCNet',
            'Transform': None,
            'Backbone': {
                'name': 'MobileNetV1Enhance',
                'scale': 0.5,
            },
            'Head': {
                'name': 'MultiHead',
                'head_list': [
                    {
                        'CTCHead': {
                            'Neck': {'name': 'svtr', 'dims': 64, 'depth': 2, 'hidden_dims': 120},
                            'Head': {'fc_decay': 0.00001},
                        }
                    },
                    {
                        'SARHead': {
                            'enc_dim': 512,
                            'max_text_length': 70,
                        }
                    },
                ],
            },
        },
        'Loss': {
            'name': 'MultiLoss',
            'loss_config_list': [
                {'CTCLoss': None},
                {'SARLoss': None},
            ],
        },
        'Optimizer': {
            'name': 'Adam',
            'beta1': 0.9,
            'beta2': 0.999,
            'lr': {
                'name': 'Cosine',
                'learning_rate': args.lr,
                'warmup_epoch': 5,
            },
            'regularizer': {
                'name': 'L2',
                'factor': 0.00001,
            },
        },
        'Train': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': '',
                'label_file_list': [train_label],
                'transforms': [
                    {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                    {'RecAug': {'use_tia': False}},
                    {'CTCLabelEncode': None},
                    {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                    {'KeepKeys': {'keep_keys': ['image', 'label', 'length']}},
                ],
            },
            'loader': {
                'shuffle': True,
                'batch_size_per_card': args.batch_size,
                'drop_last': True,
                'num_workers': 4,
            },
        },
        'Eval': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': '',
                'label_file_list': [val_label],
                'transforms': [
                    {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                    {'CTCLabelEncode': None},
                    {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                    {'KeepKeys': {'keep_keys': ['image', 'label', 'length']}},
                ],
            },
            'loader': {
                'shuffle': False,
                'batch_size_per_card': args.batch_size,
                'drop_last': False,
                'num_workers': 2,
            },
        },
    }

    import yaml
    config_path = os.path.join(output_dir, 'rec_config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f'\nTraining run: {args.run_name}')
    print(f'Output: {output_dir}')
    print(f'Config: {config_path}')
    print(f'Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}')
    print()
    print('To start training, run:')
    print(f'  python -m paddleocr.tools.train -c {config_path}')
    print()
    print('After training, promote to production:')
    print(f'  cp -r {output_dir} {os.path.join(models_dir, "trained_fonts")}')
    print(f'  # Then set USE_LOCAL_RECOGNIZER=true when running server.py')


if __name__ == '__main__':
    main()
