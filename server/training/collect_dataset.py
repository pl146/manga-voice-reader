"""
Collect training data from the live pipeline.

This script runs alongside server.py. When the server processes a page,
it optionally saves each cropped text region and the Google Vision OCR
result as a labeled training pair.

Usage:
  Enable collection in server.py by setting COLLECT_TRAINING_DATA = True
  Crops are saved to: datasets/cropped_text/<timestamp>_<index>.png
  Labels are saved to: datasets/cropped_text/labels.jsonl

  Or run this script standalone to process existing manga page images:
    python collect_dataset.py --input_dir ../datasets/manga_pages --output_dir ../datasets/cropped_text

Each line in labels.jsonl is:
  {"image": "filename.png", "text": "recognized text", "source": "vision"}
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def collect_from_pages(input_dir, output_dir, server_url='http://127.0.0.1:5055'):
    """Send manga page images to the running server and save crop+text pairs."""
    import requests

    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, 'labels.jsonl')

    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ])

    if not images:
        print(f'No images found in {input_dir}')
        return

    print(f'Processing {len(images)} images...')
    total_crops = 0

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f'  Skip (unreadable): {img_name}')
            continue

        # Encode as base64 data URL
        import base64
        _, buf = cv2.imencode('.png', img)
        b64 = base64.b64encode(buf).decode('utf-8')
        data_url = f'data:image/png;base64,{b64}'

        try:
            resp = requests.post(
                f'{server_url}/process',
                json={'image': data_url, 'dpr': 1},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f'  Skip (server error): {img_name}: {e}')
            continue

        bubbles = result.get('bubbles', [])
        if not bubbles:
            print(f'  No bubbles: {img_name}')
            continue

        # Also get raw crops by re-detecting locally
        h, w = img.shape[:2]
        for i, bubble in enumerate(bubbles):
            # Convert CSS coords back to pixel coords (dpr=1)
            crop_left = result.get('actualCropLeft', 0)
            crop_top = result.get('actualCropTop', 0)
            x = int(bubble['left'] - crop_left)
            y = int(bubble['top'] - crop_top)
            bw = int(bubble['width'])
            bh = int(bubble['height'])

            # Add padding
            pad = 8
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ts = int(time.time() * 1000)
            crop_name = f'{img_name.rsplit(".", 1)[0]}_{i:03d}_{ts}.png'
            crop_path = os.path.join(output_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            label = {
                'image': crop_name,
                'text': bubble['text'],
                'source': 'vision',
                'source_page': img_name,
                'confidence': bubble.get('conf', 0),
            }
            with open(labels_path, 'a') as f:
                f.write(json.dumps(label) + '\n')

            total_crops += 1

        print(f'  {img_name}: {len(bubbles)} crops saved')

    print(f'\nDone. {total_crops} crops saved to {output_dir}')
    print(f'Labels: {labels_path}')


def save_crop_pair(output_dir, crop_img, text, source_info='pipeline'):
    """Save a single crop+text pair. Called from server.py during live processing."""
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, 'labels.jsonl')

    ts = int(time.time() * 1000)
    crop_name = f'crop_{ts}.png'
    crop_path = os.path.join(output_dir, crop_name)
    cv2.imwrite(crop_path, crop_img)

    label = {
        'image': crop_name,
        'text': text,
        'source': source_info,
    }
    with open(labels_path, 'a') as f:
        f.write(json.dumps(label) + '\n')

    return crop_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect manga OCR training data')
    parser.add_argument('--input_dir', required=True, help='Directory with manga page images')
    parser.add_argument('--output_dir', default=None, help='Output directory for crops+labels')
    parser.add_argument('--server', default='http://127.0.0.1:5055', help='Server URL')
    args = parser.parse_args()

    output = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'datasets', 'cropped_text'
    )
    collect_from_pages(args.input_dir, output, args.server)
