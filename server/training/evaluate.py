"""
Evaluate OCR recognition accuracy on labeled validation images.

Compares recognized text against ground truth labels and reports
character-level and word-level accuracy.

Usage:
  # Evaluate Google Vision OCR on validation set
  python evaluate.py --dataset_dir ../datasets/validation

  # Evaluate local PaddleOCR recognizer
  python evaluate.py --dataset_dir ../datasets/validation --mode local

  # Compare both
  python evaluate.py --dataset_dir ../datasets/validation --mode compare

Dataset format (same as collect_dataset.py output):
  dataset_dir/
    *.png
    labels.jsonl   <- {"image": "file.png", "text": "ground truth"}
"""

import argparse
import json
import os
import sys
import time


def levenshtein(s1, s2):
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def char_accuracy(predicted, ground_truth):
    """Character-level accuracy: 1 - (edit_distance / max_len)."""
    if not ground_truth and not predicted:
        return 1.0
    max_len = max(len(predicted), len(ground_truth))
    if max_len == 0:
        return 1.0
    dist = levenshtein(predicted.lower(), ground_truth.lower())
    return max(0, 1.0 - dist / max_len)


def word_accuracy(predicted, ground_truth):
    """Word-level accuracy: fraction of words that match exactly."""
    pred_words = predicted.lower().split()
    gt_words = ground_truth.lower().split()
    if not gt_words:
        return 1.0 if not pred_words else 0.0
    matches = 0
    for pw, gw in zip(pred_words, gt_words):
        if pw == gw:
            matches += 1
    return matches / max(len(pred_words), len(gt_words))


def exact_match(predicted, ground_truth):
    """Case-insensitive exact match."""
    return predicted.strip().lower() == ground_truth.strip().lower()


def load_labels(dataset_dir):
    """Load labels.jsonl and return list of (image_path, ground_truth_text)."""
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
            if not os.path.isfile(img_path):
                continue
            entries.append((img_path, entry['text'].strip()))
    return entries


def recognize_vision(image_paths):
    """Recognize text using Google Vision API."""
    from google.cloud import vision
    import cv2

    client = vision.ImageAnnotatorClient()
    texts = []

    # Batch in groups of 16
    for i in range(0, len(image_paths), 16):
        batch_paths = image_paths[i:i + 16]
        requests = []
        for path in batch_paths:
            with open(path, 'rb') as f:
                content = f.read()
            requests.append(
                vision.AnnotateImageRequest(
                    image=vision.Image(content=content),
                    features=[vision.Feature(
                        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
                    )],
                )
            )
        response = client.batch_annotate_images(requests=requests)
        for resp in response.responses:
            if resp.full_text_annotation and resp.full_text_annotation.text:
                texts.append(resp.full_text_annotation.text.strip())
            else:
                texts.append('')
    return texts


def recognize_local(image_paths, model_dir):
    """Recognize text using local PaddleOCR recognizer."""
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    from paddleocr import PaddleOCR
    import cv2

    kwargs = {
        'use_textline_orientation': False,
        'use_doc_orientation_classify': False,
        'use_doc_unwarping': False,
    }
    if os.path.isdir(model_dir):
        kwargs['text_recognition_model_dir'] = model_dir
    else:
        kwargs['text_recognition_model_name'] = 'PP-OCRv4_mobile_rec'

    engine = PaddleOCR(**kwargs)
    texts = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            texts.append('')
            continue
        results = engine.predict(img)
        if results:
            # Collect all recognized text
            page_text = []
            for r in results:
                res = r.json.get('res', {})
                for t in res.get('rec_texts', []):
                    if t.strip():
                        page_text.append(t.strip())
            texts.append(' '.join(page_text))
        else:
            texts.append('')
    return texts


def run_evaluation(entries, predictions, label):
    """Compute and print metrics."""
    total_char_acc = 0
    total_word_acc = 0
    total_exact = 0
    errors = []

    for (img_path, gt_text), pred_text in zip(entries, predictions):
        ca = char_accuracy(pred_text, gt_text)
        wa = word_accuracy(pred_text, gt_text)
        em = exact_match(pred_text, gt_text)
        total_char_acc += ca
        total_word_acc += wa
        total_exact += int(em)

        if not em:
            errors.append((os.path.basename(img_path), gt_text, pred_text, ca))

    n = len(entries)
    print(f'\n{"=" * 60}')
    print(f'  {label} — {n} samples')
    print(f'{"=" * 60}')
    print(f'  Character accuracy:  {total_char_acc / n:.1%}')
    print(f'  Word accuracy:       {total_word_acc / n:.1%}')
    print(f'  Exact match:         {total_exact}/{n} ({total_exact / n:.1%})')

    if errors:
        # Show worst 10 errors
        errors.sort(key=lambda e: e[3])
        print(f'\n  Worst errors (lowest char accuracy):')
        for img, gt, pred, ca in errors[:10]:
            print(f'    [{ca:.0%}] {img}')
            print(f'      GT:   {gt[:80]}')
            print(f'      Pred: {pred[:80]}')

    return {
        'char_accuracy': total_char_acc / n,
        'word_accuracy': total_word_acc / n,
        'exact_match': total_exact / n,
        'total': n,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate manga OCR accuracy')
    parser.add_argument('--dataset_dir', required=True, help='Directory with crops + labels.jsonl')
    parser.add_argument('--mode', choices=['vision', 'local', 'compare'], default='vision',
                        help='Which recognizer to evaluate')
    parser.add_argument('--model_dir', default=None, help='Local model directory (for local/compare modes)')
    args = parser.parse_args()

    entries = load_labels(args.dataset_dir)
    if not entries:
        print('No valid entries found.')
        sys.exit(1)

    print(f'Loaded {len(entries)} labeled samples from {args.dataset_dir}')
    image_paths = [e[0] for e in entries]
    model_dir = args.model_dir or os.path.join(
        os.path.dirname(__file__), '..', 'models', 'trained_fonts'
    )

    if args.mode in ('vision', 'compare'):
        print('\nRunning Google Vision OCR...')
        t0 = time.time()
        vision_preds = recognize_vision(image_paths)
        print(f'  Done in {time.time() - t0:.1f}s')
        vision_results = run_evaluation(entries, vision_preds, 'Google Vision')

    if args.mode in ('local', 'compare'):
        print(f'\nRunning local recognizer ({model_dir})...')
        t0 = time.time()
        local_preds = recognize_local(image_paths, model_dir)
        print(f'  Done in {time.time() - t0:.1f}s')
        local_results = run_evaluation(entries, local_preds, 'Local PaddleOCR')

    if args.mode == 'compare':
        print(f'\n{"=" * 60}')
        print(f'  COMPARISON')
        print(f'{"=" * 60}')
        diff_char = local_results['char_accuracy'] - vision_results['char_accuracy']
        diff_word = local_results['word_accuracy'] - vision_results['word_accuracy']
        sign = lambda x: '+' if x >= 0 else ''
        print(f'  Char accuracy: Vision {vision_results["char_accuracy"]:.1%} vs Local {local_results["char_accuracy"]:.1%} ({sign(diff_char)}{diff_char:.1%})')
        print(f'  Word accuracy: Vision {vision_results["word_accuracy"]:.1%} vs Local {local_results["word_accuracy"]:.1%} ({sign(diff_word)}{diff_word:.1%})')


if __name__ == '__main__':
    main()
