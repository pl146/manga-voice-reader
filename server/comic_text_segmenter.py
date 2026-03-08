"""
Comic Text Segmenter — uses comic-text-detector ONNX model to produce
pixel-level text masks from manga/comic images.

The mask isolates text pixels from background art, giving OCR engines
clean black-on-white input instead of noisy manga artwork.

Model: comictextdetector.pt.onnx (from dmMaze/comic-text-detector)
Backend: OpenCV DNN (no PyTorch needed)
"""

import os
import logging
import cv2
import numpy as np

log = logging.getLogger('mvr')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'comic_text_detector', 'comictextdetector.pt.onnx')
INPUT_SIZE = 1024


def _letterbox(img, new_shape=1024, stride=64):
    """Resize image keeping aspect ratio, pad to square with stride alignment."""
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Ensure dimensions are divisible by stride
    new_h = max(stride, (new_h // stride) * stride)
    new_w = max(stride, (new_w // stride) * stride)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target shape
    dh = new_shape[0] - new_h
    dw = new_shape[1] - new_w
    if dh > 0 or dw > 0:
        padded = cv2.copyMakeBorder(resized, 0, dh, 0, dw,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padded = resized

    return padded, scale, dw, dh


class ComicTextSegmenter:
    """Extract text segmentation masks from manga/comic images using ONNX model."""

    def __init__(self, model_path=MODEL_PATH, input_size=INPUT_SIZE):
        self.input_size = input_size
        self.model = None
        self.output_names = None
        self._model_path = model_path

    def load(self):
        """Load the ONNX model. Call once at startup."""
        if not os.path.isfile(self._model_path):
            log.warning(f'Comic text detector model not found: {self._model_path}')
            return False
        try:
            self.model = cv2.dnn.readNetFromONNX(self._model_path)
            self.output_names = self.model.getUnconnectedOutLayersNames()
            log.info(f'Comic text segmenter loaded: {os.path.basename(self._model_path)} '
                     f'({os.path.getsize(self._model_path) / 1024 / 1024:.0f}MB), '
                     f'outputs: {self.output_names}')
            return True
        except Exception as e:
            log.warning(f'Failed to load comic text detector: {e}')
            self.model = None
            return False

    @property
    def available(self):
        return self.model is not None

    def get_text_mask(self, img):
        """Run the model on a BGR image and return a text segmentation mask.

        Returns:
            mask: uint8 array same size as input, 0-255 where 255 = text pixel
        """
        if not self.available:
            return None

        h, w = img.shape[:2]

        # Convert BGR -> RGB (model was trained on RGB)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox to input_size x input_size
        padded, scale, dw, dh = _letterbox(rgb, self.input_size)

        # Create blob: NCHW, float32, normalized to 0-1
        blob = cv2.dnn.blobFromImage(padded, scalefactor=1.0 / 255.0)
        self.model.setInput(blob)

        # Run inference — model outputs: [blks, mask, lines_map]
        outputs = self.model.forward(self.output_names)

        # Model outputs: blk (detections), det (2-ch: mask+lines), seg (1-ch: clean mask)
        # We want 'seg' — the clean single-channel text segmentation mask
        # It's the output with shape (1, 1, H, W)
        mask = None
        for out in outputs:
            if len(out.shape) == 4 and out.shape[1] == 1 and out.shape[2] == self.input_size:
                mask = out
                break

        if mask is None:
            # Fallback: use first channel of the 2-channel det output
            for out in outputs:
                if len(out.shape) == 4 and out.shape[1] == 2 and out.shape[2] == self.input_size:
                    mask = out[:, 0:1, :, :]  # first channel
                    break

        if mask is None:
            log.warning('Comic text detector: could not identify mask output')
            return None

        # Squeeze to 2D — model may output (1, C, H, W) or (C, H, W)
        mask = mask.squeeze()
        if mask.ndim == 3:
            # Multi-channel mask — take first channel (text probability)
            # Channel 0 is typically the text mask, channel 1 is lines
            mask = mask[0]
        if mask.ndim != 2:
            log.warning(f'Comic text detector: unexpected mask shape {mask.shape}')
            return None

        # Convert to uint8
        mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)

        # Remove letterbox padding
        effective_h = self.input_size - dh
        effective_w = self.input_size - dw
        mask = mask[:effective_h, :effective_w]

        # Resize back to original image size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        return mask

    def clean_crop(self, img, mask, x, y, w, h, pad=5):
        """Extract a clean black-text-on-white-background crop using the text mask.

        Args:
            img: full BGR image
            mask: text mask from get_text_mask()
            x, y, w, h: box coordinates
            pad: extra padding pixels around the box

        Returns:
            clean: BGR image with text in black on white background
        """
        img_h, img_w = img.shape[:2]

        # Clamp and pad
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        crop_mask = mask[y1:y2, x1:x2]
        crop_img = img[y1:y2, x1:x2]

        if crop_mask.size == 0 or crop_img.size == 0:
            return crop_img

        # Threshold the mask — pixels > 127 are text
        _, binary_mask = cv2.threshold(crop_mask, 100, 255, cv2.THRESH_BINARY)

        # Dilate the mask slightly to include edges of characters
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Create white background
        clean = np.full_like(crop_img, 255)

        # Copy only text pixels from original image
        text_pixels = binary_mask > 0
        clean[text_pixels] = crop_img[text_pixels]

        # Convert to grayscale and binarize for maximum OCR clarity
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

        # Otsu threshold — works great on clean text-on-white
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure text is dark on light background
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # Convert back to BGR (for compatibility with OCR pipelines expecting BGR)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def clean_crop_for_ocr(self, img, mask, x, y, w, h):
        """Get a clean crop optimized for OCR: upscaled, padded, binarized.

        Returns a PIL-ready grayscale image.
        """
        from PIL import Image

        clean = self.clean_crop(img, mask, x, y, w, h, pad=8)
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

        ch, cw = gray.shape[:2]

        # Upscale small crops (2x minimum, more if tiny)
        scale = max(2.0, 80.0 / max(1, ch))
        new_w, new_h = int(cw * scale), int(ch * scale)
        upscaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Add white padding (20px border)
        padded = np.full((new_h + 40, new_w + 40), 255, dtype=np.uint8)
        padded[20:20 + new_h, 20:20 + new_w] = upscaled

        return Image.fromarray(padded)
