# openmlkit/pipeline.py

import os
import cv2
import numpy as np
from .labelmap import LabelMap
from .detector import TextDetector
from .recognizer import TextRecognizer

class OpenMLKitOCR:
    def _ensure_model(self, models_dir, relative_path):
        """
        Checks if the model file exists locally. If not, attempts to download it
        from Hugging Face repository configured by the environment variable
        'OPENMLKIT_MODEL_REPO' (defaults to '0cve0/OpenMLKitOCR').
        """
        local_path = os.path.join(models_dir, relative_path)
        if os.path.exists(local_path):
            return local_path

        repo_id = os.environ.get("OPENMLKIT_MODEL_REPO", "0cve0/OpenMLKitOCR")
        print(f"Model file '{relative_path}' not found locally at {local_path}. Downloading from Hugging Face ({repo_id})...")
        
        try:
            from huggingface_hub import hf_hub_download
            # hf_hub_download supports subdirectories in filename
            cached_path = hf_hub_download(repo_id=repo_id, filename=relative_path)
            return cached_path
        except Exception as e:
            # Fallback to urllib.request
            import urllib.request
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            url = f"https://huggingface.co/{repo_id}/resolve/main/{relative_path}"
            try:
                print(f"Downloading {url} to {local_path}...")
                urllib.request.urlretrieve(url, local_path)
                return local_path
            except Exception as download_error:
                raise FileNotFoundError(
                    f"Required model file not found: {relative_path} at {local_path} and failed to download from {url}.\n"
                    f"Error: {download_error}\n"
                    f"Please verify internet connection or place the model file manually."
                )

    def __init__(self, models_dir=None, lang='en'):
        """
        Initializes the OCR pipeline.
        lang: 'en' (Latin) or 'ru' (Cyrillic).
        If models_dir is None, it defaults to the 'models' subdirectory inside this package.
        """
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            
        det_rel = 'detector/rpn_detector.tflite'
        
        if lang == 'ru':
            rec_rel = 'ru/recognizer_cyrl.tflite'
            labelmap_rel = 'ru/LabelMap_cyrl.pb'
        elif lang == 'zh':
            rec_rel = 'zh/recognizer_hani.tflite'
            labelmap_rel = 'zh/recognizer_hani_label_map.pb'
        elif lang == 'ko':
            rec_rel = 'ko/recognizer_kore.tflite'
            labelmap_rel = 'ko/recognizer_kore_label_map.pb'
        elif lang == 'ja':
            rec_rel = 'ja/recognizer_jpan.tflite'
            labelmap_rel = 'ja/recognizer_jpan_label_map.pb'
        elif lang == 'ar':
            rec_rel = 'ar/recognizer_arab_retrained.tflite'
            labelmap_rel = 'ar/recognizer_arab_label_map.pb'
        elif lang == 'he':
            rec_rel = 'he/hebr.tflite'
            labelmap_rel = 'he/hebr_label_map.pb'
        elif lang == 'th':
            rec_rel = 'th/recognizer_thai.tflite'
            labelmap_rel = 'th/recognizer_thai_label_map.pb'
        elif lang == 'ka':
            rec_rel = 'ka/geor.tflite'
            labelmap_rel = 'ka/geor_label_map.pb'
        elif lang == 'bn':
            rec_rel = 'bn/bede.tflite'
            labelmap_rel = 'bn/bede_label_map.pb'
        elif lang == 'ta':
            rec_rel = 'ta/recognizer_taml.tflite'
            labelmap_rel = 'ta/recognizer_taml_label_map.pb'
        elif lang == 'te':
            rec_rel = 'te/recognizer_telu.tflite'
            labelmap_rel = 'te/recognizer_telu_label_map.pb'
        elif lang == 'kn':
            rec_rel = 'kn/recognizer_knda.tflite'
            labelmap_rel = 'kn/recognizer_knda_label_map.pb'
        elif lang == 'ml':
            rec_rel = 'ml/recognizer_mlym.tflite'
            labelmap_rel = 'ml/recognizer_mlym_label_map.pb'
        elif lang == 'gu':
            rec_rel = 'gu/gocr_tflite_recognizer_gujr.tflite'
            labelmap_rel = 'gu/gocr_tflite_recognizer_gujr_label_map.pb'
        elif lang in ('en_translate', 'latn_vi'):
            rec_rel = 'vi/gocr_tflite_recognizer_latn_vi.tflite'
            labelmap_rel = 'vi/gocr_tflite_recognizer_latn_vi_label_map.pb'
        else:
            rec_rel = 'en/line_recognizer.fb'
            labelmap_rel = 'en/LabelMap.pb'
            
        det_model = self._ensure_model(models_dir, det_rel)
        rec_model = self._ensure_model(models_dir, rec_rel)
        labelmap_path = self._ensure_model(models_dir, labelmap_rel)
                
        self.label_map = LabelMap(labelmap_path)
        self.detector = TextDetector(det_model)
        self.recognizer = TextRecognizer(rec_model, self.label_map)
        
        # Configure stitching parameters based on script characteristics
        self.lang = lang
        if lang in ('zh', 'ja', 'ko'):
            self.min_match_len = 2
            self.max_scan = 12
            self.max_off = 3
        else:  # Alphabetic scripts (Latin, Cyrillic, Arabic …)
            self.min_match_len = 3
            self.max_scan = 22
            self.max_off = 6
        
    def _merge_overlapping_texts(self, t1, t2):
        """Merge two overlapping OCR chunk results using fuzzy suffix-prefix alignment.

        The model often produces small errors at chunk edges (wrong case, extra
        or missing character, boundary artefact). This method tolerates up to
        ``max(1, L // 5)`` substitutions over an overlap window of length L and
        also allows a small positional *offset* so the overlap does not have to
        start at the very first character of t2.

        Strategy (tried in order):
        1. Exact suffix-of-t1 / prefix-of-t2 character match.
        2. Exact word-level suffix-of-t1 / prefix-of-t2 match.
        3. Best fuzzy alignment (vary overlap length L, offset in t1 tail and
           t2 head) scored by matches - penalty(errors) - penalty(offset).
        4. Fallback: concatenate with a space.
        """
        t1 = t1.strip()
        t2 = t2.strip()

        if not t1:
            return t2
        if not t2:
            return t1

        max_scan = self.max_scan
        min_match = self.min_match_len
        max_off = self.max_off

        # 1. Exact character suffix-prefix match
        for L in range(min(len(t1), len(t2)), min_match - 1, -1):
            if t1[-L:] == t2[:L]:
                return t1 + t2[L:]

        # 2. Exact word-level suffix-prefix match
        w1 = t1.split()
        w2 = t2.split()
        for i in range(min(len(w1), len(w2)), 0, -1):
            if w1[-i:] == w2[:i]:
                return " ".join(w1[:-i] + w2)

        # 3. Fuzzy alignment: scan over overlap length L and small offsets
        best_score = -1
        best_cut1 = None   # keep t1[:best_cut1]
        best_start2 = None  # append t2[best_start2:]

        for L in range(min_match, min(len(t1), len(t2), max_scan) + 1):
            for off1 in range(0, min(max_off + 1, len(t1) - L + 1)):
                for off2 in range(0, min(max_off + 1, len(t2) - L + 1)):
                    s = t1[len(t1) - L - off1: len(t1) - off1].lower()
                    h = t2[off2: off2 + L].lower()
                    matches = sum(a == b for a, b in zip(s, h))
                    errors = L - matches
                    if errors > max(1, L // 5):
                        continue
                    # Score: reward long accurate matches, penalise offsets
                    score = matches * 5 - errors * 6 - (off1 + off2) * 2
                    if score > best_score:
                        best_score = score
                        best_cut1 = len(t1) - off1        # trim garbled tail of t1
                        best_start2 = off2 + L            # skip overlap head of t2

        if best_cut1 is not None and best_score > min_match * 2:
            return t1[:best_cut1] + t2[best_start2:]

        # 4. Fallback
        return t1 + " " + t2

    def _get_tiles(self, orig_w, orig_h, tile_size=512, overlap=128):
        tiles = []
        stride = tile_size - overlap
        
        y = 0
        while y < orig_h:
            end_y = min(orig_h, y + tile_size)
            start_y = max(0, end_y - tile_size)
            
            x = 0
            while x < orig_w:
                end_x = min(orig_w, x + tile_size)
                start_x = max(0, end_x - tile_size)
                
                tiles.append((start_x, start_y, end_x, end_y))
                if end_x == orig_w:
                    break
                x += stride
                
            if end_y == orig_h:
                break
            y += stride
            
        return tiles

    def _split_block_into_lines(self, block_gray, threshold_ratio=0.01, min_line_height=8):
        h_block, w_block = block_gray.shape
        
        # Adaptive background thresholding
        if np.mean(block_gray) > 127:
            _, binary = cv2.threshold(block_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(block_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        proj = np.sum(binary, axis=1)
        max_val = w_block * 255
        thresh_val = max(1000, max_val * threshold_ratio)
        
        in_line = False
        start_y = 0
        raw_lines = []
        
        for y in range(h_block):
            if proj[y] > thresh_val:
                if not in_line:
                    start_y = y
                    in_line = True
            else:
                if in_line:
                    end_y = y
                    if (end_y - start_y) >= min_line_height:
                        raw_lines.append((start_y, end_y))
                    in_line = False
        if in_line:
            if (h_block - start_y) >= min_line_height:
                raw_lines.append((start_y, h_block))
                
        refined_lines = []
        for s_y, e_y in raw_lines:
            s_y_pad = max(0, s_y - 2)
            e_y_pad = min(h_block, e_y + 2)
            refined_lines.append((s_y_pad, e_y_pad))
            
        return refined_lines

    def _find_text_horizontal_bounds(self, line_gray, threshold_ratio=0.02):
        h_line, w_line = line_gray.shape
        
        # Adaptive background thresholding
        if np.mean(line_gray) > 127:
            _, binary = cv2.threshold(line_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(line_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        proj = np.sum(binary, axis=0)
        thresh_val = h_line * 255 * threshold_ratio
        
        active_cols = np.where(proj > thresh_val)[0]
        if len(active_cols) == 0:
            return 0, w_line
            
        x_min = active_cols[0]
        x_max = active_cols[-1]
        
        # Add small padding to prevent character clipping
        x_min = max(0, x_min - 8)
        x_max = min(w_line, x_max + 8)
        
        return x_min, x_max

    def _are_boxes_close(self, b1, b2):
        y_dist = abs(b1['y_center'] - b2['y_center'])
        x_dist = max(0, b1['x_min'] - b2['x_max'], b2['x_min'] - b1['x_max'])
        # Allow generous horizontal gap so detector anchors (spaced every 16px
        # in 256-space, up to ~3.5× in a full-width image) don't split one text
        # line into multiple blocks.
        if y_dist < 40 and x_dist < 120:
            return True
        return False

    def run(self, img, score_threshold=0.35):
        """
        Runs the OCR pipeline on the input image.
        img: Can be a file path (str) or a numpy array (RGB or Grayscale).
        score_threshold: Float threshold for text detection.
        
        Returns a list of dicts:
        [
            {
                "box": [x_min, y_min, x_max, y_max],  # Bounding box in original image pixels
                "text": "..."  # Recognized text string
            },
            ...
        ]
        """
        # Load image if file path is provided
        if isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(f"Image file not found: {img}")
            img_bgr = cv2.imread(img)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                img_rgb = img
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError("img must be a file path (str) or numpy array")
            
        orig_h, orig_w = img_rgb.shape[:2]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Determine tiles
        if orig_w <= 512 and orig_h <= 512:
            tiles = [(0, 0, orig_w, orig_h)]
        else:
            tiles = self._get_tiles(orig_w, orig_h, tile_size=512, overlap=128)
            
        # 2. Run detector on each tile and map box predictions back to global space
        global_boxes = []
        for tx_min, ty_min, tx_max, ty_max in tiles:
            tile_w = tx_max - tx_min
            tile_h = ty_max - ty_min
            tile_crop = img_gray[ty_min:ty_max, tx_min:tx_max]
            
            # Resize tile to 256x256 and run detector
            cls_probs, dequantized = self.detector.detect_raw(tile_crop)
            
            scale_x = tile_w / 256.0
            scale_y = tile_h / 256.0
            
            for y in range(16):
                for x in range(16):
                    prob = cls_probs[y, x]
                    if prob > score_threshold:
                        vals = dequantized[0, y, x, :]
                        cy = y * 16 + 8
                        cx = x * 16 + 8
                        
                        y_center_local = cy + vals[2] * 16
                        h_local = 14.0
                        y_min_local = y_center_local - h_local / 2
                        y_max_local = y_center_local + h_local / 2
                        
                        x_min_local = cx + vals[3] * 16
                        x_max_local = cx + vals[1] * 16
                        
                        # Map to global coordinates
                        x_min_global = tx_min + x_min_local * scale_x
                        x_max_global = tx_min + x_max_local * scale_x
                        y_min_global = ty_min + y_min_local * scale_y
                        y_max_global = ty_min + y_max_local * scale_y
                        
                        global_boxes.append({
                            'x_min': x_min_global,
                            'x_max': x_max_global,
                            'y_min': y_min_global,
                            'y_max': y_max_global,
                            'y_center': (y_min_global + y_max_global) / 2
                        })
                        
        if not global_boxes:
            return []
            
        # 3. Cluster predictions into global blocks
        blocks = []
        visited = set()
        for i, box in enumerate(global_boxes):
            if i in visited:
                continue
            block = []
            queue = [i]
            visited.add(i)
            while queue:
                curr_idx = queue.pop(0)
                curr_box = global_boxes[curr_idx]
                block.append(curr_box)
                for nbr_idx, nbr_box in enumerate(global_boxes):
                    if nbr_idx not in visited:
                        if self._are_boxes_close(curr_box, nbr_box):
                            visited.add(nbr_idx)
                            queue.append(nbr_idx)
            blocks.append(block)
            
        # Sort blocks top-to-bottom
        blocks.sort(key=lambda b: min(box['y_min'] for box in b))
        
        results = []
        
        # 4. Extract lines and run OCR using horizontal chunking
        for block in blocks:
            min_x = min(box['x_min'] for box in block)
            max_x = max(box['x_max'] for box in block)
            min_y = min(box['y_min'] for box in block)
            max_y = max(box['y_max'] for box in block)
            
            # Pad block bounds slightly
            min_x = max(0, int(min_x) - 16)
            max_x = min(orig_w, int(max_x) + 16)
            min_y = max(0, int(min_y) - 8)
            max_y = min(orig_h, int(max_y) + 8)
            
            block_crop = img_gray[min_y:max_y, min_x:max_x]
            if block_crop.size == 0:
                continue
                
            line_splits = self._split_block_into_lines(block_crop)
            for y_min_rel, y_max_rel in line_splits:
                y_min_line = min_y + y_min_rel
                y_max_line = min_y + y_max_rel
                
                line_gray = img_gray[y_min_line:y_max_line, min_x:max_x]
                x_min_rel, x_max_rel = self._find_text_horizontal_bounds(line_gray)
                
                x_min_line = max(0, int(min_x + x_min_rel))
                x_max_line = min(orig_w, int(min_x + x_max_rel))
                
                # Skip empty lines
                if (x_max_line - x_min_line) <= 0 or (y_max_line - y_min_line) <= 0:
                    continue
                    
                width = x_max_line - x_min_line
                line_h = y_max_line - y_min_line

                # Compute the maximum chunk width that the recognizer can handle
                # without squishing the text horizontally.  The recognizer's
                # input tensor is (target_h × target_w); keeping aspect ratio
                # means the original-space chunk width must satisfy:
                #   chunk_w * (target_h / line_h) <= target_w
                # => chunk_w <= line_h * target_w / target_h
                # We retrieve target dimensions from the recognizer's input tensor.
                rec_shape = self.recognizer.input_details[0]['shape']  # [1, H, W, 1]
                rec_dim1, rec_dim2 = rec_shape[1], rec_shape[2]
                rec_target_h = min(rec_dim1, rec_dim2)
                rec_target_w = max(rec_dim1, rec_dim2)

                max_chunk_w = max(20, int(line_h * rec_target_w / rec_target_h) - 4)

                if self.lang in ('zh', 'ja', 'ko'):
                    overlap_ratio = 0.40
                else:
                    overlap_ratio = 0.55

                chunk_w = max_chunk_w
                overlap = int(chunk_w * overlap_ratio)
                
                if width <= chunk_w:
                    crop = img_rgb[y_min_line:y_max_line, x_min_line:x_max_line]
                    text = self.recognizer.recognize(crop)
                else:
                    chunks = []
                    curr_x = x_min_line
                    while curr_x < x_max_line:
                        end_x = min(x_max_line, curr_x + chunk_w)
                        chunks.append((curr_x, end_x))
                        if end_x == x_max_line:
                            break
                        curr_x = end_x - overlap
                        
                    text = ""
                    for cx_min, cx_max in chunks:
                        crop = img_rgb[y_min_line:y_max_line, cx_min:cx_max]
                        chunk_text = self.recognizer.recognize(crop)
                        text = self._merge_overlapping_texts(text, chunk_text)
                        
                if self.lang in ('ar', 'he'):
                    text = text[::-1]
                    
                results.append({
                    "box": [x_min_line, y_min_line, x_max_line, y_max_line],
                    "text": text
                })
                
        # Sort results top-to-bottom based on y_center of box
        results.sort(key=lambda r: (r['box'][1] + r['box'][3]) / 2)
        return results
