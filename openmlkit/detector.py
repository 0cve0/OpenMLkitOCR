# openmlkit/detector.py

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

class TextDetector:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def detect_raw(self, img_gray_256):
        """
        Runs the text detector model on a 256x256 grayscale image.
        Returns:
            cls_probs: [16, 16] probability map
            dequantized: [1, 16, 16, 4] bounding box regression offsets
        """
        if img_gray_256.shape != (256, 256):
            img_gray_256 = cv2.resize(img_gray_256, (256, 256), interpolation=cv2.INTER_LINEAR)
            
        input_data = img_gray_256.reshape((1, 256, 256, 1))
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize output
        scale, zero_point = self.output_details[0]['quantization']
        dequantized = (output_data.astype(np.float32) - zero_point) * scale
        
        cls_logits = dequantized[0, :, :, 0]
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
        
        return cls_probs, dequantized

    def detect(self, img_gray, score_threshold=0.35):
        """
        Detects text regions in a grayscale image (256x256 shape expected).
        Returns a list of bounding boxes [x_min, y_min, x_max, y_max] in normalized (0 to 256) scale.
        """
        cls_probs, dequantized = self.detect_raw(img_gray)
        
        # Step 1: Decode local boxes for active cells
        local_boxes = []
        for y in range(16):
            for x in range(16):
                prob = cls_probs[y, x]
                if prob > score_threshold:
                    vals = dequantized[0, y, x, :]
                    cy = y * 16 + 8
                    cx = x * 16 + 8
                    
                    y_center = cy + vals[2] * 16
                    # Force a tight, fixed height of 14 pixels in 256 space for text lines
                    h = 14.0
                    y_min = y_center - h / 2
                    y_max = y_center + h / 2
                    
                    x_min = cx + vals[3] * 16
                    x_max = cx + vals[1] * 16
                    
                    local_boxes.append({
                        'x_min': x_min, 'x_max': x_max,
                        'y_min': y_min, 'y_max': y_max,
                        'y_center': y_center, 'h': h
                    })
                    
        # Step 2: Group local boxes into lines
        line_groups = []
        for box in local_boxes:
            merged = False
            for group in line_groups:
                group_y_centers = [b['y_center'] for b in group]
                avg_y_center = np.mean(group_y_centers)
                
                # Check horizontal proximity
                horiz_close = False
                for g_box in group:
                    dist = max(0, box['x_min'] - g_box['x_max'], g_box['x_min'] - box['x_max'])
                    if dist < 24: # max horizontal gap in 256 space
                        horiz_close = True
                        break
                        
                if abs(box['y_center'] - avg_y_center) < 5.0 and horiz_close:
                    group.append(box)
                    merged = True
                    break
                    
            if not merged:
                line_groups.append([box])
                
        # Step 3: Compute final bounding boxes in normalized 0-256 scale
        boxes = []
        for group in line_groups:
            min_x = min(b['x_min'] for b in group)
            max_x = max(b['x_max'] for b in group)
            mean_y_min = np.mean([b['y_min'] for b in group])
            mean_y_max = np.mean([b['y_max'] for b in group])
            
            # Pad horizontally slightly to prevent character clipping (pad by 8 pixels)
            min_x = max(0, min_x - 8)
            max_x = min(256, max_x + 8)
            
            boxes.append([min_x, mean_y_min, max_x, mean_y_max])
            
        # Sort boxes top-to-bottom
        boxes.sort(key=lambda b: b[1])
        return boxes
