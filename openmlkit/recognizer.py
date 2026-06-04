# openmlkit/recognizer.py

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

class TextRecognizer:
    def __init__(self, model_path, label_map):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.label_map = label_map
        
    def recognize(self, crop_img):
        """
        Recognizes the text in a cropped image (can be color or grayscale).
        Returns the decoded text string.
        """
        if len(crop_img.shape) == 3:
            crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        else:
            crop_gray = crop_img.copy()
            
        # Determine model layout and target dimensions
        shape = self.input_details[0]['shape']
        dim1, dim2 = shape[1], shape[2]
        
        if dim1 > dim2:
            # Model expects [batch, width, height, channels] layout
            target_w = dim1
            target_h = dim2
            transpose = True
        else:
            # Model expects [batch, height, width, channels] layout
            target_h = dim1
            target_w = dim2
            transpose = False

        # Preserve aspect ratio: scale by height, pad width with background.
        # Directly squishing wide text into target_w causes garbled output.
        h_src, w_src = crop_gray.shape
        scale = target_h / h_src
        new_w = int(round(w_src * scale))

        if new_w <= target_w:
            # Scale to target height, then right-pad with background colour
            interp_flag = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(crop_gray, (new_w, target_h), interpolation=interp_flag)
            # Detect background colour (most frequent of corners)
            corners = [crop_gray[0, 0], crop_gray[0, -1],
                       crop_gray[-1, 0], crop_gray[-1, -1]]
            bg = int(np.median(corners))
            canvas = np.full((target_h, target_w), bg, dtype=np.uint8)
            canvas[:, :new_w] = resized
            crop_resized = canvas
        else:
            # Chunk is still wider than the model window — squish as last resort
            crop_resized = cv2.resize(crop_gray, (target_w, target_h),
                                      interpolation=cv2.INTER_AREA)

        # Reshape to standard height-width layout
        input_data = crop_resized.reshape((1, target_h, target_w, 1))

        if transpose:
            input_data = np.transpose(input_data, (0, 2, 1, 3))
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Dynamically locate the 3D output tensor (CTC logits)
        output_detail = None
        for out in self.output_details:
            if len(out['shape']) == 3:
                output_detail = out
                break
        if output_detail is None:
            output_detail = self.output_details[0]
            
        output_data = self.interpreter.get_tensor(output_detail['index'])
        
        # Dequantize output
        scale, zero_point = output_detail['quantization']
        dequantized = (output_data.astype(np.float32) - zero_point) * scale
        
        # Decode text
        return self._ctc_decode(dequantized)
        
    def _ctc_decode(self, output_tensor):
        # output_tensor shape: [1, 42, V]
        logits = output_tensor[0]
        best_paths = np.argmax(logits, axis=-1)
        
        decoded_indices = []
        prev_idx = -1
        
        # CTC blank token: conventionally the last class index (V - 1)
        V = output_tensor.shape[-1]
        blank_token = V - 1
        
        for idx in best_paths:
            if idx != blank_token:
                if idx != prev_idx:
                    decoded_indices.append(idx)
            prev_idx = idx
            
        chars = [self.label_map.get(idx, "") for idx in decoded_indices]
        return "".join(chars)
