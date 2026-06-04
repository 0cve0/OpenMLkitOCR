# openmlkit/labelmap.py

class LabelMap:
    def __init__(self, pb_path):
        self.mapping = self._parse_pb(pb_path)
        
    def _skip_field(self, data, idx, wire_type):
        if wire_type == 0:  # Varint
            while True:
                b = data[idx]
                idx += 1
                if not (b & 0x80):
                    break
        elif wire_type == 1:  # 64-bit
            idx += 8
        elif wire_type == 2:  # Length-delimited
            flen = 0
            shift = 0
            while True:
                b = data[idx]
                idx += 1
                flen |= (b & 0x7f) << shift
                if not (b & 0x80):
                    break
                shift += 7
            idx += flen
        elif wire_type == 5:  # 32-bit
            idx += 4
        return idx

    def _parse_pb(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        
        idx = 0
        total_len = len(data)
        label_map = {}
        
        while idx < total_len:
            tag_byte = data[idx]
            tag = tag_byte >> 3
            wire = tag_byte & 0x07
            idx += 1
            
            # Read varint value (used for length or integer value)
            val = 0
            shift = 0
            while True:
                b = data[idx]
                idx += 1
                val |= (b & 0x7f) << shift
                if not (b & 0x80):
                    break
                shift += 7
                
            if tag == 1 and wire == 2:
                # Length-delimited Entry message
                end_idx = idx + val
                char_str = ""
                class_idx = 0
                while idx < end_idx:
                    inner_tag_byte = data[idx]
                    inner_tag = inner_tag_byte >> 3
                    inner_wire = inner_tag_byte & 0x07
                    idx += 1
                    
                    if inner_tag == 1 and inner_wire == 2:
                        char_len = 0
                        shift = 0
                        while True:
                            b = data[idx]
                            idx += 1
                            char_len |= (b & 0x7f) << shift
                            if not (b & 0x80):
                                break
                            shift += 7
                        char_str = data[idx:idx+char_len].decode('utf-8', errors='replace')
                        idx += char_len
                    elif inner_tag == 2 and inner_wire == 0:
                        class_val = 0
                        shift = 0
                        while True:
                            b = data[idx]
                            idx += 1
                            class_val |= (b & 0x7f) << shift
                            if not (b & 0x80):
                                break
                            shift += 7
                        class_idx = class_val
                    else:
                        idx = self._skip_field(data, idx, inner_wire)
                label_map[class_idx] = char_str
            else:
                # Skip other top-level tags
                if wire == 1:
                    idx += 8
                elif wire == 2:
                    idx += val
                elif wire == 5:
                    idx += 4
                
        return label_map

    def get(self, index, default=""):
        return self.mapping.get(index, default)
