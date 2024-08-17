classes = [
    'person',
    'hard-hat',
    'gloves',
    'mask',
    'glasses',
    'boots',
    'vest',
    'ppe-suit',
    'ear-protector',
    'safety-harness'
]

class_colors = {
    'person': (255, 0, 0), # Red
    'hard-hat': (0, 255, 0), # Green
    'gloves': (0, 0, 255), # Blue
    'mask': (255, 255, 0), # Cyan
    'glasses': (255, 0, 255), # Magenta
    'boots': (0, 255, 255), # Yellow
    'vest': (128, 0, 128), # Purple
    'ppe-suit': (0, 128, 128), # Teal
    'ear-protector': (128, 128, 0), # Olive
    'safety-harness': (255, 165, 0) # Orange
}

ppe_max_count = {
    'person': 1,
    'hard-hat': 1,
    'gloves': 2,
    'mask': 1,
    'glasses': 1,
    'boots': 2,
    'vest': 1,
    'ppe-suit': 1,
    'ear-protector': 2,
    'safety-harness': 1
}
