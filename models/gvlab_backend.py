import os

import clip
import torch
from PIL import Image

from config import IMAGES_FOLDER_PATH

device = "cuda" # if torch.cuda.is_available() else "cpu"

class BackendModel:

    def __init__(self):
        self.clip_version = 'ViT-B/32' # 1 x 512
        self.model, self.preprocess = clip.load(self.clip_version, device=device)
        self.images_directory = IMAGES_FOLDER_PATH
        self.image_dim = self.load_and_encode_img('acid.jpg').shape[1]
        self.text_dim = self.encode_text('acid').shape[1]

    def load_and_encode_img(self, img):
        img_path = os.path.join(self.images_directory, img)
        img_rgb = Image.open(img_path).convert('RGB')
        clip_image = self.preprocess(img_rgb).unsqueeze(0).to(device)
        clip_image_feats = self.model.encode_image(clip_image)
        clip_image_feats /= clip_image_feats.norm(dim=-1, keepdim=True)
        return clip_image_feats

    def encode_text(self, text):
        clip_text = self.get_clip_txt(text)
        cue_clip_txt_encoded = self.model.encode_text(clip_text)
        cue_clip_txt_encoded /= cue_clip_txt_encoded.norm(dim=-1, keepdim=True)
        return cue_clip_txt_encoded # looks like 1 x 512

    def get_clip_txt(self, text):
        text = text.lower()
        vowels = ["a", "e", "i", "o", "u"]
        if any(text.startswith(x) for x in vowels):
            clip_txt = f"An {text}"
        else:
            clip_txt = f"A {text}"
        clip_txt_tokenized = clip.tokenize([clip_txt]).to(device)
        return clip_txt_tokenized

BackendModel()