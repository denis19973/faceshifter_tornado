import time

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from faceshifter.face_modules.model import Backbone
from faceshifter.face_modules.mtcnn import MTCNN
from faceshifter.network.aei import *
from web.settings import ARCFACE_MODEL_PATH, GENERATOR_MODEL_PATH


def initialize_inference_models(device):
    device = torch.device(device)
    G = AEI_Net(c_id=512)
    G = G.to(device)
    G.eval()
    G.load_state_dict(torch.load(GENERATOR_MODEL_PATH, map_location=device), strict=False)
    detector = MTCNN()
    arcface_model = Backbone(50, 0.6, 'ir_se').to(device)
    arcface_model.eval()
    arcface_model.load_state_dict(torch.load(ARCFACE_MODEL_PATH, map_location=device), strict=False)
    return arcface_model, detector, G


def swap_faces(arcface_model, detector, G, device, Xs_raw, Xt_raw):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    Xs_img = Image.fromarray(Xs_raw)
    Xt_img = Image.fromarray(Xt_raw)
    Xs = detector.align(Xs_img, crop_size=(64, 64))
    Xt, trans_inv = detector.align(Xt_img, crop_size=(64, 64), return_trans_inv=True)

    source_face = Xs is not None
    target_face = Xt is not None
    if not source_face or not target_face:
        return source_face, target_face, None

    Xs = test_transform(Xs)
    Xs = Xs.unsqueeze(0).to(device)
    with torch.no_grad():
        embeds, Xs_feats = arcface_model(F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True))
        embeds = embeds.mean(dim=0, keepdim=True)

    mask = np.zeros([64, 64], dtype=np.float)
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i - 32) ** 2 + (j - 32) ** 2) / 32
            dist = np.minimum(dist, 1)
            mask[i, j] = 1 - dist
    mask = cv2.dilate(mask, None, iterations=20)

    Xt = test_transform(Xt)
    Xt = Xt.unsqueeze(0).to(device)

    with torch.no_grad():
        st = time.time()
        Yt, _ = G(Xt, embeds)
        Yt = Yt.squeeze().detach().cpu().numpy()
        st = time.time() - st
        print(f'inference time: {st} sec')
        Yt = Yt.transpose([1, 2, 0]) * 0.5 + 0.5
        Yt_trans_inv = cv2.warpAffine(
            Yt,
            trans_inv,
            (np.size(Xt_raw, 1), np.size(Xt_raw, 0)),
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        mask_ = cv2.warpAffine(
            mask,
            trans_inv,
            (np.size(Xt_raw, 1), np.size(Xt_raw, 0)),
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        mask_ = np.expand_dims(mask_, 2)
        merge = mask_ * Yt_trans_inv + (1 - mask_) * (Xt_raw.astype(np.float) / 255.)
        return source_face, target_face, merge
