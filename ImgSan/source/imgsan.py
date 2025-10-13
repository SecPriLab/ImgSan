import os, math, base64

import cv2
import json
import torch
import torchvision
import numpy as np
import torch.nn as nn
import supervision as sv
from pathlib import Path
import torchvision.transforms as T
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image, ImageOps
from pwlf import PiecewiseLinFit
from pycocotools.coco import COCO
from IPython.display import display
from torch.nn import functional as F
from groundingdino.util.inference import Model
from utils.mask_hook_logger import hook_prs_logger
from utils.factory import create_model_and_transforms, get_tokenizer
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class HeatmapSimilarityCNN(nn.Module):
    def __init__(self):
        super(HeatmapSimilarityCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.intermediate_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, orig, pert, return_embedding=False):
        x = torch.cat((orig, pert), dim=1)
        feat64 = self.feature_extractor(x)
        feat16 = self.intermediate_fc(feat64)
        if return_embedding:
            return feat16
        out = self.classifier(feat16).squeeze(1)
        return out


class HeatmapMaskCNN(nn.Module):
    def __init__(self):
        super(HeatmapMaskCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, heatmap, mask):
        x = torch.stack([heatmap, mask], dim=1)
        features = self.conv_layers(x)
        output = self.fc_layers(features)
        return output.squeeze(1)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "PrivacyVQA2K"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"
OUTPUT_DIR = PROJECT_ROOT / "ImgSan" / "sanitized_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OBF_INT_DIR = PROJECT_ROOT / "ImgSan" / "evaluation" / "obfuscation_intensities"
OBF_INT_DIR.mkdir(parents=True, exist_ok=True)

BLUR_PRIVACY = True
BLUR_PRIVACY_STRENGTH = 5
SEGMENTION_MODEL_TYPE = "vit_l"
SEGMETION_MODEL_PATH = PROJECT_ROOT / "ImgSan" / "models" / "sam_vit_l.pth"
BILINEAR_PROCESS = True
BOX_NMS_THRESH = 0.2
ENHANCE_COE_FOR_GEN_IMPORTANT = 15
VISUAL_MODEL_NAME = "ViT-L-14-336"
PRETRAINED = 'openai'
GROUDING_MODEL_PATH = PROJECT_ROOT / "ImgSan" / "models" / "GroundingDINO" / "groundingdino_swinb_cogcoor.pth"
GROUDING_MODEL_CONFIG_PATH = PROJECT_ROOT / "ImgSan" / "models" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py"
DISTINGUISH_MODEL = PROJECT_ROOT / "ImgSan" / "models" / "discriminator.pth"
MAP_WS_MODEL_FOLDER_PATH = PROJECT_ROOT / "ImgSan" / "models" / "estimators"
MAP_WS_TUPLES = [(0, 13), (24, 36)]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

visual_model, _, preprocess = create_model_and_transforms(VISUAL_MODEL_NAME, pretrained=PRETRAINED)
visual_model.to(DEVICE)
visual_model.eval()
tokenizer = get_tokenizer(VISUAL_MODEL_NAME)

grounding_dino_model = Model(
    model_config_path=str(GROUDING_MODEL_CONFIG_PATH),
    model_checkpoint_path=str(GROUDING_MODEL_PATH)
)

segmention_model = sam_model_registry[SEGMENTION_MODEL_TYPE](checkpoint=str(SEGMETION_MODEL_PATH)).to(device=DEVICE)

distinguish_model = HeatmapSimilarityCNN().to(DEVICE)
state_dict = torch.load(DISTINGUISH_MODEL, map_location=DEVICE)
distinguish_model.load_state_dict(state_dict)
distinguish_model.eval()

map_ws_models = []
for model_path in sorted(os.listdir(MAP_WS_MODEL_FOLDER_PATH)):
    full_model_path = MAP_WS_MODEL_FOLDER_PATH / model_path
    map_ws_model = HeatmapMaskCNN().to(DEVICE)
    state_dict = torch.load(full_model_path, map_location=DEVICE)
    map_ws_model.load_state_dict(state_dict)
    map_ws_model.eval()
    map_ws_models.append(map_ws_model)

class ImgSan:
    def __init__(
        self,
        visual_model,
        tokenizer,
        segmention_model,
        grounding_model,
        distinguish_model,
        map_ws_models,
        map_ws_tuples,
        device,
        layer_index=23
    ):
        self.device = device
        self.visual_model = visual_model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.prs = hook_prs_logger(visual_model, self.device, layer_index)
        self.segmention_model = segmention_model
        self.grounding_model = grounding_model
        self.distinguish_model = distinguish_model
        self.map_ws_models = map_ws_models
        self.map_ws_tuples = map_ws_tuples

    def gen_images(self, image_paths):
        image_originals = []
        for image_path in image_paths:
            image_origin = Image.open(image_path)
            image_origin = ImageOps.exif_transpose(image_origin).convert("RGB")
            image_originals.append(image_origin)
            return image_originals

    def gen_heatmaps(
        self,
        image_list,
        questions,
        kernel_size=3,
        enhance_coe=10,
        use_conv=True,
        use_for_distinguish=False
    ):
        images_processeds = []
        for image in image_list:
            image_processed = self.preprocess(image)[np.newaxis, :, :, :]
            images_processeds.append(image_processed)
        images = torch.cat(images_processeds, dim=0).to(self.device)
        self.prs.reinit()
        with torch.no_grad():
            representation = self.visual_model.encode_image(
                images,
                attn_method='head',
                normalize=False
            )
            attentions, mlps = self.prs.finalize(representation)
            questions = questions if isinstance(questions, list) else [questions]
            questions_tokenizeds = self.tokenizer(questions).to(self.device)  # tokenize
            questions_embeddings = self.visual_model.encode_text(questions_tokenizeds)
            questions_embeddings = F.normalize(questions_embeddings, dim=-1)
            cls_attentions = attentions[:, 0, 1:, :]
            cls_similarities = torch.einsum('bnd,bd->bn', cls_attentions, questions_embeddings)
            HW = int(np.sqrt(cls_similarities.shape[1]))
            batch_size = cls_similarities.shape[0]
            cls_similarities = cls_similarities.view(batch_size, HW, HW)
            patch_similarities = torch.einsum('bnd,bd->bn', mlps[:, 0, :, :], questions_embeddings)
            patch_similarities = patch_similarities.view(batch_size, HW, HW)
            heatmaps = []
            for cls_similarity, patch_similarity, image in zip(
                cls_similarities,
                patch_similarities,
                image_list
            ):
                cls_similarity = self.normalize(cls_similarity, "min")
                if enhance_coe != 0:
                    cls_similarity = self.enhance(cls_similarity, coe=enhance_coe)
                patch_similarity = self.normalize(patch_similarity, "max")
                if use_conv:
                    assert kernel_size % 2 == 1
                    padding_size = int((kernel_size - 1) / 2)
                    conv = torch.nn.Conv2d(
                        1, 1,
                        kernel_size=kernel_size,
                        padding=padding_size,
                        padding_mode="replicate",
                        stride=1,
                        bias=False
                    )
                    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size ** 2
                    conv.to(self.device)
                    cls_similarity = conv(cls_similarity.unsqueeze(0))[0]
                    patch_similarity = conv(patch_similarity.unsqueeze(0))[0]
                if not use_for_distinguish:
                    cls_similarity = self.normalize(cls_similarity, "min")
                heatmaps.append(cls_similarity)

        return heatmaps

    def normalize(self, mat, method="max"):
        if method == "max":
            return (mat.max() - mat) / (mat.max() - mat.min())
        elif method == "min":
            return (mat - mat.min()) / (mat.max() - mat.min())
        else:
            raise NotImplementedError

    def enhance(self, mat, coe=10):
        mat = mat - mat.mean()
        mat = mat / mat.std()
        mat = mat * coe
        mat = torch.sigmoid(mat)
        mat = mat.clamp(0, 1)
        return mat

    def expand_heatmap(self, heatmap, image):
        heatmap = heatmap.squeeze(0).squeeze(0)
        W, H = image.size
        if BILINEAR_PROCESS:
            expanded_heatmap = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear'
            )
        else:
            expanded_heatmap = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='nearest'
            )
        return expanded_heatmap.squeeze(0).squeeze(0)

    def gen_annotations(self, image, **kwargs):
        image = np.array(image)
        annotations_generator = SamAutomaticMaskGenerator(self.segmention_model, **kwargs)
        annotations = annotations_generator.generate(image)
        sorted_annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        return sorted_annotations

    def filter_small_segmentations(self, annotations, threshold=0.001):
        threshold_area = threshold * annotations[0]['segmentation'].shape[0] * annotations[0]['segmentation'].shape[1]
        filtered_annotations = [
            annotation for annotation in annotations if annotation['area'] >= threshold_area
        ]
        return filtered_annotations

    def filter_by_coverage(self, annotations, coverage_thresh=0.95):
        if len(annotations) == 0:
            return []
        filtered = [annotations[0]]
        combined_mask = annotations[0]['segmentation'].copy()
        for i in range(1, len(annotations)):
            ann = annotations[i]
            seg = ann['segmentation']
            area = ann['area']
            intersection = np.logical_and(seg, combined_mask)
            intersection_area = np.sum(intersection)
            if intersection_area / max(area, 1) >= coverage_thresh:
                continue
            else:
                filtered.append(ann)
                combined_mask = np.logical_or(combined_mask, seg)
        return filtered

    def process_segmentations(self, annotations, threshold=0, dig=False):
        if dig:
            annotations = self.filter_by_coverage(annotations)
        if threshold > 0:
            annotations = self.filter_small_segmentations(annotations, threshold)
        return annotations

    def fill_none_sorted_desc(self, arr):
        if None not in arr:
            return arr
        idx = arr.index(None)
        left = arr[idx - 1] if idx > 0 else None
        right = arr[idx + 1] if idx < len(arr) - 1 else None
        if left is not None and right is not None:
            arr[idx] = (left + right) / 2
        elif left is not None:
            arr[idx] = left * 0.8
        elif right is not None:
            arr[idx] = right * 1.2
        return arr

    def gen_split_mask(self, heatmap, annotations, swap_mask_threshold=0.05):
        for current_annotation in annotations:
            current_seg = current_annotation['segmentation']
            crucial_area_importance = heatmap[current_seg].mean().cpu().numpy()
            current_annotation['importance'] = crucial_area_importance
        sorted_annotations = sorted(annotations, key=lambda x: x['importance'], reverse=True)
        split_annotations = self.regression_split(sorted_annotations)
        if swap_mask_threshold > 0:
            required_area = swap_mask_threshold * annotations[0]['segmentation'].shape[0] * annotations[0]['segmentation'].shape[1]
            while sum(annotation['area'] for annotation in split_annotations[0]) < required_area and len(split_annotations[1]) > 0:
                split_annotations[0].append(split_annotations[1].pop(0))
        masks = []
        importances = []
        for annotations in split_annotations:
            if len(annotations) == 0:
                mask = np.full((heatmap.shape[0], heatmap.shape[1]), False, dtype=bool)
                importance = None
            else:
                mask = self.merge_mask([annotation['segmentation'] for annotation in annotations])
                importance = np.array([annotation['importance'] for annotation in annotations]).mean()
            masks.append(mask)
            importances.append(importance)
        return masks, self.fill_none_sorted_desc(importances)

    def regression_split(self, annotations, split_num=3):
        if len(annotations) == 0:
            return
        x = np.arange(len(annotations))
        y = np.array([annotation['importance'] for annotation in annotations])
        split_annotations = []
        if len(annotations) >= 3 or len(annotations) == 1:
            if len(annotations) >= 3:
                model = PiecewiseLinFit(x, y)
                breaks = model.fit(split_num)
            else:
                breaks = [0.0, 0.0, 0.0, 0.0]
            for i in range(len(breaks) - 1):
                start = math.ceil(breaks[i])
                end = math.ceil(breaks[i + 1])
                split_annotations.append(
                    annotations[start:end if end != len(y) - 1 else end + 1]
                )
        else:
            split_annotations.append(annotations[0:1])
            split_annotations.append(annotations[1:1])
            split_annotations.append(annotations[1:2])
        return split_annotations

    def merge_mask(self, masks):
        if len(masks) == 0:
            return
        combined_mask = np.logical_or.reduce(masks)
        return combined_mask

    def map_ws(self, heatmap, masks):
        strengths = [0]
        with torch.no_grad():
            heatmap_resized = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(32, 32),
                mode='nearest'
            ).squeeze(0)
            for mask, model, scope in zip(masks[1:], self.map_ws_models, self.map_ws_tuples):
                mask = torch.from_numpy(mask.astype(np.float32)).to(self.device)
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(32, 32),
                    mode='nearest'
                ).squeeze(0)
                output = model(heatmap_resized, mask_resized)
                ws = int(round(
                    (output * (scope[1] - scope[0]) + scope[0])
                    .clamp(scope[0], scope[1])
                    .cpu().numpy()[0]
                ))
                strengths.append(ws)
        return strengths

    def impaint_masks(self, image, questions, masks, important_privacy_mask, strengths, perturb_way):
        strengths[1] = 9
        perturb_img = np.array(image)
        img_list = []
        privacy_array = np.zeros(perturb_img.shape[:2], dtype=np.uint8)

        for strength in strengths:
            cur_img_array = self.ws_blur(strength, np.array(image), perturb_way)
            img_list.append((cur_img_array, strength))

        if BLUR_PRIVACY:
            sam_found_mask = self.merge_mask(masks)
            sam_not_found_mask = ~sam_found_mask
            sam_not_found_img = self.ws_blur(9, np.array(image), perturb_way)
            perturb_img[sam_not_found_mask] = sam_not_found_img[sam_not_found_mask]
            privacy_array[sam_not_found_mask] = BLUR_PRIVACY_STRENGTH

        for (img, strength), mask in zip(reversed(img_list), reversed(masks)):
            perturb_img[mask] = img[mask]
            privacy_array[mask] = strength

        if important_privacy_mask is not None:
            choice_strengths = [0, 2, 4, 7, 12, 20, 33]
            choice_imgs = [perturb_img.copy() for _ in choice_strengths]
            privacy_arrays = [privacy_array.copy() for _ in choice_strengths]
            for index, ws in enumerate(choice_strengths):
                choice_perturb_img = self.ws_blur(ws, np.array(image), perturb_way)
                choice_imgs[index][important_privacy_mask] = choice_perturb_img[important_privacy_mask]
                privacy_arrays[index][important_privacy_mask] = ws
                choice_img, choice_index = self.choose_under_usable_most_privacy_img(
                    questions, image, choice_imgs
                )
                return choice_img, privacy_arrays[choice_index]
        else:
            return perturb_img, privacy_array

    def choose_under_usable_most_privacy_img(self, questions, origin_img, choice_imgs):
        orgin_heatmap = self.gen_heatmaps(
            [origin_img],
            questions,
            kernel_size=3,
            enhance_coe=25,
            use_conv=False,
            use_for_distinguish=True
        )[0].unsqueeze(0).unsqueeze(0)

        usable_img = choice_imgs[0]
        choice_index = 0

        for index, choice_img in enumerate(choice_imgs):
            perturbed_heatmap = self.gen_heatmaps(
                [Image.fromarray(choice_img)],
                questions,
                kernel_size=3,
                enhance_coe=25,
                use_conv=False,
                use_for_distinguish=True
            )[0].unsqueeze(0).unsqueeze(0)
            outputs = self.distinguish_model(orgin_heatmap, perturbed_heatmap)
            prediction = (outputs > 0.5).int().cpu().numpy()[0]
            if prediction == 1:
                choice_index = index
                usable_img = choice_img
            else:
                break

        return usable_img, choice_index

    def gen_privacy_masks(
        self,
        image,
        privacy_items,
        box_threshold=0.5,
        text_threshold=0.5,
        nms_threshold=0.5
    ):
        image_np = np.array(image)
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        all_detections = None
        all_labels = None

        for privacy_item in privacy_items:
            detections, labels = self.grounding_model.predict_with_caption(
                image=image,
                caption=privacy_item,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            if detections:
                if all_detections is None and len(labels) != 0:
                    all_detections = sv.Detections(xyxy=detections.xyxy, confidence=detections.confidence)
                    all_labels = labels
                else:
                    all_detections = sv.Detections(
                        xyxy=np.concatenate([all_detections.xyxy, detections.xyxy], axis=0),
                        confidence=np.concatenate([all_detections.confidence, detections.confidence], axis=0),
                    )
                    all_labels = all_labels + labels

        if all_detections is None:
            return [np.full((image_np.shape[0], image_np.shape[1]), False, dtype=bool)], image_np

        class_mapping = {cls: idx for idx, cls in enumerate(set(all_labels))}
        class_ids = np.array([class_mapping[cls] for cls in all_labels])
        all_detections.class_id = class_ids

        nms_idx = torchvision.ops.nms(
            torch.from_numpy(all_detections.xyxy),
            torch.from_numpy(all_detections.confidence),
            nms_threshold
        ).numpy().tolist()

        all_detections.xyxy = all_detections.xyxy[nms_idx]
        all_detections.confidence = all_detections.confidence[nms_idx]
        all_detections.class_id = all_detections.class_id[nms_idx]
        all_labels = [all_labels[i] for i in nms_idx]

        sorted_indices = sorted(
            range(len(all_detections.confidence)),
            key=lambda i: all_detections.confidence[i],
            reverse=True
        )
        all_detections.xyxy = all_detections.xyxy[sorted_indices]
        all_detections.confidence = all_detections.confidence[sorted_indices]
        all_detections.class_id = all_detections.class_id[sorted_indices]
        all_labels = [all_labels[i] for i in sorted_indices]

        all_detections.xyxy = all_detections.xyxy[:1]
        all_detections.confidence = all_detections.confidence[:1]
        all_detections.class_id = all_detections.class_id[:1]
        all_labels = all_labels[:1]

        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [f"{cls}: {conf:.2f}" for cls, conf in zip(all_labels, all_detections.confidence)]

        annotated_image = box_annotator.annotate(scene=image, detections=all_detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=all_detections, labels=labels)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        sam_predictor = SamPredictor(self.segmention_model)
        sam_predictor.set_image(image)

        result_masks = []
        for box in all_detections.xyxy:
            masks, scores, _ = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])

        return result_masks, annotated_image

    def gen_important_privacy_mask(self, masks, privacy_masks, impainting_threshold=0.2):
        privacy_mask = self.merge_mask(privacy_masks)
        important_mask = masks[0]
        important_privacy_mask = np.logical_and(important_mask, privacy_mask)

        total_important_pixels = np.sum(important_mask)
        total_important_privacy_pixels = np.sum(important_privacy_mask)
        ratio = (
            total_important_privacy_pixels / total_important_pixels
            if total_important_pixels > 0 else 0
        )

        if ratio < impainting_threshold:
            masks[0] = np.logical_and(masks[0], np.logical_not(important_privacy_mask))
            masks[-1] = np.logical_or(masks[-1], important_privacy_mask)
            important_privacy_mask = None

        return masks, important_privacy_mask

    def ws_blur(self, ws, img_np, mode):
        img_np = img_np.copy()
        if ws == 0:
            return img_np

        h, w = img_np.shape[:2]
        rows = np.arange(0, h, ws)
        cols = np.arange(0, w, ws)

        for i in rows:
            for j in cols:
                h_end = min(i + ws, h)
                w_end = min(j + ws, w)
                window = img_np[i:h_end, j:w_end]
                if mode == "shuffle":
                    flat_window = window.reshape(-1, 3)
                    shuffled_idx = np.random.permutation(len(flat_window))
                    flat_window[:] = flat_window[shuffled_idx]
                    img_np[i:h_end, j:w_end] = flat_window.reshape(h_end - i, w_end - j, 3)
                elif mode == "mean":
                    mean_pixel = window.mean(axis=(0, 1)).astype(int)
                    img_np[i:h_end, j:w_end] = mean_pixel
                else:
                    raise ValueError(
                        f"Unsupported mode: {mode}. Choose from bilatseg, 'shuffle' or 'mean'."
                    )

        return img_np

with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

imgsan = ImgSan(
    visual_model=visual_model,
    tokenizer=tokenizer,
    segmention_model=segmention_model,
    grounding_model=grounding_dino_model,
    distinguish_model=distinguish_model,
    map_ws_models=map_ws_models,
    map_ws_tuples=MAP_WS_TUPLES,
    device=DEVICE
)

# You can customize the privacy-sensitive objects to be detected here.
privacy_items = [
    "condom", "newspaper", "pregnancy test", "receipt", "phone screen",
    "student ID", "driver's license", "QR code", "credit card", "pill bottle",
    "passport", "license plate", "face"
]

for item in tqdm(data):
    image_paths = [os.path.join(IMAGES_DIR, item["image"])]
    image_list = imgsan.gen_images(image_paths)
    annotations = imgsan.gen_annotations(
        image_list[0],
        pred_iou_thresh=0.96,
        stability_score_thresh=0.96,
        box_nms_thresh=BOX_NMS_THRESH
    )
    annotations = imgsan.process_segmentations(annotations, threshold=0.00001, dig=True)
    privacy_masks, _ = imgsan.gen_privacy_masks(image_list[0], privacy_items)
    questions = item["qas_en"][0]
    heatmaps = imgsan.gen_heatmaps(
        image_list,
        questions,
        kernel_size=3,
        enhance_coe=ENHANCE_COE_FOR_GEN_IMPORTANT,
        use_conv=True,
        use_for_distinguish=False
    )
    res_masks, importances = imgsan.gen_split_mask(
        imgsan.expand_heatmap(heatmaps[0], image_list[0]), annotations
    )
    res_masks, important_privacy_mask = imgsan.gen_important_privacy_mask(res_masks, privacy_masks)
    strengths = imgsan.map_ws(heatmaps[0], res_masks)
    choice_img, privacy_array = imgsan.impaint_masks(
        image_list[0], questions, res_masks, important_privacy_mask, strengths, perturb_way="mean"
    )
    img = Image.fromarray(choice_img)
    img.save(os.path.join(OUTPUT_DIR, item["image"]))
    np.save(os.path.join(OBF_INT_DIR, f"{item['image'].split('.', 1)[0]}.npy"), privacy_array)