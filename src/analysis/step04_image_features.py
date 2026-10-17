"""
Step 4: 封面图像特征提取 (优化版 v2)
研究题目: 标题和封面双模态协同对社交媒体平台视频传播的影响——以 B 站知识类视频为例

性能优化要点:
  1. CLIP 文本向量预计算一次 (最大收益: 原来每张图×11类 → 现在只算1次)
  2. CLIP 批量图像推理 (CLIP_BATCH_SIZE 可调, GPU 建议 128)
  3. 基础视觉 + OCR + 人物检测 用 ThreadPoolExecutor 并行
  4. EasyOCR 快速熵预筛 (熵极低=无文字, 直接跳过)
  5. 所有模型只加载一次, 主进程共享

Windows 非 ASCII 路径修复:
  所有图像读取均通过 PIL.Image.open() → numpy array,
  规避 cv2.imread() 在 Windows 下遇到汉字路径返回 None 的问题。

依赖包:
  pip install pillow opencv-python-headless scikit-image numpy pandas
              tqdm openpyxl snownlp easyocr torch torchvision
  CLIP:  pip install git+https://github.com/openai/CLIP.git
  可选:  pip install ultralytics   (YOLOv8 人物检测)
         pip install mediapipe      (人脸检测)
"""

# ── 标准库 ────────────────────────────────────────────────────────────────────
import os, sys, math, time, logging, warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── 第三方基础库 ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2

# ── 路径配置 ───────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(".")
DATA_DIR  = ROOT_DIR / "outputs"
COVER_DIR = ROOT_DIR / "covers"
OUT_DIR   = ROOT_DIR / "outputs"
FIG_DIR   = OUT_DIR  / "figures"
LOG_DIR   = OUT_DIR  / "logs"
for d in [OUT_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 日志 ───────────────────────────────────────────────────────────────────────
log_path = LOG_DIR / f"step4_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("step4")

# ══════════════════════════════════════════════════════════════════════════════
# 可调参数
# ══════════════════════════════════════════════════════════════════════════════
INPUT_CSV           = DATA_DIR / "propagation_metrics.csv"
CLIP_THRESHOLD      = 0.25   # CLIP 零样本二值化阈值
CLIP_BATCH_SIZE     = 32     # CLIP 批量推理大小 (GPU 可调到 64-128)
N_WORKERS           = 4      # 并行线程数 (基础视觉+OCR+人物)
OCR_ENTROPY_SKIP    = 1.5    # 灰度熵低于此值时跳过 OCR (几乎无文字)
FACE_CONF_THRESHOLD = 0.5
SENTIMENT_MODEL     = "snownlp"


# ══════════════════════════════════════════════════════════════════════════════
# §0  图像读取工具
# ══════════════════════════════════════════════════════════════════════════════

def safe_open_image(img_path) -> Image.Image:
    """PIL 打开图像 → RGB。规避 cv2.imread 的 Windows 汉字路径问题。"""
    return Image.open(img_path).convert("RGB")

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def pil_to_gray(pil_img):
    return np.array(pil_img.convert("L"))


# ══════════════════════════════════════════════════════════════════════════════
# §1  基础视觉特征 (纯 CPU)
# ══════════════════════════════════════════════════════════════════════════════

def extract_basic_visual(pil_img: Image.Image) -> dict:
    w, h    = pil_img.size
    img_arr = np.array(pil_img)
    gray    = pil_to_gray(pil_img)
    bgr     = pil_to_bgr(pil_img)
    hsv     = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    R = img_arr[:,:,0].astype(float)
    G = img_arr[:,:,1].astype(float)
    B = img_arr[:,:,2].astype(float)
    rg = R - G
    yb = 0.5*(R+G) - B
    colorfulness = float(
        math.sqrt(np.std(rg)**2 + np.std(yb)**2)
        + 0.3*math.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    )
    mean_r, mean_b = float(np.mean(R)), float(np.mean(B))
    edges        = cv2.Canny(bgr, 100, 200)
    edge_density = float(np.sum(edges>0) / (w*h))

    image_entropy = float(sum(
        -np.sum((p := np.bincount(img_arr[:,:,c].flatten(), minlength=256)/(w*h))
                * np.log2(p + 1e-12))
        for c in range(3)
    ) / 3)

    small    = pil_img.resize((64,64))
    n_colors = len(set(small.getdata())) / (64*64)

    def mm(v, lo, hi):
        return max(0.0, min(1.0, (v-lo)/(hi-lo+1e-8)))

    return dict(
        image_width       = w,
        image_height      = h,
        image_size        = w*h,
        image_brightness  = float(np.mean(gray)),
        image_contrast    = float(np.std(gray)),
        image_saturation  = float(np.mean(hsv[:,:,1])),
        image_colorfulness= colorfulness,
        image_warm_cold   = (mean_r-mean_b)/(mean_r+mean_b+1e-8),
        edge_density      = edge_density,
        image_entropy     = image_entropy,
        visual_complexity = float(np.mean([mm(edge_density,0,0.15),
                                           mm(image_entropy,0,8),
                                           mm(n_colors,0,1)])),
    )


# ══════════════════════════════════════════════════════════════════════════════
# §2  OCR 文字特征
# ══════════════════════════════════════════════════════════════════════════════

def _load_ocr_engine():
    try:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        logger.info("OCR 引擎: PaddleOCR")
        return engine, "paddleocr"
    except Exception as e:
        logger.warning(f"PaddleOCR 不可用: {e}")
    try:
        import easyocr
        # gpu=False 保持稳定; 若有 GPU 可改为 True
        engine = easyocr.Reader(["ch_sim","en"], gpu=False, verbose=False)
        logger.info("OCR 引擎: EasyOCR (fallback)")
        return engine, "easyocr"
    except Exception as e:
        logger.warning(f"EasyOCR 不可用: {e}")
    logger.error("所有 OCR 引擎不可用")
    return None, "none"


def _ocr_paddle(engine, img_path) -> list:
    img_arr = np.array(Image.open(img_path).convert("RGB"))
    results = engine.ocr(img_arr, cls=True)
    boxes = []
    if results and results[0]:
        for line in results[0]:
            bbox, (text, conf) = line
            boxes.append({"text": text, "confidence": conf, "bbox": bbox})
    return boxes


def _ocr_easy(engine, img_path) -> list:
    img_arr = np.array(Image.open(img_path).convert("RGB"))
    results = engine.readtext(img_arr, detail=1)
    return [{"text":t,"confidence":c,"bbox":b} for b,t,c in results]


def _quick_entropy(gray: np.ndarray) -> float:
    p = np.bincount(gray.flatten(), minlength=256).astype(float) / gray.size
    return float(-np.sum(p * np.log2(p+1e-12)))


def _text_sentiment(text: str):
    if not text or not text.strip():
        return np.nan, np.nan
    try:
        from snownlp import SnowNLP
        s = SnowNLP(text)
        score     = float(s.sentiments)
        intensity = abs(score-0.5)*2
        return score, intensity
    except Exception:
        return np.nan, np.nan


def extract_ocr_features(img_path, pil_img, ocr_engine, ocr_engine_name) -> dict:
    null = dict(cover_text="", cover_has_text=0, cover_text_len=0,
                cover_text_box_count=0, cover_text_area_ratio=0.0,
                cover_text_punctuation_count=0,
                cover_text_sentiment_score=np.nan,
                cover_text_sentiment_intensity=np.nan)

    if ocr_engine is None or ocr_engine_name == "none":
        return null

    # 快速熵预筛 (熵低 → 背景色块 → 无文字, 直接跳过)
    gray = pil_to_gray(pil_img)
    if _quick_entropy(gray) < OCR_ENTROPY_SKIP:
        return null

    try:
        boxes = _ocr_paddle(ocr_engine, img_path) \
                if ocr_engine_name=="paddleocr" else _ocr_easy(ocr_engine, img_path)
    except Exception as e:
        logger.warning(f"OCR 失败 {Path(img_path).name}: {e}")
        return null

    if not boxes:
        return null

    w, h      = pil_img.size
    full_text = "".join(b["text"] for b in boxes)
    PUNCT     = set('，。！？、；：""''（）【】《》…—～·,.!?;:\'"()[]{}<>')

    total_box_area = 0.0
    for b in boxes:
        pts = np.array(b["bbox"], dtype=float)
        xs, ys = pts[:,0], pts[:,1]
        total_box_area += (xs.max()-xs.min())*(ys.max()-ys.min())

    sent_score, sent_intensity = _text_sentiment(full_text)

    return dict(
        cover_text                     = full_text,
        cover_has_text                 = int(len(full_text.strip())>0),
        cover_text_len                 = len(full_text),
        cover_text_box_count           = len(boxes),
        cover_text_area_ratio          = min(1.0, total_box_area/(w*h+1e-8)),
        cover_text_punctuation_count   = sum(1 for c in full_text if c in PUNCT),
        cover_text_sentiment_score     = sent_score,
        cover_text_sentiment_intensity = sent_intensity,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §3  人物 / 人脸检测
# ══════════════════════════════════════════════════════════════════════════════

def _load_human_detector():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        logger.info("人物检测: YOLOv8n")
        return model, "yolov8"
    except Exception as e:
        logger.warning(f"YOLOv8 不可用: {e}")
    try:
        import mediapipe as mp
        model = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=FACE_CONF_THRESHOLD)
        logger.info("人脸检测: mediapipe (fallback)")
        return model, "mediapipe"
    except Exception as e:
        logger.warning(f"mediapipe 不可用: {e}")
    try:
        cascade_path = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty(): raise RuntimeError("加载失败")
        logger.info("人脸检测: OpenCV Haar (fallback)")
        return cascade, "haar"
    except Exception as e:
        logger.warning(f"Haar 不可用: {e}")
    logger.error("所有人物/人脸检测不可用")
    return None, "none"


def extract_human_features(pil_img, model, model_name) -> dict:
    null = dict(human_presence=np.nan, human_count=np.nan,
                face_presence=np.nan,  face_count=np.nan)
    if model is None or model_name=="none":
        return null
    img_arr = np.array(pil_img)
    try:
        if model_name=="yolov8":
            res = model(img_arr, verbose=False)[0]
            n   = int(np.sum(res.boxes.cls.cpu().numpy().astype(int)==0))
            return dict(human_presence=int(n>0),human_count=n,
                        face_presence=int(n>0),face_count=n)
        elif model_name=="mediapipe":
            res = model.process(img_arr)
            n   = len(res.detections) if res.detections else 0
            return dict(human_presence=int(n>0),human_count=n,
                        face_presence=int(n>0),face_count=n)
        elif model_name=="haar":
            gray  = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            faces = model.detectMultiScale(gray,1.1,5,minSize=(30,30))
            n     = len(faces)
            return dict(human_presence=int(n>0),human_count=n,
                        face_presence=int(n>0),face_count=n)
    except Exception as e:
        logger.warning(f"人物检测失败: {e}")
    return null


# ══════════════════════════════════════════════════════════════════════════════
# §4  CLIP 推理器 ← 核心优化: 预计算文本向量 + 批量图像推理
# ══════════════════════════════════════════════════════════════════════════════

# 零样本分类提示词
CLIP_CATEGORIES = {
    "cover_is_slide":       (["PPT slide, presentation slide, lecture notes, whiteboard with text"],
                             ["photo","person portrait"]),
    "cover_has_chart":      (["bar chart, line chart, pie chart, data visualization, infographic"],
                             ["plain text","person"]),
    "cover_has_formula":    (["mathematical formula, equation, LaTeX, scientific notation"],
                             ["photo","cartoon"]),
    "cover_has_experiment": (["laboratory, scientific experiment, lab equipment, test tube"],
                             ["cartoon","slide"]),
    "cover_has_object":     (["product photo, physical object, real-world item"],
                             ["cartoon","formula"]),
    "cover_is_screenshot":  (["mobile screenshot, computer screen capture, UI interface"],
                             ["outdoor","person portrait"]),
    "cover_acgn_style":     (["anime, manga, cartoon illustration, ACGN style"],
                             ["real photo","formula"]),
    "cover_text_only":      (["text-only image, pure typography, big text no background"],
                             ["photo","person","chart"]),
    "cover_real_photo":     (["real photograph, realistic photo, natural scene"],
                             ["cartoon","anime","slide"]),
    "cover_cartoon_style":  (["cartoon illustration, flat design, vector art, comic style"],
                             ["real photo","formula"]),
    # 内部情感键 (以 _ 开头, 不输出为特征)
    "_sent_positive":       (["happy, excited, inspiring, cheerful, enthusiastic"],
                             ["sad","fearful","angry"]),
    "_sent_negative":       (["scary, shocking, warning, alarming, serious, threatening"],
                             ["happy","cheerful"]),
    "_sent_neutral":        (["neutral, calm, informational, documentary, educational"],
                             ["exciting","scary"]),
}


class CLIPInferencer:
    """
    关键优化:
      - _precompute_text(): 把所有提示词文本向量一次性算好, 存为矩阵
      - score_batch():       给一批图像向量打分, 一次矩阵乘法搞定全部类别

    原版耗时 ≈ N_img × N_cat 次 encode_text + N_img 次 encode_image
    优化后   ≈ 1 次 encode_text + ceil(N_img/B) 次批量 encode_image
    """

    def __init__(self):
        self.model      = None
        self.preprocess = None
        self.device     = None
        self.clip_type  = "none"
        self._text_vecs  = None   # 预计算文本向量 (n_prompts, d)
        self._cat_slices = {}     # cat → (pos_slice, neg_slice)

    def load(self):
        try:
            import torch
            import clip as openai_clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = openai_clip.load("ViT-B/32", device=device)
            self.model, self.preprocess = model, preprocess
            self.device, self.clip_type = device, "openai_clip"
            logger.info(f"CLIP 加载成功, device={device}")
            self._precompute_text()
            return True
        except Exception as e:
            logger.warning(f"OpenAI CLIP 不可用: {e}")

        try:
            import torch
            from transformers import ChineseCLIPProcessor, ChineseCLIPModel
            name  = "OFA-Sys/chinese-clip-vit-base-patch16"
            model = ChineseCLIPModel.from_pretrained(name)
            prep  = ChineseCLIPProcessor.from_pretrained(name)
            dev   = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = model.to(dev), prep
            self.device, self.clip_type = dev, "chinese_clip"
            logger.info(f"Chinese-CLIP 加载成功, device={dev}")
            self._precompute_text()
            return True
        except Exception as e:
            logger.warning(f"Chinese-CLIP 不可用: {e}")

        logger.error("CLIP 不可用")
        return False

    def _precompute_text(self):
        """把所有提示词拼成一个列表, 一次性编码成向量矩阵。"""
        import torch
        all_prompts, cat_slices = [], {}
        for cat, (pos, neg) in CLIP_CATEGORIES.items():
            ps = slice(len(all_prompts), len(all_prompts)+len(pos))
            all_prompts.extend(pos)
            ns = slice(len(all_prompts), len(all_prompts)+len(neg))
            all_prompts.extend(neg)
            cat_slices[cat] = (ps, ns)
        self._cat_slices = cat_slices

        if self.clip_type == "openai_clip":
            import clip as openai_clip
            tokens = openai_clip.tokenize(all_prompts, truncate=True).to(self.device)
            with torch.no_grad():
                vecs = self.model.encode_text(tokens).float()
                vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        else:
            inputs = self.preprocess(
                text=all_prompts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                vecs = self.model.get_text_features(**inputs).float()
                vecs = vecs / vecs.norm(dim=-1, keepdim=True)

        self._text_vecs = vecs   # stays on device
        logger.info(f"CLIP 文本向量预计算完成: {len(all_prompts)} 条提示词, shape={tuple(vecs.shape)}")

    def encode_images_batch(self, pil_imgs: list):
        """批量编码图像, 返回归一化向量 tensor (batch, d)。"""
        import torch
        if self.clip_type == "openai_clip":
            tensors = torch.stack([self.preprocess(img) for img in pil_imgs]).to(self.device)
            with torch.no_grad():
                vecs = self.model.encode_image(tensors).float()
        else:
            inputs = self.preprocess(images=pil_imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                vecs = self.model.get_image_features(**inputs).float()
        return vecs / vecs.norm(dim=-1, keepdim=True)

    def score_batch(self, img_vecs) -> list:
        """
        一次矩阵乘法给整个 batch 打分。
        img_vecs: tensor (batch, d) on device
        返回: list of dicts, 每个 dict 包含该图的全部 CLIP 特征
        """
        # sims: (batch, n_prompts)
        sims = (img_vecs @ self._text_vecs.T).cpu().numpy()

        results = []
        for row in sims:
            # softmax 归一
            e = np.exp(row - row.max())
            p = e / e.sum()

            cat_scores = {cat: float(np.mean(p[ps]))
                          for cat, (ps, _) in self._cat_slices.items()}

            feat = {}
            for cat in CLIP_CATEGORIES:
                if cat.startswith("_"):
                    continue
                s = cat_scores[cat]
                feat[f"{cat}_conf"] = s
                feat[cat]           = int(s >= CLIP_THRESHOLD)

            feat["cover_has_chart_or_formula"]   = max(feat.get("cover_has_chart",0),
                                                       feat.get("cover_has_formula",0))
            feat["cover_is_slide_or_screenshot"] = max(feat.get("cover_is_slide",0),
                                                       feat.get("cover_is_screenshot",0))
            sp = cat_scores.get("_sent_positive", np.nan)
            sn = cat_scores.get("_sent_negative", np.nan)
            sm = cat_scores.get("_sent_neutral",  np.nan)
            feat["_clip_sentiment_score"]     = float(sp)
            feat["_clip_sentiment_valence"]   = float(sp-sn)
            feat["_clip_sentiment_intensity"] = float(1-sm)
            results.append(feat)

        return results


# ══════════════════════════════════════════════════════════════════════════════
# §5  封面综合情感
# ══════════════════════════════════════════════════════════════════════════════

def build_cover_sentiment(ocr_feat: dict, clip_feat: dict) -> dict:
    ov = ocr_feat.get("cover_text_sentiment_score",  np.nan)
    cs = clip_feat.get("_clip_sentiment_score",      np.nan)
    cv = clip_feat.get("_clip_sentiment_valence",    np.nan)
    ci = clip_feat.get("_clip_sentiment_intensity",  np.nan)
    ov_ok = isinstance(ov, float) and not np.isnan(ov)
    cs_ok = isinstance(cs, float) and not np.isnan(cs)

    if ov_ok and cs_ok:
        score, valence = 0.5*ov+0.5*cs, cv
        intensity = max(abs(ov-0.5)*2, ci)
    elif ov_ok:
        score, valence, intensity = ov, (ov-0.5)*2, abs(ov-0.5)*2
    elif cs_ok:
        score, valence, intensity = cs, cv, ci
    else:
        score = valence = intensity = np.nan

    return dict(cover_sentiment_score    = score,
                cover_sentiment_valence  = valence,
                cover_sentiment_intensity= intensity)


# ══════════════════════════════════════════════════════════════════════════════
# §6  单张图片工作函数 (线程池调用 — 非 CLIP 部分)
# ══════════════════════════════════════════════════════════════════════════════

def _process_single(args):
    """基础视觉 + OCR + 人物检测。CLIP 单独批量处理。"""
    row, ocr_engine, ocr_engine_name, human_model, human_model_name = args
    bvid     = row["bvid"]
    img_path = COVER_DIR / str(row["domain"]) / str(row["level"]) / f"{bvid}.jpg"
    rec      = {"bvid": bvid, "_img_path": str(img_path)}

    if not img_path.exists():
        return bvid, rec, {"bvid":bvid,"reason":"file_not_found","path":str(img_path)}

    try:
        pil_img = safe_open_image(img_path)
    except Exception as e:
        return bvid, rec, {"bvid":bvid,"reason":str(e),"path":str(img_path)}

    try:    rec.update(extract_basic_visual(pil_img))
    except Exception as e: logger.debug(f"基础视觉失败 {bvid}: {e}")

    try:    rec.update(extract_ocr_features(img_path, pil_img, ocr_engine, ocr_engine_name))
    except Exception as e: logger.debug(f"OCR 失败 {bvid}: {e}")

    try:    rec.update(extract_human_features(pil_img, human_model, human_model_name))
    except Exception as e: logger.debug(f"人物失败 {bvid}: {e}")

    return bvid, rec, None


# ══════════════════════════════════════════════════════════════════════════════
# §7  主流程
# ══════════════════════════════════════════════════════════════════════════════

def run():
    t0 = time.time()
    logger.info(f"加载输入数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    logger.info(f"共 {len(df)} 条记录")

    valid_domains = {"科学科普","社科人文","财经商业","职场教育"}
    unexpected = set(df["domain"].unique()) - valid_domains
    if unexpected:
        logger.warning(f"未预期 domain: {unexpected}")

    # ── 加载所有模型 ──────────────────────────────────────────────────────────
    ocr_engine,  ocr_engine_name  = _load_ocr_engine()
    human_model, human_model_name = _load_human_detector()
    clip_inf = CLIPInferencer()
    clip_inf.load()
    clip_type = clip_inf.clip_type

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 1: 多线程处理基础视觉 + OCR + 人物
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"[阶段1] 基础视觉/OCR/人物 — 线程数={N_WORKERS}")
    rows_list = [row for _, row in df.iterrows()]
    args_list = [(r, ocr_engine, ocr_engine_name, human_model, human_model_name)
                 for r in rows_list]

    bvid_to_rec: dict = {}
    damaged:     list = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_process_single, a): a[0]["bvid"] for a in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="基础视觉+OCR+人物"):
            bvid, rec, dmg = fut.result()
            bvid_to_rec[bvid] = rec
            if dmg:
                damaged.append(dmg)

    logger.info(f"[阶段1] 完成 {time.time()-t0:.1f}s, 损坏图片={len(damaged)}")

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 2: CLIP 批量推理 (文本向量已预计算好, 只需逐批 encode_image)
    # ══════════════════════════════════════════════════════════════════════════
    clip_results: dict = {}
    if clip_type != "none":
        logger.info(f"[阶段2] CLIP 批量推理 batch_size={CLIP_BATCH_SIZE}")
        t1 = time.time()

        # 收集有效图片
        valid_items = []
        for bvid, rec in bvid_to_rec.items():
            p = Path(rec["_img_path"])
            if not p.exists():
                continue
            try:
                valid_items.append((bvid, safe_open_image(p)))
            except Exception:
                pass

        for i in tqdm(range(0, len(valid_items), CLIP_BATCH_SIZE), desc="CLIP 批量推理"):
            batch  = valid_items[i:i+CLIP_BATCH_SIZE]
            bvids  = [x[0] for x in batch]
            imgs   = [x[1] for x in batch]
            try:
                vecs  = clip_inf.encode_images_batch(imgs)
                feats = clip_inf.score_batch(vecs)
                for bvid, feat in zip(bvids, feats):
                    clip_results[bvid] = feat
            except Exception as e:
                logger.warning(f"CLIP 批次失败 {bvids[0]}...: {e}")

        logger.info(f"[阶段2] CLIP 完成 {time.time()-t1:.1f}s")
    else:
        logger.warning("[阶段2] CLIP 不可用, 跳过")

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段 3: 合并特征
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("[阶段3] 合并特征...")
    all_records = []
    for _, row in df.iterrows():
        bvid = row["bvid"]
        rec  = bvid_to_rec.get(bvid, {"bvid": bvid})
        rec.pop("_img_path", None)
        cf   = clip_results.get(bvid, {})
        rec.update(cf)
        rec.update(build_cover_sentiment(rec, cf))
        all_records.append(rec)

    feat_df = pd.DataFrame(all_records)

    # ── 输出文件 ──────────────────────────────────────────────────────────────
    basic_cols = ["bvid","image_width","image_height","image_size",
                  "image_brightness","image_contrast","image_saturation",
                  "image_colorfulness","image_warm_cold","edge_density",
                  "image_entropy","visual_complexity",
                  "human_presence","human_count","face_presence","face_count",
                  "cover_sentiment_score","cover_sentiment_valence",
                  "cover_sentiment_intensity"]
    ocr_cols   = ["bvid","cover_text","cover_has_text","cover_text_len",
                  "cover_text_box_count","cover_text_area_ratio",
                  "cover_text_punctuation_count",
                  "cover_text_sentiment_score","cover_text_sentiment_intensity"]

    feat_df[[c for c in basic_cols if c in feat_df.columns]].to_csv(
        OUT_DIR/"image_features.csv", index=False, encoding="utf-8-sig")
    feat_df[[c for c in ocr_cols if c in feat_df.columns]].to_csv(
        OUT_DIR/"ocr_features.csv", index=False, encoding="utf-8-sig")
    logger.info("image_features.csv / ocr_features.csv 已保存")

    pd.DataFrame(damaged).to_csv(LOG_DIR/"damaged_images.csv",
                                  index=False, encoding="utf-8-sig")

    # OCR 抽检
    if "cover_has_text" in feat_df.columns:
        ocr_valid = feat_df[feat_df["cover_has_text"]==1]
        n = min(100, len(ocr_valid))
        if n:
            ocr_valid[[c for c in ocr_cols if c in feat_df.columns]] \
                .sample(n=n, random_state=42) \
                .to_excel(LOG_DIR/"cover_ocr_audit_samples.xlsx", index=False)

    # CLIP 分类抽检
    clip_cats = [c for c in feat_df.columns
                 if c in ("cover_is_slide","cover_has_chart","cover_has_formula",
                           "cover_has_experiment","cover_has_object",
                           "cover_is_screenshot","cover_acgn_style",
                           "cover_text_only","cover_real_photo","cover_cartoon_style")]
    frames = []
    for cat in clip_cats:
        pos = feat_df[feat_df[cat]==1]
        n   = min(20, len(pos))
        if n:
            s = pos.sample(n=n, random_state=42).copy()
            s["audit_category"] = cat
            frames.append(s)
    if frames:
        pd.concat(frames).to_excel(LOG_DIR/"cover_visual_category_audit.xlsx", index=False)

    # 变量说明表
    var_rows = [
        ("image_width","图像宽度","像素","基础视觉"),
        ("image_height","图像高度","像素","基础视觉"),
        ("image_size","图像面积","像素²","基础视觉"),
        ("image_brightness","封面亮度","灰度0-255均值","基础视觉"),
        ("image_contrast","封面对比度","灰度标准差","基础视觉"),
        ("image_saturation","封面饱和度","HSV-S均值","基础视觉"),
        ("image_colorfulness","色彩丰富度","Hasler-Süsstrunk公式","基础视觉"),
        ("image_warm_cold","暖冷色倾向","-1冷~+1暖","基础视觉"),
        ("edge_density","边缘密度","Canny边缘像素比","基础视觉"),
        ("image_entropy","图像熵","Shannon熵","基础视觉"),
        ("visual_complexity","视觉复杂度","综合指数0-1","基础视觉"),
        ("cover_text","封面OCR文本","字符串","OCR文字"),
        ("cover_has_text","是否含文字","0/1","OCR文字"),
        ("cover_text_len","文字字符数","整数","OCR文字"),
        ("cover_text_box_count","文字框数量","整数","OCR文字"),
        ("cover_text_area_ratio","文字框面积占比","0-1","OCR文字"),
        ("cover_text_punctuation_count","标点数量","整数","OCR文字"),
        ("cover_text_sentiment_score","封面文字情感","0-1(SnowNLP)","OCR情感"),
        ("cover_text_sentiment_intensity","文字情感强度","0-1","OCR情感"),
        ("human_presence","是否含人物","0/1","人物人脸"),
        ("human_count","人物数量","整数","人物人脸"),
        ("face_presence","是否含人脸","0/1","人物人脸"),
        ("face_count","人脸数量","整数","人物人脸"),
        ("cover_is_slide","PPT/讲义式","0/1","CLIP知识"),
        ("cover_has_chart","含图表","0/1","CLIP知识"),
        ("cover_has_formula","含公式","0/1","CLIP知识"),
        ("cover_has_experiment","含实验场景","0/1","CLIP知识"),
        ("cover_has_object","含实物对象","0/1","CLIP知识"),
        ("cover_is_screenshot","截图式","0/1","CLIP知识"),
        ("cover_acgn_style","ACGN风格","0/1","CLIP知识"),
        ("cover_text_only","纯文字封面","0/1","CLIP知识"),
        ("cover_real_photo","真实照片式","0/1","CLIP知识"),
        ("cover_cartoon_style","卡通插画式","0/1","CLIP知识"),
        ("cover_has_chart_or_formula","含图表或公式","0/1","CLIP知识"),
        ("cover_is_slide_or_screenshot","PPT或截图式","0/1","CLIP知识"),
        ("cover_sentiment_score","封面综合情感","0-1","封面情感"),
        ("cover_sentiment_valence","情感效价","-1负~+1正","封面情感"),
        ("cover_sentiment_intensity","情感强度","0-1","封面情感"),
    ]
    var_dict = pd.DataFrame(var_rows, columns=["variable_name","chinese_alias","unit","category"])
    model_info = pd.DataFrame([
        {"item":"ocr_engine",        "value": ocr_engine_name},
        {"item":"human_det",         "value": human_model_name},
        {"item":"clip_model",        "value": clip_type},
        {"item":"clip_threshold",    "value": CLIP_THRESHOLD},
        {"item":"clip_batch_size",   "value": CLIP_BATCH_SIZE},
        {"item":"n_workers",         "value": N_WORKERS},
        {"item":"ocr_entropy_skip",  "value": OCR_ENTROPY_SKIP},
        {"item":"run_time",          "value": datetime.now().isoformat()},
        {"item":"python",            "value": sys.version},
    ])
    with pd.ExcelWriter(OUT_DIR/"variable_dictionary_step4.xlsx", engine="openpyxl") as w:
        var_dict.to_excel(w,   sheet_name="变量说明",  index=False)
        model_info.to_excel(w, sheet_name="模型信息",  index=False)

    # 质量报告
    num_cols = [c for c in feat_df.columns
                if c not in ("bvid","cover_text")
                and pd.api.types.is_numeric_dtype(feat_df[c])]
    qa = feat_df[num_cols].agg(["count","mean","std","min","max"]).T
    qa["missing_n"]   = len(feat_df) - qa["count"]
    qa["missing_pct"] = (qa["missing_n"] / len(feat_df) * 100).round(2)
    qa.to_excel(OUT_DIR/"quality_report_step4.xlsx")

    elapsed = time.time() - t0
    logger.info(f"Step 4 全部完成 ✓  总耗时 {elapsed/60:.1f} 分钟")
    return feat_df


if __name__ == "__main__":
    run()