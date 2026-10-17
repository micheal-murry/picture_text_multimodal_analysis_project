# 封面样例

本目录仅保留 **12 张封面样例**（四个知识领域各 3 张），用于展示原始数据形态与封面特征的处理对象。

- 全量封面约 **4129 张、1GB**，受版权与仓库体积限制不随仓库分发，已在 `.gitignore` 中排除；
- 封面视觉特征（亮度、色彩、复杂度、OCR 文字、人脸等）**成品结果已提交**于
  `outputs/image_features.csv`、`outputs/ocr_features.csv`，无需图片即可完成全部分析；
- 如需从 0 复现步骤 4 的图像特征，请先运行 `src/scraper/download_covers.py` 下载全量封面
  （公开接口、礼貌限速），再执行 `python src/analysis/step04_image_features.py`，
  依赖见 `requirements-features.txt`。

文件命名规则：`{领域}_{BV号}.jpg`。
