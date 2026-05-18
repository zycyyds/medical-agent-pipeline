# -*- coding: utf-8 -*-
"""
OCR 工具（OCR Tools）

使用 RapidOCR（本地离线）进行图像文字识别。
若已安装 onnxruntime-gpu 且当前环境可用 CUDA，则自动使用 GPU（检测/方向/识别均走 CUDA EP）。
设置环境变量 RAPIDOCR_USE_GPU=0 可强制只用 CPU。

工具列表：
  - ocr_image        对单张图片进行 OCR，返回提取文本
  - preprocess_text  清洗医学文本（去噪、规范化符号）
  - ocr_and_clean    OCR + 清洗一步完成
"""
import os
import re
import sys
from typing import Any, Dict

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 模块级单例，避免重复加载模型
_engine = None


def _want_rapidocr_gpu() -> bool:
    flag = os.environ.get("RAPIDOCR_USE_GPU", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    try:
        import onnxruntime as ort

        return (
            "CUDAExecutionProvider" in ort.get_available_providers()
            and ort.get_device() == "GPU"
        )
    except Exception:
        return False


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        if _want_rapidocr_gpu():
            _engine = RapidOCR(
                det_use_cuda=True,
                cls_use_cuda=True,
                rec_use_cuda=True,
            )
        else:
            _engine = RapidOCR()
    return _engine


def ocr_image(image_path: str) -> Dict[str, Any]:
    """
    对图片文件进行 OCR 识别，提取原始文本。

    Args:
        image_path: 图片文件路径（jpg/png/tiff 等）

    Returns:
        {
          "success": bool,
          "text": str,
          "char_count": int,
          "error": str | None,
        }
    """
    if not os.path.exists(image_path):
        return {"success": False, "text": "", "char_count": 0,
                "error": f"文件不存在: {image_path}"}
    try:
        engine = _get_engine()
        result, _ = engine(image_path)
        if not result:
            return {"success": True, "text": "", "char_count": 0, "error": None}
        # result: [[[坐标], 文本, 置信度], ...]，按行拼接
        text = "\n".join(item[1] for item in result if len(item) >= 2)
        return {"success": True, "text": text, "char_count": len(text), "error": None}
    except Exception as e:
        return {"success": False, "text": "", "char_count": 0, "error": str(e)}


def preprocess_text(text: str) -> Dict[str, Any]:
    """
    对医学文本进行清洗：去除多余空白、规范化标点符号。

    Args:
        text: 原始文本字符串

    Returns:
        {
          "success": bool,
          "cleaned_text": str,
          "original_length": int,
          "cleaned_length": int,
          "error": str | None,
        }
    """
    if not text or not text.strip():
        return {"success": False, "cleaned_text": "", "original_length": 0,
                "cleaned_length": 0, "error": "输入文本为空"}
    try:
        cleaned = re.sub(r"\r\n|\r", "\n", text)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        return {
            "success": True,
            "cleaned_text": cleaned,
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "error": None,
        }
    except Exception as e:
        return {
            "success": True,
            "cleaned_text": text,
            "original_length": len(text),
            "cleaned_length": len(text),
            "error": str(e),
        }


def ocr_and_clean(image_path: str) -> Dict[str, Any]:
    """
    对图片执行 OCR 识别，然后对结果进行文本清洗（两步合一）。

    Args:
        image_path: 图片文件路径

    Returns:
        {
          "success": bool,
          "ocr_text": str,
          "cleaned_text": str,
          "char_count": int,
          "error": str | None,
        }
    """
    ocr = ocr_image(image_path)
    if not ocr["success"]:
        return {"success": False, "ocr_text": "", "cleaned_text": "",
                "char_count": 0, "error": ocr["error"]}
    clean = preprocess_text(ocr["text"])
    return {
        "success": True,
        "ocr_text": ocr["text"],
        "cleaned_text": clean["cleaned_text"],
        "char_count": clean["cleaned_length"],
        "error": clean.get("error"),
    }
