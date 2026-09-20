import cv2
import numpy as np

from skimage import img_as_float

from scipy.signal import convolve2d
from scipy import special

import torch
import piq

import os

import json


# ============= Optional Gemini Planner Setup =============

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

if load_dotenv is not None:
    load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_PLANNER_MODEL = os.getenv("GEMINI_PLANNER_MODEL")


def get_gemini_client():
    """
    Create a Gemini client only when the planner branch is explicitly used.
    This keeps deterministic GIRA functional even without Gemini installed/configured.
    """
    if genai is None or types is None:
        raise RuntimeError(
            "Gemini planner is unavailable. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install google-genai"
        )

    if not GOOGLE_API_KEY:
        raise RuntimeError("GEMINI_API_KEY was not found in the environment or .env file.")

    if not GEMINI_PLANNER_MODEL:
        raise RuntimeError("GEMINI_PLANNER_MODEL was not found in the environment or .env file.")

    return genai.Client(api_key=GOOGLE_API_KEY)

# ============= 1. Image I/O & Grayscale =============

def load_image_rgb(path):
    """Read image from disk and return RGB float32 in [0, 1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return img_as_float(rgb).astype(np.float32)


def to_gray(rgb):
    """Convert RGB float32 [0, 1] to grayscale float32 [0, 1]."""
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    return gray.astype(np.float32) / 255.0



# ============= 2. NIQE proxy (MSCN + AGGD) =============

def compute_mscn(gray):
    """Compute MSCN coefficients for NIQE-like statistics."""
    kernel = np.array(
        [[1, 2, 3, 2, 1],
         [2, 4, 6, 4, 2],
         [3, 6, 8, 6, 3],
         [2, 4, 6, 4, 2],
         [1, 2, 3, 2, 1]],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    mu = convolve2d(gray, kernel, mode="same")
    sigma = np.sqrt(np.abs(convolve2d(gray * gray, kernel, mode="same") - mu * mu))
    return (gray - mu) / (sigma + 1e-6)


def estimate_aggd_beta(vec):
    """Estimate AGGD parameters used in NIQE-like computation."""
    gam = np.arange(0.2, 10, 0.001)
    r_gam = (special.gamma(2 / gam) ** 2) / (
        special.gamma(1 / gam) * special.gamma(3 / gam)
    )
    r_hat = (np.mean(np.abs(vec)) ** 2) / np.mean(vec ** 2)
    gamma = gam[np.argmin((r_gam - r_hat) ** 2)]

    sigma_l = np.sqrt(((vec[vec < 0]) ** 2).mean()) if np.any(vec < 0) else 0.0
    sigma_r = np.sqrt(((vec[vec > 0]) ** 2).mean()) if np.any(vec > 0) else 0.0
    return gamma, sigma_l, sigma_r


def compute_niqe(img_rgb):
    """Simplified NIQE proxy (higher is worse)."""
    gray = to_gray(img_rgb)
    mscn = compute_mscn(gray).flatten().astype(np.float32)

    if mscn.size == 0:
        return float("inf")

    gamma, sigma_l, sigma_r = estimate_aggd_beta(mscn)

    if gamma == 0:
        return float("inf")

    return float((sigma_l + sigma_r) / gamma)

# ============= 3. BRISQUE (via piq) =============

def compute_brisque_score(img_rgb):
    """Compute BRISQUE score with piq (lower is better)."""
    img_t = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float()

    try:
        with torch.no_grad():
            score = piq.brisque(img_t, data_range=1.0).item()

        return float(score)

    except Exception as e:
        print("BRISQUE failed:", e)
        return float("inf")
    

# ============= 4. IQA Features & Global Quality Score =============

def compute_basic_iqa_features(img_rgb):
    """Compute basic hand-crafted IQA features."""
    gray = to_gray(img_rgb)
    h, w = gray.shape[:2]

    brightness_mean = gray.mean()
    brightness_std = gray.std()
    p_dark = float(np.mean(gray < 30 / 255.0))
    p_bright = float(np.mean(gray > 225 / 255.0))

    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
    lap_var = float(lap.var())

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_var = float((gray - blur).var())

    return {
        "height": float(h),
        "width": float(w),
        "brightness_mean": float(brightness_mean),
        "brightness_std": float(brightness_std),
        "p_dark": p_dark,
        "p_bright": p_bright,
        "contrast_std": float(brightness_std),
        "laplacian_var": lap_var,
        "noise_var": noise_var,
    }


def extract_iqa_features_from_rgb(img_rgb):
    """Compute basic features + NIQE + BRISQUE."""
    feats = compute_basic_iqa_features(img_rgb)
    feats["niqe"] = compute_niqe(img_rgb)
    feats["brisque"] = compute_brisque_score(img_rgb)
    return feats


def extract_iqa_features_from_path(path):
    """Convenience wrapper: load image then extract IQA features."""
    return extract_iqa_features_from_rgb(load_image_rgb(path))


def normalize_feature(x, min_val, max_val, invert=False):
    """Normalize scalar feature to [0, 1], with optional inversion."""
    x_norm = (x - min_val) / (max_val - min_val + 1e-8)
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    return 1.0 - x_norm if invert else x_norm


def compute_quality_score(feats, w_contrast=0.4, w_sharpness=0.4, w_niqe=0.2):
    """
    Compute a single scalar quality score Q (higher is better):

        Q = w_contrast * norm(contrast) +
            w_sharpness * norm(sharpness) -
            w_niqe * norm(NIQE)
    """
    contrast = feats["contrast_std"]
    sharp = feats["laplacian_var"]
    niqe_val = feats["niqe"]

    c_n = normalize_feature(contrast, 0.0, 0.25)
    s_n = normalize_feature(sharp, 0.0, 300.0)
    n_n = normalize_feature(niqe_val, 0.0, 20.0)

    Q = w_contrast * c_n + w_sharpness * s_n - w_niqe * n_n

    return float(Q), {
        "contrast_norm": c_n,
        "sharpness_norm": s_n,
        "niqe_norm": n_n,
    }



# ============= 5. Rule-based Diagnosis (SV/RS) =============

def diagnose_image_from_path(img_path, source_type=None):
    """
    Diagnose degradations from an image path.

    - Infer source_type (SV / RS) if not given.
    - Return: source_type, boolean problem flags, and IQA features.
    """
    feats = extract_iqa_features_from_path(img_path)
    full_lower = img_path.lower()
    ext = os.path.splitext(img_path)[1].lower()

    # HERMES-safe normalization:
    # GPA usually returns view_types as "sv" or "rs".
    if isinstance(source_type, str):
        source_type = source_type.strip().upper()

    # Infer source type if not specified
    if source_type in ["SV", "RS"]:
        st = source_type
    elif "satellite" in full_lower or "rsi_" in full_lower:
        st = "RS"
    elif ext in [".tif", ".tiff"]:
        st = "RS"
    elif ext in [".jpg", ".jpeg", ".png"]:
        st = "SV"
    else:
        st = "SV"

    probs = {
        "low_light": False,
        "over_exposed": False,
        "low_contrast": False,
        "blur": False,
        "high_noise": False,
        "haze_like": False,
        "backlight": False,
        "low_resolution": False,
        "high_cloud": False,
    }

    f = feats

    # Global thresholds
    if (f["brightness_mean"] < 0.5) or (f["p_dark"] > 0.2):
        probs["low_light"] = True

    if f["p_bright"] > 0.4 and f["brightness_mean"] > 0.7:
        probs["over_exposed"] = True

    if f["contrast_std"] < 0.05:
        probs["low_contrast"] = True

    if f["laplacian_var"] < 20:
        probs["blur"] = True

    if f["noise_var"] > 0.002:
        probs["high_noise"] = True

    if (
        0.3 < f["brightness_mean"] < 0.7
        and f["contrast_std"] < 0.06
        and f["niqe"] > 6
    ):
        probs["haze_like"] = True

    # Extra checks for SV
    if st == "SV":
        if f["brightness_mean"] < 0.55 and f["p_bright"] > 0.2:
            probs["backlight"] = True

    else:  # RS specific
        h, w = f["height"], f["width"]

        if min(h, w) < 512:
            probs["low_resolution"] = True

        if probs["haze_like"]:
            probs["high_cloud"] = True

    return {
        "source_type": st,
        "problems": probs,
        "features": feats,
    }


# ========= 5bis. Decide whether restoration is needed =========

def should_restore(feats, probs, Q_before, q_thresh=0.35):
    """
    Decide if an image should go through the full restoration pipeline.
    Uses both global Q and rule-based problem flags.
    """
    if Q_before < q_thresh:
        return True

    problematic = [
        "low_light",
        "over_exposed",
        "low_contrast",
        "blur",
        "high_noise",
        "haze_like",
        "backlight",
        "low_resolution",
        "high_cloud",
    ]

    for k in problematic:
        if probs.get(k, False):
            return True

    return False


# ============= 6. Restoration Tools (SV + RS) =============

def SV_low_light_enhance(img_rgb):
    """SV: gamma + CLAHE enhancement for low-light / backlight scenes."""
    img = (img_rgb * 255).astype(np.uint8)
    gamma = 1.3
    img_gamma = np.power(np.clip(img / 255.0, 0, 1), 1.0 / gamma)
    img_gamma = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)

    ycrcb = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    out = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)
    return img_as_float(out).astype(np.float32)


def SV_deblur_unsharp(img_rgb):
    """SV: simple unsharp masking for deblurring."""
    img = (img_rgb * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    usm = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return np.clip(usm.astype(np.float32) / 255.0, 0.0, 1.0)


def RS_super_resolution_bicubic(img_rgb, scale=2):
    """RS: simple bicubic upsampling as super-resolution baseline."""
    h, w = img_rgb.shape[:2]
    img = (img_rgb * 255).astype(np.uint8)
    out = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    return img_as_float(out).astype(np.float32)


def RS_dehaze_simple(img_rgb):
    """RS: CLAHE on Y channel + mild gamma correction."""
    img = (img_rgb * 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    out = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)
    out_f = img_as_float(out).astype(np.float32)
    gamma = 1.1
    out_gamma = np.power(np.clip(out_f, 0, 1), 1.0 / gamma)
    return np.clip(out_gamma, 0.0, 1.0)


TOOL_FUNCTIONS = {
    "SV_low_light_enhance": SV_low_light_enhance,
    "SV_deblur_unsharp": SV_deblur_unsharp,
    "RS_super_resolution_bicubic": RS_super_resolution_bicubic,
    "RS_dehaze_simple": RS_dehaze_simple,
}

ALLOWED_SV_TOOLS = ["SV_low_light_enhance", "SV_deblur_unsharp"]
ALLOWED_RS_TOOLS = ["RS_super_resolution_bicubic", "RS_dehaze_simple"]


def apply_tool_chain(img_rgb, tool_chain):
    """Apply a sequence of restoration tools in order."""
    out = img_rgb.copy()

    for name in tool_chain:
        func = TOOL_FUNCTIONS.get(name)

        if func is None:
            print(f"[WARN] Unknown tool: {name}")
            continue

        out = func(out)

    return out


# ============= 7. Baseline Q_base (defect-driven) =============

def estimate_defect_scores(feats):
    """
    Estimate per-dimension defect scores in [0, 1]:
    brightness / contrast / sharpness / noise (higher = worse).
    """
    b_mean = feats["brightness_mean"]
    contrast = feats["contrast_std"]
    sharp = feats["laplacian_var"]
    noise = feats["noise_var"]

    dark_score = max(0.0, 0.5 - b_mean) / 0.5
    bright_score = max(0.0, b_mean - 0.8) / 0.2
    brightness_defect = max(dark_score, bright_score)

    contrast_defect = max(0.0, 0.06 - contrast) / 0.06
    sharp_defect = max(0.0, 40.0 - sharp) / 40.0
    noise_defect = max(0.0, noise - 0.002) / 0.01

    return {
        "brightness": float(np.clip(brightness_defect, 0, 1)),
        "contrast": float(np.clip(contrast_defect, 0, 1)),
        "sharpness": float(np.clip(sharp_defect, 0, 1)),
        "noise": float(np.clip(noise_defect, 0, 1)),
    }


def choose_baseline_tools_from_defect(source_type, defect_scores):
    """
    Choose a single baseline tool from the worst defect dimension.
    Always select at least one tool to create a clear baseline.
    """
    worst_type = max(defect_scores, key=lambda k: defect_scores[k])

    if source_type == "SV":
        if worst_type in ["brightness", "contrast"]:
            tools = ["SV_low_light_enhance"]
        else:
            tools = ["SV_deblur_unsharp"]

    else:
        if worst_type in ["brightness", "contrast", "noise"]:
            tools = ["RS_dehaze_simple"]
        else:
            tools = ["RS_super_resolution_bicubic"]

    return tools, worst_type, defect_scores[worst_type]


def compute_Q_base_for_image(
    img_path, source_type_override=None, save_base_image=True, suffix="_base"
):
    """
    Compute Q_base (baseline quality) for a single image.

    Steps:
      1) Run rule-based diagnostics.
      2) Estimate defect scores per dimension.
      3) Choose one baseline tool.
      4) Apply baseline tool(s) and recompute Q.
      5) Optionally save baseline image.
    """
    diag = diagnose_image_from_path(img_path, source_type_override)
    source_type = diag["source_type"]
    feats_before = diag["features"]

    img_rgb = load_image_rgb(img_path)
    Q_before_base, _ = compute_quality_score(feats_before)

    defect_scores = estimate_defect_scores(feats_before)
    base_tools, worst_type, worst_score = choose_baseline_tools_from_defect(
        source_type, defect_scores
    )

    base_img = apply_tool_chain(img_rgb, base_tools)
    feats_after_base = extract_iqa_features_from_rgb(base_img)
    Q_base, _ = compute_quality_score(feats_after_base)

    base_img_path = None

    if save_base_image:
        folder = os.path.dirname(img_path)
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        base_img_path = os.path.join(folder, base_name + suffix + ext)

        out_bgr = cv2.cvtColor((base_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(base_img_path, out_bgr)

    return {
        "image": os.path.basename(img_path),
        "path": img_path,
        "source_type": source_type,
        "Q_before_base": float(Q_before_base),
        "Q_base": float(Q_base),
        "base_tools": base_tools,
        "base_image_path": base_img_path,
        "defect_scores": defect_scores,
        "worst_defect_type": worst_type,
        "worst_defect_score": worst_score,
    }

# ============= 8. Tool-name normalization & problem-to-tool fallback =============

def normalize_tool_name(name, source_type):
    """Map predicted or informal tool name to canonical tool name."""
    if not isinstance(name, str):
        return None

    s = name.strip()

    if not s:
        return None

    s_lower = s.lower()

    tool_synonyms_sv = {
        "sv_low_light_enhance": "SV_low_light_enhance",
        "low_light_enhance": "SV_low_light_enhance",
        "low-light-enhance": "SV_low_light_enhance",
        "low light enhance": "SV_low_light_enhance",
        "sv_deblur_unsharp": "SV_deblur_unsharp",
        "deblur_unsharp": "SV_deblur_unsharp",
        "deblur": "SV_deblur_unsharp",
    }

    tool_synonyms_rs = {
        "rs_super_resolution_bicubic": "RS_super_resolution_bicubic",
        "super_resolution_bicubic": "RS_super_resolution_bicubic",
        "super resolution": "RS_super_resolution_bicubic",
        "sr": "RS_super_resolution_bicubic",
        "rs_dehaze_simple": "RS_dehaze_simple",
        "dehaze_simple": "RS_dehaze_simple",
        "dehaze": "RS_dehaze_simple",
        "dehazing": "RS_dehaze_simple",
    }

    if isinstance(source_type, str):
        source_type = source_type.strip().upper()

    if source_type == "SV":
        if s in ALLOWED_SV_TOOLS:
            return s

        return tool_synonyms_sv.get(s_lower)

    else:
        if s in ALLOWED_RS_TOOLS:
            return s

        return tool_synonyms_rs.get(s_lower)


def tools_from_problems(detected_problems, source_type):
    """
    Rule-based fallback mapping from problem types to restoration tool chain.
    """
    if detected_problems is None:
        detected_problems = []

    if isinstance(source_type, str):
        source_type = source_type.strip().upper()

    tools = []

    if source_type == "SV":
        if "low_light" in detected_problems or "backlight" in detected_problems:
            tools.append("SV_low_light_enhance")

        if "blur" in detected_problems:
            tools.append("SV_deblur_unsharp")

    else:
        if "low_resolution" in detected_problems:
            tools.append("RS_super_resolution_bicubic")

        if (
            "haze_or_cloud" in detected_problems
            or "high_cloud" in detected_problems
            or "low_light" in detected_problems
            or "low_contrast" in detected_problems
            or "noise" in detected_problems
            or "high_noise" in detected_problems
        ):
            tools.append("RS_dehaze_simple")

    out = []

    for t in tools:
        if t not in out:
            out.append(t)

    return out


# ============= 9. Merge Rule-based & Planner Diagnostics =============

def reconcile_problems(rule_probs, detected_problems):
    """
    Merge rule-based and planner-based diagnostics into final_problems.

    Rules:
    - If planner returns ["none"], mark all problems as False.
    - Otherwise, start from rule_probs and force True for planner problems.
    """
    final = dict(rule_probs)

    if detected_problems is None:
        detected_problems = []

    detected_problems = [
        str(p).strip().lower()
        for p in detected_problems
        if isinstance(p, str) and str(p).strip()
    ]

    gem2rule = {
        "low_light": "low_light",
        "backlight": "backlight",
        "blur": "blur",
        "low_contrast": "low_contrast",
        "noise": "high_noise",
        "high_noise": "high_noise",
        "low_resolution": "low_resolution",
        "haze_or_cloud": "haze_like",
        "high_cloud": "high_cloud",
        "compression_artifacts": None,
        "none": None,
    }

    if (not detected_problems) or (detected_problems == ["none"]):
        for k in final:
            final[k] = False

        return final

    for p in detected_problems:
        key = gem2rule.get(p)

        if key and key in final:
            final[key] = True

    return final

# ============= 10A. LLM Planner Prompt & Call =============

def encode_image_to_jpeg_bytes(img_rgb):
    """Encode RGB float32 image to JPEG bytes for planner vision input."""
    img_u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    ok, buf = cv2.imencode(".jpg", bgr)

    if not ok:
        raise RuntimeError("cv2.imencode failed.")

    return buf.tobytes()


def build_restoration_planner_prompt(source_type, hazard_type, iqa_feats):
    """
    Build the LLM planner prompt.

    The planner is allowed to recommend only deterministic restoration tools.
    It must not perform generative restoration.
    """
    if isinstance(source_type, str):
        source_type = source_type.strip().upper()

    allowed_tools = ALLOWED_SV_TOOLS if source_type == "SV" else ALLOWED_RS_TOOLS
    iqa_json = json.dumps(iqa_feats, indent=2)

    return f"""
You are the GeoImageRestoration planning module of HERMES.

Your task:
- Analyze disaster-related imagery.
- Use the image and the no-reference IQA features.
- Detect image-quality degradations.
- Recommend ONLY deterministic, non-generative restoration tools.
- Do NOT hallucinate content.
- Do NOT propose tools outside the allowed list.

Image metadata:
- source_type = "{source_type}"
- hazard_type = "{hazard_type}"

No-reference IQA features:
{iqa_json}

Allowed deterministic tools:
{allowed_tools}

Possible detected problem types:
- "low_light"
- "backlight"
- "blur"
- "low_contrast"
- "noise"
- "high_noise"
- "low_resolution"
- "haze_or_cloud"
- "high_cloud"
- "compression_artifacts"
- "none"

Decision rules:
- For remote-sensing imagery, mark "low_resolution" only if min(height, width) < 512.
- Do not recommend "RS_super_resolution_bicubic" for images with width >= 512 and height >= 512 unless there is clear visual evidence of severe detail loss.
- If image quality is already good and no deterministic restoration is necessary, return:
  detected_problems = ["none"]
  recommended_tools = []

Return ONLY valid JSON with this exact schema:
{{
  "detected_problems": ["..."],
  "recommended_tools": ["..."],
  "agent_decision": "short English explanation"
}}
"""


def parse_planner_json(raw_text):
    """
    Parse planner JSON robustly, including cases where the model wraps JSON
    inside markdown code fences.
    """
    if raw_text is None:
        raise ValueError("Planner response text is empty.")

    text = raw_text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


def call_gemini_for_plan(img_rgb, source_type, hazard_type, iqa_feats):
    """
    Call Gemini planner model to get:
    - detected problems
    - recommended deterministic restoration tools
    - planner decision text

    Returns:
      detected_problems, normalized_tools, agent_decision
    """
    if isinstance(source_type, str):
        source_type = source_type.strip().upper()

    client = get_gemini_client()

    prompt = build_restoration_planner_prompt(
        source_type=source_type,
        hazard_type=hazard_type,
        iqa_feats=iqa_feats,
    )

    img_bytes = encode_image_to_jpeg_bytes(img_rgb)

    try:
        response = client.models.generate_content(
            model=GEMINI_PLANNER_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            ],
            config={"response_modalities": ["TEXT"]},
        )

        raw_text = response.text.strip()
        data = parse_planner_json(raw_text)

    except Exception as e:
        print("[WARN] Gemini planner failed:", e)

        data = {
            "detected_problems": ["none"],
            "recommended_tools": [],
            "agent_decision": "Gemini planner failed; falling back to deterministic logic.",
        }

    detected_problems = data.get("detected_problems", [])
    raw_tools = data.get("recommended_tools", [])
    agent_decision = data.get("agent_decision", "")

    if isinstance(detected_problems, str):
        detected_problems = [detected_problems]

    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]

    detected_problems = [
        str(p).strip().lower()
        for p in detected_problems
        if isinstance(p, str) and str(p).strip()
    ]

    normalized_tools = []

    for tool_name in raw_tools:
        canonical = normalize_tool_name(tool_name, source_type)

        if canonical and canonical not in normalized_tools:
            normalized_tools.append(canonical)

    allowed_tools = ALLOWED_SV_TOOLS if source_type == "SV" else ALLOWED_RS_TOOLS
    normalized_tools = [t for t in normalized_tools if t in allowed_tools]

    # Fallback: if planner detects problems but returns no valid tools
    if (
        not normalized_tools
        and detected_problems
        and detected_problems != ["none"]
    ):
        fallback_tools = tools_from_problems(detected_problems, source_type)
        fallback_tools = [t for t in fallback_tools if t in allowed_tools]

        if fallback_tools:
            normalized_tools = fallback_tools
            agent_decision = (
                agent_decision
                + " | Fallback deterministic tools selected: "
                + ", ".join(normalized_tools)
            )

    return detected_problems, normalized_tools, agent_decision


# ============= 10. Full Deterministic Restoration Pipeline =============

def save_rgb_image(img_rgb, out_path):
    """Save RGB float32 [0, 1] image to disk."""
    out_dir = os.path.dirname(out_path)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    img_u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(out_path, bgr)

    if not ok:
        raise RuntimeError(f"Could not write image: {out_path}")

    return out_path


def restore_single_image(
    img_path,
    source_type_override=None,
    output_dir="outputs/restored",
    q_thresh=0.35,
    min_delta_q=0.01,
    save_restored=True,
):
    """
    Run the full deterministic GeoImageRestorationAgent pipeline for one image.

    Steps:
      1) Diagnose image degradations.
      2) Compute Q_before.
      3) Decide whether restoration is needed.
      4) Select deterministic restoration tools from defect scores.
      5) Apply restoration.
      6) Compute Q_after.
      7) Accept restored image only if Q improves enough.
      8) Return structured result.
    """
    diag = diagnose_image_from_path(img_path, source_type_override)

    source_type = diag["source_type"]
    rule_probs = diag["problems"]
    feats_before = diag["features"]

    Q_before, Q_before_components = compute_quality_score(feats_before)

    needs_restoration = should_restore(
        feats_before,
        rule_probs,
        Q_before,
        q_thresh=q_thresh,
    )

    result = {
        "image": os.path.basename(img_path),
        "input_path": img_path,
        "source_type": source_type,
        "needs_restoration": bool(needs_restoration),
        "accepted": False,
        "final_path": img_path,
        "restored_path": None,
        "tool_chain": [],
        "Q_before": float(Q_before),
        "Q_after": None,
        "delta_Q": 0.0,
        "Q_before_components": Q_before_components,
        "Q_after_components": None,
        "problems": rule_probs,
        "features_before": feats_before,
        "features_after": None,
        "defect_scores": None,
        "worst_defect_type": None,
        "worst_defect_score": None,
        "decision": "No restoration needed.",
    }

    if not needs_restoration:
        return result

    img_rgb = load_image_rgb(img_path)

    defect_scores = estimate_defect_scores(feats_before)
    tool_chain, worst_type, worst_score = choose_baseline_tools_from_defect(
        source_type,
        defect_scores,
    )

    restored_img = apply_tool_chain(img_rgb, tool_chain)

    feats_after = extract_iqa_features_from_rgb(restored_img)
    Q_after, Q_after_components = compute_quality_score(feats_after)

    delta_q = Q_after - Q_before
    accepted = delta_q >= min_delta_q

    restored_path = None

    if save_restored:
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        restored_name = base_name + "_restored" + ext
        restored_path = os.path.join(output_dir, restored_name)
        save_rgb_image(restored_img, restored_path)

    result.update(
        {
            "accepted": bool(accepted),
            "final_path": restored_path if accepted and restored_path else img_path,
            "restored_path": restored_path,
            "tool_chain": tool_chain,
            "Q_after": float(Q_after),
            "delta_Q": float(delta_q),
            "Q_after_components": Q_after_components,
            "features_after": feats_after,
            "defect_scores": defect_scores,
            "worst_defect_type": worst_type,
            "worst_defect_score": float(worst_score),
            "decision": (
                "Restoration accepted; Q improved sufficiently."
                if accepted
                else "Restoration rejected; Q improvement was insufficient."
            ),
        }
    )

    return result

# ============= 10B. Full Planner-based Restoration Pipeline =============

def restore_single_image_with_planner(
    img_path,
    source_type_override=None,
    hazard_type="unknown",
    output_dir="outputs/restored",
    q_thresh=0.35,
    min_delta_q=0.01,
    save_restored=True,
):
    """
    Run the full planner-based GeoImageRestorationAgent pipeline for one image.

    This is GIRA v2:
      1) Run rule-based diagnostics.
      2) Compute Q_before.
      3) Decide whether restoration is needed.
      4) Ask the LLM planner to detect degradations and recommend deterministic tools.
      5) Normalize and validate planner tools.
      6) Fall back to deterministic defect-based tools if planner fails or returns no valid tools.
      7) Apply deterministic restoration tools.
      8) Compute Q_after.
      9) Accept restored image only if Q improves enough.
      10) Return structured result.

    Important:
      - The LLM does NOT generate or modify the image.
      - The LLM only plans which deterministic tools should be used.
    """
    diag = diagnose_image_from_path(img_path, source_type_override)

    source_type = diag["source_type"]
    rule_probs = diag["problems"]
    feats_before = diag["features"]

    Q_before, Q_before_components = compute_quality_score(feats_before)

    needs_restoration = should_restore(
        feats_before,
        rule_probs,
        Q_before,
        q_thresh=q_thresh,
    )

    result = {
        "image": os.path.basename(img_path),
        "input_path": img_path,
        "source_type": source_type,
        "hazard_type": hazard_type,
        "mode": "planner",
        "needs_restoration": bool(needs_restoration),
        "accepted": False,
        "final_path": img_path,
        "restored_path": None,
        "tool_chain": [],
        "tool_selector": None,
        "Q_before": float(Q_before),
        "Q_after": None,
        "delta_Q": 0.0,
        "Q_before_components": Q_before_components,
        "Q_after_components": None,
        "rule_problems": rule_probs,
        "planner_detected_problems": [],
        "final_problems": rule_probs,
        "features_before": feats_before,
        "features_after": None,
        "defect_scores": None,
        "worst_defect_type": None,
        "worst_defect_score": None,
        "planner_decision": None,
        "decision": "No restoration needed.",
    }

    if not needs_restoration:
        return result

    img_rgb = load_image_rgb(img_path)

    # Deterministic fallback is always computed.
    # This guarantees that planner mode remains usable even if Gemini fails.
    defect_scores = estimate_defect_scores(feats_before)
    fallback_tool_chain, worst_type, worst_score = choose_baseline_tools_from_defect(
        source_type,
        defect_scores,
    )

    planner_detected_problems = []
    planner_tool_chain = []
    planner_decision = ""

    try:
        (
            planner_detected_problems,
            planner_tool_chain,
            planner_decision,
        ) = call_gemini_for_plan(
            img_rgb=img_rgb,
            source_type=source_type,
            hazard_type=hazard_type,
            iqa_feats=feats_before,
        )

        # Safety filter:
        # Do not accept "low_resolution" / SR recommendation for RS images
        # that are already at least 512x512, matching the deterministic rule.
        h = feats_before.get("height", 0)
        w = feats_before.get("width", 0)

        if source_type == "RS" and min(h, w) >= 512:
            planner_detected_problems = [
                p for p in planner_detected_problems
                if p != "low_resolution"
            ]

            planner_tool_chain = [
                t for t in planner_tool_chain
                if t != "RS_super_resolution_bicubic"
            ]

            if "low_resolution" in planner_decision.lower():
                planner_decision += (
                    " | HERMES safety filter removed low_resolution/SR because "
                    "the image is already at least 512x512."
                )

    except Exception as e:
        planner_detected_problems = []
        planner_tool_chain = []
        planner_decision = (
            f"Gemini planner call failed; using deterministic fallback. Error: {e}"
        )

    # If planner gives valid deterministic tools, use them.
    # Otherwise, use deterministic defect-based fallback.
    if planner_tool_chain:
        tool_chain = planner_tool_chain
        tool_selector = "llm_planner"

        if planner_detected_problems and planner_detected_problems != ["none"]:
            final_problems = reconcile_problems(rule_probs, planner_detected_problems)
        else:
            final_problems = rule_probs

    else:
        tool_chain = fallback_tool_chain
        tool_selector = "deterministic_fallback"
        final_problems = rule_probs

        if not planner_decision:
            planner_decision = (
                "Planner returned no valid tools; deterministic fallback was used."
            )

    restored_img = apply_tool_chain(img_rgb, tool_chain)

    feats_after = extract_iqa_features_from_rgb(restored_img)
    Q_after, Q_after_components = compute_quality_score(feats_after)

    delta_q = Q_after - Q_before
    accepted = delta_q >= min_delta_q

    restored_path = None

    if save_restored:
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        restored_name = base_name + "_planner_restored" + ext
        restored_path = os.path.join(output_dir, restored_name)
        save_rgb_image(restored_img, restored_path)

    result.update(
        {
            "accepted": bool(accepted),
            "final_path": restored_path if accepted and restored_path else img_path,
            "restored_path": restored_path,
            "tool_chain": tool_chain,
            "tool_selector": tool_selector,
            "Q_after": float(Q_after),
            "delta_Q": float(delta_q),
            "Q_after_components": Q_after_components,
            "planner_detected_problems": planner_detected_problems,
            "final_problems": final_problems,
            "features_after": feats_after,
            "defect_scores": defect_scores,
            "worst_defect_type": worst_type,
            "worst_defect_score": float(worst_score),
            "planner_decision": planner_decision,
            "decision": (
                "Planner-based restoration accepted; Q improved sufficiently."
                if accepted
                else "Planner-based restoration rejected; Q improvement was insufficient."
            ),
        }
    )

    return result


# ============= 10C. Multi-image / Pair Restoration Wrapper =============

def _normalize_path_list(image_paths):
    """
    Normalize image_paths input to a list of strings.

    Accepts:
      - single string path
      - comma-separated string
      - semicolon-separated string
      - list/tuple of paths
    """
    if image_paths is None:
        return []

    if isinstance(image_paths, str):
        text = image_paths.strip()

        if not text:
            return []

        if ";" in text:
            return [p.strip() for p in text.split(";") if p.strip()]

        if "," in text:
            return [p.strip() for p in text.split(",") if p.strip()]

        return [text]

    if isinstance(image_paths, (list, tuple)):
        return [str(p).strip() for p in image_paths if str(p).strip()]

    return [str(image_paths).strip()]


def _normalize_source_types(source_types, n):
    """
    Normalize source_types to a list of length n.

    Accepts:
      - None
      - single string: 'RS', 'SV', 'RS,SV', 'RS;SV'
      - list/tuple
    """
    if n <= 0:
        return []

    if source_types is None:
        return [None] * n

    if isinstance(source_types, str):
        text = source_types.strip()

        if not text:
            return [None] * n

        if ";" in text:
            items = [s.strip().upper() for s in text.split(";") if s.strip()]
        elif "," in text:
            items = [s.strip().upper() for s in text.split(",") if s.strip()]
        else:
            items = [text.upper()]

    elif isinstance(source_types, (list, tuple)):
        items = [
            str(s).strip().upper() if s is not None else None
            for s in source_types
        ]

    else:
        items = [str(source_types).strip().upper()]

    if len(items) < n:
        items = items + [None] * (n - len(items))

    return items[:n]


def restore_image_collection(
    image_paths,
    source_types=None,
    hazard_type="unknown",
    output_dir="outputs/restored",
    q_thresh=0.35,
    min_delta_q=0.01,
    mode="planner",
    save_restored=True,
):
    """
    Run GIRA on a collection of images, including image pairs.

    This wrapper does NOT perform pair-level reasoning.
    It applies the existing single-image restoration pipeline independently
    to each image and returns the final path for every image.

    Intended use:
      - single image
      - RS + SVI post-disaster pair
      - pre/post SV pair
      - pre/post RS pair

    Pair-level interpretation remains the responsibility of GPA and GDAA.
    GIRA only ensures that each image passed downstream has gone through
    the same restoration / quality-gating logic.
    """
    paths = _normalize_path_list(image_paths)

    if not paths:
        raise ValueError("No image paths were provided to restore_image_collection().")

    mode = str(mode).strip().lower()
    source_type_list = _normalize_source_types(source_types, len(paths))

    results = []
    final_paths = []
    restored_paths = []
    accepted_flags = []
    needs_restoration_flags = []

    for idx, path in enumerate(paths):
        source_type = source_type_list[idx]

        if mode == "planner":
            result = restore_single_image_with_planner(
                img_path=path,
                source_type_override=source_type,
                hazard_type=hazard_type,
                output_dir=output_dir,
                q_thresh=q_thresh,
                min_delta_q=min_delta_q,
                save_restored=save_restored,
            )

        else:
            result = restore_single_image(
                img_path=path,
                source_type_override=source_type,
                output_dir=output_dir,
                q_thresh=q_thresh,
                min_delta_q=min_delta_q,
                save_restored=save_restored,
            )

            result["mode"] = "deterministic"

        result["image_index"] = idx

        results.append(result)
        final_paths.append(result.get("final_path", path))
        restored_paths.append(result.get("restored_path"))
        accepted_flags.append(bool(result.get("accepted", False)))
        needs_restoration_flags.append(bool(result.get("needs_restoration", False)))

    return {
        "mode": mode,
        "num_images": len(paths),
        "input_paths": paths,
        "source_types": source_type_list,
        "hazard_type": hazard_type,
        "final_paths": final_paths,
        "restored_paths": restored_paths,
        "accepted": accepted_flags,
        "needs_restoration": needs_restoration_flags,
        "results": results,
    }


# ============= 11. Fast Pre-screening: Need Restoration? =============

def need_restoration_basic(
    img_path,
    source_type_override=None,
    thr_brightness_low=0.35,
    thr_brightness_high=0.80,
    thr_contrast=0.04,
    thr_lap=15.0,
):
    """
    Very fast screening:
    Use only simple statistics (brightness / contrast / Laplacian)
    to decide whether an image should enter the full restoration branch.

    Returns:
      need_restore : bool
      quick_feats  : dict
      reason       : str
    """
    img_rgb = load_image_rgb(img_path)
    gray = to_gray(img_rgb)

    brightness_mean = float(gray.mean())
    contrast_std = float(gray.std())
    lap_var = float(
        cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F).var()
    )

    quick_feats = {
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "laplacian_var": lap_var,
    }

    reasons = []

    if brightness_mean < thr_brightness_low:
        reasons.append("too dark")
    elif brightness_mean > thr_brightness_high:
        reasons.append("too bright")

    if contrast_std < thr_contrast:
        reasons.append("low contrast")

    if lap_var < thr_lap:
        reasons.append("blurry")

    if len(reasons) == 0:
        need_restore = False
        reason_text = (
            "No restoration needed (brightness/contrast/sharpness within normal range)."
        )
    else:
        need_restore = True
        reason_text = "Need restoration: " + ", ".join(reasons)

    return need_restore, quick_feats, reason_text