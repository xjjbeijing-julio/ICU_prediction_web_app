#!/usr/bin/env python3
"""
ICU患者死亡风险预测 — Web工具
基于液态神经网络(CfC/LNN) + 逻辑回归模型
"""

import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, io, pickle, json, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify

# ncps for CfC model
from ncps.torch import CfC

app = Flask(__name__)
DEVICE = torch.device("cpu")

# ============================================================================
# 模型定义 (与训练时完全一致)
# ============================================================================
class CfCPredictor(nn.Module):
    def __init__(self, seq_dim, static_dim):
        super().__init__()
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 24), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(24, 24), nn.ReLU())
        self.seq_model = CfC(seq_dim, 48, batch_first=True, mixed_memory=True, mode="default")
        self.head = nn.Sequential(
            nn.Linear(48+24, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))

    def forward(self, x_seq, x_static):
        seq_out, _ = self.seq_model(x_seq)
        static_feat = self.static_net(x_static)
        return self.head(torch.cat([seq_out[:, -1, :], static_feat], dim=-1)).squeeze(-1)

# ============================================================================
# 加载模型
# ============================================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

with open(os.path.join(MODEL_DIR, "lr_model.pkl"), "rb") as f:
    lr_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "calibrator.pkl"), "rb") as f:
    calibrator = pickle.load(f)

cfc_model = CfCPredictor(30, 9)
cfc_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "cfc_model.pt"),
                                      map_location="cpu", weights_only=True))
cfc_model.eval()

with open(os.path.join(MODEL_DIR, "model_meta.json"), "r") as f:
    meta = json.load(f)

print("[OK] All models loaded successfully")

# ============================================================================
# 特征名称
# ============================================================================
STATIC_NAMES = meta["feature_names"]["static"]
VITAL_NAMES = meta["feature_names"]["vital_names"]

# ============================================================================
# 预测函数
# ============================================================================
def predict_lr(static_values, vital_last_values):
    """逻辑回归预测 (19维特征: 9 static + 10 vital)"""
    static = np.array(static_values, dtype=np.float32)
    vital = np.array(vital_last_values, dtype=np.float32)
    features = np.concatenate([static, vital])  # 19维
    features_s = scaler.transform(features.reshape(1, -1))
    raw_prob = lr_model.predict_proba(features_s)[0, 1]
    calibrated_prob = float(np.clip(calibrator.predict([raw_prob])[0], 0.0, 1.0))
    return {"raw_prob": float(raw_prob), "calibrated_prob": calibrated_prob}

def predict_cfc(static_values, vital_sequence):
    """CfC液态神经网络预测 (30维序列)"""
    vital = np.array(vital_sequence, dtype=np.float32)  # (T, 10)
    T = vital.shape[0]

    # 构建三段式输入 (value + mask + delta): (T, 30)
    mask = np.ones((T, 10), dtype=np.float32)
    delta = np.zeros((T, 10), dtype=np.float32)
    for i in range(1, T):
        delta[i] = delta[i-1] + 1

    seq = np.concatenate([vital, mask, delta], axis=-1)  # (T, 30)

    static = np.array(static_values, dtype=np.float32)

    with torch.no_grad():
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T, 30)
        static_t = torch.tensor(static, dtype=torch.float32).unsqueeze(0)  # (1, 9)
        logit = cfc_model(seq_t, static_t)
        prob = float(torch.sigmoid(logit).cpu().numpy())

    return {"raw_prob": prob, "calibrated_prob": prob}

# ============================================================================
# 风险等级判定
# ============================================================================
def get_risk_level(prob):
    if prob < 0.15:
        return {"level": "low", "color": "#22c55e", "label": "低风险",
                "icon": "🟢", "description": "患者死亡风险较低，建议常规监护"}
    elif prob < 0.35:
        return {"level": "moderate", "color": "#eab308", "label": "中等风险",
                "icon": "🟡", "description": "患者存在一定死亡风险，建议加强监护频率"}
    elif prob < 0.60:
        return {"level": "high", "color": "#f97316", "label": "高风险",
                "icon": "🟠", "description": "患者死亡风险较高，建议启动预警并医疗干预"}
    else:
        return {"level": "critical", "color": "#ef4444", "label": "极高风险",
                "icon": "🔴", "description": "患者死亡风险极高，需立即启动急救响应"}

# ============================================================================
# 路由
# ============================================================================
@app.route("/")
def index():
    return render_template("index.html",
                          static_names=STATIC_NAMES,
                          vital_names=VITAL_NAMES)

@app.route("/api/predict/lr", methods=["POST"])
def api_predict_lr():
    """逻辑回归预测API"""
    try:
        data = request.get_json()
        static_values = [
            float(data.get("age", 65)),
            float(data.get("gender", 0)),
            float(data.get("bmi", 28)),
            float(data.get("emergency", 0)),
            float(data.get("charlson", 3)),
            float(data.get("sofa", 5)),
            float(data.get("sbp", 120)),
            float(data.get("hr", 80)),
            float(data.get("temp", 37)),
        ]
        vital_last = [
            float(data.get("vital_hr", 80)),
            float(data.get("vital_sbp", 120)),
            float(data.get("vital_dbp", 70)),
            float(data.get("vital_rr", 18)),
            float(data.get("vital_spo2", 97)),
            float(data.get("vital_temp", 37)),
            float(data.get("vital_gcs", 15)),
            float(data.get("vital_urine", 50)),
            float(data.get("vital_lactate", 1.5)),
            float(data.get("vital_ph", 7.4)),
        ]
        result = predict_lr(static_values, vital_last)
        risk = get_risk_level(result["calibrated_prob"])

        return jsonify({
            "success": True,
            "model": "LogisticRegression + Isotonic Calibration",
            "probability": round(result["calibrated_prob"] * 100, 1),
            "probability_raw": round(result["raw_prob"], 3),
            "risk": risk,
            "interpretation": generate_interpretation(static_values, vital_last, result["calibrated_prob"])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/predict/cfc", methods=["POST"])
def api_predict_cfc():
    """CfC液态神经网络预测API"""
    try:
        data = request.get_json()
        static_values = [
            float(data.get("age", 65)),
            float(data.get("gender", 0)),
            float(data.get("bmi", 28)),
            float(data.get("emergency", 0)),
            float(data.get("charlson", 3)),
            float(data.get("sofa", 5)),
            float(data.get("sbp", 120)),
            float(data.get("hr", 80)),
            float(data.get("temp", 37)),
        ]
        # 接收3个时间点的生命体征 (可扩展)
        vital_seq = []
        for t in range(3):
            vital_seq.append([
                float(data.get(f"vital_hr_t{t}", 80)),
                float(data.get(f"vital_sbp_t{t}", 120)),
                float(data.get(f"vital_dbp_t{t}", 70)),
                float(data.get(f"vital_rr_t{t}", 18)),
                float(data.get(f"vital_spo2_t{t}", 97)),
                float(data.get(f"vital_temp_t{t}", 37)),
                float(data.get(f"vital_gcs_t{t}", 15)),
                float(data.get(f"vital_urine_t{t}", 50)),
                float(data.get(f"vital_lactate_t{t}", 1.5)),
                float(data.get(f"vital_ph_t{t}", 7.4)),
            ])
        result = predict_cfc(static_values, vital_seq)
        risk = get_risk_level(result["calibrated_prob"])

        return jsonify({
            "success": True,
            "model": "CfC Liquid Neural Network",
            "probability": round(result["calibrated_prob"] * 100, 1),
            "risk": risk,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

def generate_interpretation(static, vital, prob):
    """生成临床解读"""
    items = []
    age, gender, bmi, emergency, charlson, sofa, sbp_b, hr_b, temp_b = static
    hr, sbp, dbp, rr, spo2, vt, gcs, urine, lactate, ph = vital

    if age > 70:
        items.append(f"年龄 {age:.0f} 岁属于高危年龄段")
    if sofa > 6:
        items.append(f"SOFA评分 {sofa:.0f} 提示多器官功能障碍")
    if charlson > 4:
        items.append(f"Charlson指数 {charlson:.0f} 提示较重合并症负担")
    if spo2 < 92:
        items.append(f"血氧饱和度 {spo2:.1f}% 偏低，提示呼吸功能受损")
    if lactate > 2.0:
        items.append(f"血乳酸 {lactate:.1f} mmol/L 升高，提示组织灌注不足")
    if gcs < 13:
        items.append(f"GCS评分 {gcs:.0f} 提示意识障碍")
    if hr > 100:
        items.append(f"心率 {hr:.0f} bpm 偏快")
    if sbp < 90:
        items.append(f"收缩压 {sbp:.0f} mmHg 偏低，需关注循环状态")
    if not items:
        items.append("当前各项指标在参考范围内")
    return items

# ============================================================================
# 启动
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" ICU Mortality Risk Prediction Web Tool")
    print("="*60)
    print("  URL: http://localhost:5000")
    print("  Models: LogisticRegression + CfC LNN")
    print("="*60 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
