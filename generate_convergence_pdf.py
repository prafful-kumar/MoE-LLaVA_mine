import os
import json
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

# --------- EDIT THESE PATHS IF NEEDED ---------

experiments = {
    "Random Init":
    "random_init_student_only_training_norm_./DYN_HYP_Fisher5k-T1.0_0.6-W0.0_0.0-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",

    "Fisher (No Norm)":
    "fisher_no_norm_input_norm_weight_during_training_./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",

    "Fisher (TS + Norm)":
    "fisher_TS_norm_training_norm_input_norm_weight./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",
}

def smooth_ema(values, alpha=0.05):
    smoothed = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed

processed = {}

for name, folder in experiments.items():
    path = os.path.join(folder, "trainer_state.json")

    with open(path) as f:
        data = json.load(f)

    steps = []
    losses = []

    for entry in data["log_history"]:
        if "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    losses = smooth_ema(losses)
    processed[name] = (steps, losses)

# --------- Plot 1: Full Convergence ---------

plt.figure(figsize=(8,6))
for name, (steps, losses) in processed.items():
    plt.plot(steps, losses, label=name)

plt.xlabel("Step")
plt.ylabel("Training Loss (EMA)")
plt.title("Full Convergence Comparison")
plt.legend()
plt.grid(True)

plt.savefig("full_convergence.png", dpi=300, bbox_inches="tight")
plt.close()

# --------- Plot 2: Early Training ---------

plt.figure(figsize=(8,6))
for name, (steps, losses) in processed.items():
    early_s = []
    early_l = []
    for s, l in zip(steps, losses):
        if s <= 1000:
            early_s.append(s)
            early_l.append(l)
    plt.plot(early_s, early_l, label=name)

plt.xlabel("Step")
plt.ylabel("Training Loss (EMA)")
plt.title("Early Training (0–1000 Steps)")
plt.legend()
plt.grid(True)

plt.savefig("early_convergence.png", dpi=300, bbox_inches="tight")
plt.close()

# --------- Plot 3: Rolling Variance ---------

plt.figure(figsize=(8,6))
for name, (steps, losses) in processed.items():
    window = 50
    var = []
    var_steps = []
    for i in range(window, len(losses)):
        var.append(np.var(losses[i-window:i]))
        var_steps.append(steps[i])
    plt.plot(var_steps, var, label=name)

plt.xlabel("Step")
plt.ylabel("Rolling Variance (window=50)")
plt.title("Training Stability Comparison")
plt.legend()
plt.grid(True)

plt.savefig("loss_variance.png", dpi=300, bbox_inches="tight")
plt.close()

# --------- Build PDF ---------

doc = SimpleDocTemplate("MoE_Convergence_Comparison.pdf", pagesize=letter)
elements = []

style = ParagraphStyle(name="Title", fontSize=16, spaceAfter=12)

elements.append(Paragraph("MoE Initialization Convergence Analysis", style))
elements.append(Spacer(1, 0.3 * inch))
elements.append(Image("full_convergence.png", width=6*inch, height=4.5*inch))
elements.append(PageBreak())

elements.append(Paragraph("Early Training Dynamics (Cold Start)", style))
elements.append(Spacer(1, 0.3 * inch))
elements.append(Image("early_convergence.png", width=6*inch, height=4.5*inch))
elements.append(PageBreak())

elements.append(Paragraph("Training Stability (Rolling Variance)", style))
elements.append(Spacer(1, 0.3 * inch))
elements.append(Image("loss_variance.png", width=6*inch, height=4.5*inch))

doc.build(elements)

print("PDF generated: MoE_Convergence_Comparison.pdf")