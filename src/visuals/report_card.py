import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_report_card(image_path, report_ref, report_pred, concepts_top, deltas_top, out_path):
    img = Image.open(image_path).convert("RGB")

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 1.2])

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img)
    ax0.axis("off")
    ax0.set_title("Chest X-ray")

    ax1 = fig.add_subplot(gs[1])
    labels = [c for c, _ in concepts_top]
    vals = [v for _, v in concepts_top]
    ax1.barh(labels[::-1], vals[::-1], color="#2f4f4f")
    ax1.set_xlim(0, 1)
    ax1.set_title("Concept Evidence (sigmoid probs)")

    ax2 = fig.add_subplot(gs[2])
    labels2 = [c for c, _ in deltas_top]
    vals2 = [v for _, v in deltas_top]
    ax2.barh(labels2[::-1], vals2[::-1], color="#8b0000")
    ax2.set_title("Counterfactual Influence (LL drop)")

    text = "Reference:\n" + report_ref + "\n\nGenerated:\n" + report_pred
    fig.text(0.02, 0.02, text, fontsize=9, va="bottom", ha="left", wrap=True)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

