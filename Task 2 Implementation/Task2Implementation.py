__version__ = "2.0.0"

import csv
import asyncio
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve
from openai import AsyncOpenAI, RateLimitError
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY") or "GET NVIDIA API KEY TO USE THIS CODE :D"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "openai/gpt-oss-120b"
NUM_RESPONSES = 3
RPM_LIMIT = 35
MAX_RETRIES = 5

if not API_KEY:
    raise ValueError("No API key found. Set NVIDIA_API_KEY in your .env file.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready.\n")

BG = "#0f0f0f"
PANEL = "#1a1a1a"
WHITE = "#e8e8e8"
DIM = "#888888"
C_EM = "#00d4ff"
C_COS = "#ff6b6b"
C_CLUST = "#a8ff78"
C_COMB = "#f5a623"
C_CORRECT = "#a8ff78"
C_INCORRECT = "#ff6b6b"


# Applies dark theme styling to a matplotlib axis.
def style_axis(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_color("#333333")


# Limits how often we call the API to stay under the RPM cap.
class RateLimiter:
    def __init__(self, rpm):
        self.min_interval = 60.0 / rpm
        self.lock = asyncio.Lock()
        self.last_call = 0.0

    async def acquire(self):
        async with self.lock:
            wait = self.min_interval - (time.monotonic() - self.last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_call = time.monotonic()


rate_limiter = RateLimiter(RPM_LIMIT)


# Reads prompts and answers from a CSV file.
def load_prompts(csv_file):
    prompts, answers = [], []
    with open(csv_file, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            prompts.append(row['prompt'])
            answers.append(row.get('Answer', row.get('answer', 'N/A')))
    return prompts, answers


# Returns True if the model answer matches the ground truth via exact or partial string match.
def is_correct(model_answer, ground_truth):
    a = str(model_answer).lower().strip().rstrip('.')
    b = str(ground_truth).lower().strip().rstrip('.')
    return a == b or b in a.split() or (a.split()[0] == b if a else False)


# Sends a prompt to the model and returns the response, retrying on failure.
async def fetch_response(prompt):
    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Answer with only the final result. No explanations, no working, no reasoning. Just the answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            if response is None or not response.choices:
                await asyncio.sleep(2)
                continue

            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
            if content:
                return content.strip()

            reasoning = getattr(choice.message, "reasoning_content", None)
            if reasoning:
                lines = [l.strip() for l in reasoning.strip().splitlines() if l.strip()]
                return lines[-1] if lines else "No answer"

            return "No answer"

        except RateLimitError:
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            print(f"⚠️ {e}, retrying in 3s...")
            await asyncio.sleep(3)

    return "No answer"


# Scores responses by what fraction share the most common answer.
def exact_match_score(responses):
    normalized = [r.lower().strip() for r in responses]
    counts = {r: normalized.count(r) for r in set(normalized)}
    return max(counts.values()) / len(responses)


# Measures response agreement using cosine similarity of sentence embeddings.
def cosine_similarity_score(responses):
    embeddings = embedder.encode(responses, normalize_embeddings=True)
    sim_matrix = np.dot(embeddings, embeddings.T)
    mask = ~np.eye(len(responses), dtype=bool)
    return float(sim_matrix[mask].mean())


# Groups responses into semantic clusters and scores based on how few clusters there are.
def semantic_cluster_score(responses):
    threshold = 0.85
    embeddings = embedder.encode(responses, normalize_embeddings=True)
    n = len(responses)
    clusters = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.dot(embeddings[i], embeddings[j])) >= threshold:
                old = clusters[j]
                new = clusters[i]
                clusters = [new if c == old else c for c in clusters]

    num_clusters = len(set(clusters))
    return 1.0 - (num_clusters - 1) / max(n - 1, 1)


# Returns the response closest to the average embedding, i.e. the most central answer.
def pick_best_response(responses):
    embeddings = embedder.encode(responses, normalize_embeddings=True)
    mean_vec = embeddings.mean(axis=0)
    mean_vec /= np.linalg.norm(mean_vec)
    return responses[np.argmax(np.dot(embeddings, mean_vec))]


# Averages the three scoring methods into a single confidence score.
def combined_score(em, cos, cluster):
    return (em + cos + cluster) / 3


# Prints a formatted summary for a single question including responses and scores.
def print_result(index, total, prompt, ground_truth, responses, best, em, cos, cluster):
    comb = combined_score(em, cos, cluster)
    confidence = "✅ High" if comb >= 0.75 else "⚠️ Medium" if comb >= 0.5 else "❌ Low"
    correct = is_correct(best, ground_truth)

    print(f"┌─ [{index}/{total}] {'─' * 50}")
    print(f"│ Prompt: {prompt[:82]}{'...' if len(prompt) > 82 else ''}")
    print(f"│ Ground Truth: {ground_truth}")
    print(f"│")
    for j, response in enumerate(responses):
        marker = " ◀ best" if response == best else ""
        print(f"│ [{index}{chr(ord('a') + j)}/{total}] {response}{marker}")
    print(f"│")
    print(f"│ Correct: {'✅ Yes' if correct else '❌ No'}")
    print(f"│ Confidence: {confidence} (combined: {comb:.2f})")
    print(f"│ Exact Match: {em:.2f} | Cosine: {cos:.2f} | Cluster: {cluster:.2f}")
    print(f"└{'─' * 66}\n")


# Writes all results to a CSV file including scores and AUROC values.
def save_results_csv(results, filename="results.csv"):
    fieldnames = [
        "question_id", "prompt", "selected_answer", "ground_truth", "is_correct",
        "all_responses", "em_score", "cosine_score", "cluster_score", "combined_score",
        "confidence_label", "auroc_em", "auroc_cosine", "auroc_cluster", "auroc_combined",
    ]

    labels = [r["correct"] for r in results]
    em_scores = [r["em_score"] for r in results]
    cos_scores = [r["cos_score"] for r in results]
    cluster_scores = [r["cluster_score"] for r in results]
    combined_scores = [combined_score(e, c, k) for e, c, k in zip(em_scores, cos_scores, cluster_scores)]

    if len(set(labels)) >= 2:
        auroc_em = roc_auc_score(labels, em_scores)
        auroc_cos = roc_auc_score(labels, cos_scores)
        auroc_cluster = roc_auc_score(labels, cluster_scores)
        auroc_combined = roc_auc_score(labels, combined_scores)
    else:
        auroc_em = auroc_cos = auroc_cluster = auroc_combined = float("nan")

    def fmt_auroc(val):
        return round(val, 4) if not np.isnan(val) else "N/A"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            em = r["em_score"]
            cos = r["cos_score"]
            cluster = r["cluster_score"]
            comb = combined_score(em, cos, cluster)
            confidence = "High" if comb >= 0.75 else "Medium" if comb >= 0.5 else "Low"
            writer.writerow({
                "question_id": r["question_id"],
                "prompt": r["prompt"],
                "selected_answer": r["selected_answer"],
                "ground_truth": r["ground_truth"],
                "is_correct": r["correct"],
                "all_responses": " | ".join(r["all_responses"]),
                "em_score": round(em, 4),
                "cosine_score": round(cos, 4),
                "cluster_score": round(cluster, 4),
                "combined_score": round(comb, 4),
                "confidence_label": confidence,
                "auroc_em": fmt_auroc(auroc_em),
                "auroc_cosine": fmt_auroc(auroc_cos),
                "auroc_cluster": fmt_auroc(auroc_cluster),
                "auroc_combined": fmt_auroc(auroc_combined),
            })

    print(f"💾 Results saved to {filename}")


# Plots ROC curves for each scoring method to compare their ability to predict correctness.
def plot_auroc(ax, labels, em_scores, cos_scores, cluster_scores):
    combined_scores = [combined_score(e, c, k) for e, c, k in zip(em_scores, cos_scores, cluster_scores)]
    style_axis(ax)

    for scores, name, color in [
        (em_scores, "Exact Match", C_EM),
        (cos_scores, "Cosine Sim", C_COS),
        (cluster_scores, "Semantic Cluster", C_CLUST),
        (combined_scores, "Combined", C_COMB),
    ]:
        auroc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--", label="Random (0.500)")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=[0, 1], ylim=[0, 1.02])
    ax.set_title("AUROC — UQ Score vs Correctness", fontsize=11)
    ax.legend(facecolor="#111", labelcolor=WHITE, fontsize=8)


# Shows violin plots of each score split by whether the answer was correct or not.
def plot_score_distributions(ax, labels, em_scores, cos_scores, cluster_scores):
    style_axis(ax)

    labels_arr = np.array(labels)
    metrics = [
        (em_scores, "EM", C_EM),
        (cos_scores, "Cosine", C_COS),
        (cluster_scores, "Cluster", C_CLUST),
    ]

    tick_positions, tick_labels = [], []
    for idx, (scores, name, color) in enumerate(metrics):
        arr = np.array(scores)
        pc = idx * 2 + 1
        pi = idx * 2 + 2
        for vals, pos, alpha in [(arr[labels_arr == 1], pc, 1.0), (arr[labels_arr == 0], pi, 0.5)]:
            if len(vals) > 1:
                vp = ax.violinplot(vals, positions=[pos], widths=0.7, showmedians=True)
                for body in vp['bodies']:
                    body.set_facecolor(color)
                    body.set_alpha(alpha)
                    body.set_edgecolor("#333")
                vp['cmedians'].set_color(WHITE)
                for part in ['cbars', 'cmins', 'cmaxes']:
                    vp[part].set_color(color)
        tick_positions.append((pc + pi) / 2)
        tick_labels.append(name)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set(ylabel="Score", ylim=[0, 1.05])
    ax.set_title("Score Distributions: Correct vs Incorrect", fontsize=11)
    ax.legend(
        handles=[Patch(facecolor=WHITE, alpha=1.0, label="Correct"), Patch(facecolor=WHITE, alpha=0.4, label="Incorrect")],
        facecolor="#111", labelcolor=WHITE, fontsize=8,
    )


# Shows accuracy per confidence tier (Low/Medium/High) to check if confidence is well calibrated.
def plot_confidence_accuracy(ax, labels, em_scores, cos_scores, cluster_scores):
    style_axis(ax)

    combined = np.array([combined_score(e, c, k) for e, c, k in zip(em_scores, cos_scores, cluster_scores)])
    labels_arr = np.array(labels)

    tiers = [
        ("Low\n(<0.5)", combined < 0.5, C_INCORRECT),
        ("Medium\n(0.5-0.75)", (combined >= 0.5) & (combined < 0.75), C_COS),
        ("High\n(>=0.75)", combined >= 0.75, C_CORRECT),
    ]

    for i, (label, mask, color) in enumerate(tiers):
        n = mask.sum()
        acc = labels_arr[mask].mean() if n > 0 else 0
        ax.bar(label, acc, color=color, width=0.5, edgecolor="#333", linewidth=0.8)
        ax.text(i, acc + 0.02, f"n={n}\n{acc:.0%}", ha='center', va='bottom', color=WHITE, fontsize=9)

    ax.axhline(np.mean(labels), color=DIM, lw=1.2, linestyle="--", label=f"Overall acc ({np.mean(labels):.0%})")
    ax.set(ylabel="Accuracy", ylim=[0, 1.15])
    ax.set_title("Calibration: Confidence Bucket vs Accuracy", fontsize=11)
    ax.legend(facecolor="#111", labelcolor=WHITE, fontsize=8)


# Plots combined confidence for every question in order, with a rolling average line.
def plot_score_scatter(ax, labels, em_scores, cos_scores, cluster_scores):
    style_axis(ax)

    combined = [combined_score(e, c, k) for e, c, k in zip(em_scores, cos_scores, cluster_scores)]
    colors = [C_CORRECT if l == 1 else C_INCORRECT for l in labels]
    ax.scatter(range(1, len(combined) + 1), combined, c=colors, s=18, alpha=0.8, linewidths=0)

    window = min(10, len(combined))
    rolling = np.convolve(combined, np.ones(window) / window, mode='valid')
    ax.plot(range(window, len(combined) + 1), rolling, color=C_COMB, lw=1.8, label=f"Rolling avg (w={window})")

    ax.axhline(0.75, color="#555", lw=1, linestyle="--")
    ax.axhline(0.5, color="#555", lw=1, linestyle="--")
    ax.text(len(combined) * 1.01, 0.76, "High", color=DIM, fontsize=7, va='bottom')
    ax.text(len(combined) * 1.01, 0.51, "Med", color=DIM, fontsize=7, va='bottom')
    ax.set(xlabel="Question Index", ylabel="Combined Confidence Score", ylim=[0, 1.05])
    ax.set_title("Confidence Over Questions", fontsize=11)
    ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C_CORRECT, markersize=7, label="Correct"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C_INCORRECT, markersize=7, label="Incorrect"),
            Line2D([0], [0], color=C_COMB, lw=2, label="Rolling avg"),
        ],
        facecolor="#111", labelcolor=WHITE, fontsize=8,
    )


# Scatterplots pairs of scoring methods against each other to show how much they agree.
def plot_score_correlation(ax, em_scores, cos_scores, cluster_scores):
    style_axis(ax)

    em = np.array(em_scores)
    cos = np.array(cos_scores)
    cl = np.array(cluster_scores)
    x_range = np.linspace(0, 1, 100)

    pairs = [
        (em, cos, C_EM, "EM vs Cosine"),
        (em, cl, C_CLUST, "EM vs Cluster"),
        (cos, cl, C_COS, "Cosine vs Cluster"),
    ]

    for x, y, color, label in pairs:
        ax.scatter(x, y, s=12, alpha=0.5, color=color, label=label)
        m, b = np.polyfit(x, y, 1)
        ax.plot(x_range, m * x_range + b, color=color, lw=1.2, alpha=0.7)

    ax.set(xlabel="Score (method A)", ylabel="Score (method B)", xlim=[0, 1.05], ylim=[0, 1.05])
    ax.set_title("UQ Method Agreement (Pairwise)", fontsize=11)
    ax.legend(facecolor="#111", labelcolor=WHITE, fontsize=8)


# Donut chart showing how many questions fell into each confidence tier.
def plot_confidence_pie(ax, labels, em_scores, cos_scores, cluster_scores):
    style_axis(ax)

    combined = np.array([combined_score(e, c, k) for e, c, k in zip(em_scores, cos_scores, cluster_scores)])
    labels_arr = np.array(labels)

    high = combined >= 0.75
    medium = (combined >= 0.5) & ~high
    low = combined < 0.5

    ax.pie(
        [high.sum(), medium.sum(), low.sum()],
        labels=[f"High\n({high.sum()})", f"Medium\n({medium.sum()})", f"Low\n({low.sum()})"],
        colors=[C_CORRECT, C_COS, C_INCORRECT],
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2),
        textprops=dict(color=WHITE, fontsize=9),
    )
    ax.text(0, 0, f"{labels_arr.mean():.0%}\naccuracy", ha='center', va='center', color=WHITE, fontsize=11, fontweight='bold')
    ax.set_title("Confidence Tier Distribution", fontsize=11)


# Renders and saves each plot individually, then assembles them all into a dashboard.
def plot_all(results):
    labels = [r["correct"] for r in results]
    em_scores = [r["em_score"] for r in results]
    cos_scores = [r["cos_score"] for r in results]
    cluster_scores = [r["cluster_score"] for r in results]
    has_both = len(set(labels)) >= 2

    individual_plots = [
        ("plot_distributions.png", (7, 5), lambda ax: plot_score_distributions(ax, labels, em_scores, cos_scores, cluster_scores)),
        ("plot_calibration.png", (7, 5), lambda ax: plot_confidence_accuracy(ax, labels, em_scores, cos_scores, cluster_scores)),
        ("plot_confidence_over_time.png", (9, 4), lambda ax: plot_score_scatter(ax, labels, em_scores, cos_scores, cluster_scores)),
        ("plot_method_agreement.png", (6, 6), lambda ax: plot_score_correlation(ax, em_scores, cos_scores, cluster_scores)),
        ("plot_confidence_tiers.png", (6, 6), lambda ax: plot_confidence_pie(ax, labels, em_scores, cos_scores, cluster_scores)),
    ]

    if has_both:
        individual_plots.insert(0, (
            "plot_auroc.png", (7, 6),
            lambda ax: plot_auroc(ax, labels, em_scores, cos_scores, cluster_scores),
        ))

    for filename, figsize, draw in individual_plots:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BG)
        draw(ax)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.show()
        print(f"📊 Saved {filename}")

    def auroc_placeholder(ax):
        ax.text(0.5, 0.5, "Need both correct\n& incorrect answers\nfor AUROC",
                ha='center', va='center', color=DIM, transform=ax.transAxes)

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle("UQ Evaluation Dashboard", color=WHITE, fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    panels = [
        lambda ax: plot_auroc(ax, labels, em_scores, cos_scores, cluster_scores) if has_both else auroc_placeholder(ax),
        lambda ax: plot_score_distributions(ax, labels, em_scores, cos_scores, cluster_scores),
        lambda ax: plot_confidence_accuracy(ax, labels, em_scores, cos_scores, cluster_scores),
        lambda ax: plot_score_scatter(ax, labels, em_scores, cos_scores, cluster_scores),
        lambda ax: plot_score_correlation(ax, em_scores, cos_scores, cluster_scores),
        lambda ax: plot_confidence_pie(ax, labels, em_scores, cos_scores, cluster_scores),
    ]

    for (row, col), draw in zip([(r, c) for r in range(2) for c in range(3)], panels):
        ax = fig.add_subplot(gs[row, col])
        style_axis(ax)
        draw(ax)

    plt.savefig("dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print("📊 Saved dashboard.png")


# Fetches all responses for a single question, scores them, and appends the result.
async def generate_and_score(index, prompt, ground_truth, total, prev_event, my_event, results):
    responses = [await fetch_response(prompt) for _ in range(NUM_RESPONSES)]
    best = pick_best_response(responses)
    em = exact_match_score(responses)
    cos = cosine_similarity_score(responses)
    cluster = semantic_cluster_score(responses)

    results.append({
        "question_id": index,
        "prompt": prompt,
        "selected_answer": best,
        "ground_truth": ground_truth,
        "correct": int(is_correct(best, ground_truth)),
        "all_responses": responses,
        "em_score": em,
        "cos_score": cos,
        "cluster_score": cluster,
    })

    await prev_event.wait()
    print_result(index, total, prompt, ground_truth, responses, best, em, cos, cluster)
    my_event.set()


# Runs all questions concurrently and collects results, printing them in order as they finish.
async def run_all(prompts, ground_truths):
    total = len(prompts)
    api_calls = total * NUM_RESPONSES
    est_secs = api_calls * 60 // RPM_LIMIT
    results = []

    print(f"Prompts: {total}")
    print(f"Responses: {NUM_RESPONSES} per prompt ({api_calls} total API calls)")
    print(f"Rate Limit: {RPM_LIMIT} RPM")
    print(f"Est. Time: ~{est_secs}s ({est_secs // 60}m {est_secs % 60}s)\n")
    print("=" * 66 + "\nLIVE RESULTS\n" + "=" * 66 + "\n")

    events = [asyncio.Event() for _ in range(total + 1)]
    events[0].set()

    await asyncio.gather(*[
        generate_and_score(i + 1, prompt, gt, total, events[i], events[i + 1], results)
        for i, (prompt, gt) in enumerate(zip(prompts, ground_truths))
    ])

    return results


# Entry point: loads questions from a CSV, runs evaluation, prints summary, and saves outputs.
def run_prompts(csv_file):
    prompts, ground_truth = load_prompts(csv_file)

    print("=" * 66 + "\nGENERATING & SCORING\n" + "=" * 66 + "\n")
    results = asyncio.run(run_all(prompts, ground_truth))

    labels = [r["correct"] for r in results]
    print("=" * 66 + "\nEVALUATION\n" + "=" * 66 + "\n")
    print(f"Correct: {sum(labels)}/{len(labels)} ({sum(labels) / len(labels):.1%})\n")

    plot_all(results)
    save_results_csv(results)

    print("\n" + "=" * 66 + "\nDONE\n" + "=" * 66)


run_prompts("SVAMP_200.csv")