import re
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
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# CSS2 UQM Research Project

API_KEY = os.getenv("NVIDIA_API_KEY") or "GET NAVIDIA API KEY TO USE THIS CODE :D"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "openai/gpt-oss-120b"
NUM_RESPONSES = 5
RPM_LIMIT = 35
MAX_RETRIES = 5

IS_CORRECT_THRESHOLD = 0.5
IS_CORRECT_EMBED_THRESH = 0.5

BG = "#0f0f0f"
PANEL = "#1a1a1a"
WHITE = "#e8e8e8"
DIM = "#888888"

C_F1 = "#00d4ff"
C_COS = "#ff6b6b"
C_CLUST = "#a8ff78"
C_LEX = "#d4aaff"
C_COMB = "#f5a623"
C_OK = "#a8ff78"
C_FAIL = "#ff6b6b"

METHODS = [
    ("Token F1", C_F1),
    ("Cosine Sim", C_COS),
    ("Semantic Cluster", C_CLUST),
    ("Lexical Sim", C_LEX),
    ("Combined", C_COMB),
]

if not API_KEY:
    raise ValueError("No API key found. Set NVIDIA_API_KEY in your .env file.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-large")


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


# Strips punctuation and lowercases text for fair comparison.
def clean(text):
    return re.sub(r'[^\w\s]', '', str(text).lower().strip())


# Returns True if the model answer matches the ground truth via substring, semantic similarity, or token F1.
def is_correct(model_answer, ground_truth):
    a, b = clean(model_answer), clean(ground_truth)

    if a in b or b in a:
        return True

    if float(cross_encoder.predict([(a, b)])[0]) >= IS_CORRECT_EMBED_THRESH:
        return True

    a_tokens = set(a.split())
    b_tokens = set(b.split())
    common = a_tokens & b_tokens
    if not common:
        return False

    precision = len(common) / len(a_tokens)
    recall = len(common) / len(b_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= IS_CORRECT_THRESHOLD


# Sends a prompt to the model and returns the response, retrying on failure.
async def fetch_response(prompt):
    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Answer concisely and correctly using only 1 word to answer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            if not resp or not resp.choices:
                await asyncio.sleep(2)
                continue

            msg = resp.choices[0].message
            if msg.content:
                return msg.content.strip()

            if getattr(msg, "reasoning_content", None):
                lines = [l.strip() for l in msg.reasoning_content.strip().splitlines() if l.strip()]
                return lines[-1] if lines else "No answer"

            return "No answer"

        except RateLimitError:
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            print(f"⚠️ {e}, retrying in 3s...")
            await asyncio.sleep(3)

    return "No answer"


# Measures how similar the responses are to each other using token-level F1.
def token_f1_score(responses):
    def f1(a, b):
        a, b = set(a.lower().split()), set(b.lower().split())
        common = a & b
        if not common:
            return 0.0
        precision = len(common) / len(a)
        recall = len(common) / len(b)
        return 2 * precision * recall / (precision + recall)

    n = len(responses)
    scores = [f1(responses[i], responses[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(scores)) if scores else 1.0


# Measures response agreement using cosine similarity of sentence embeddings.
def cosine_similarity_score(responses):
    embeddings = embedder.encode(responses, normalize_embeddings=True)
    sim_matrix = np.dot(embeddings, embeddings.T)
    off_diag = sim_matrix[~np.eye(len(responses), dtype=bool)]
    return float(off_diag.mean())


# Groups responses into semantic clusters and returns a score based on how few clusters there are.
def semantic_cluster_score(responses):
    embeddings = embedder.encode(responses, normalize_embeddings=True)
    n = len(responses)
    cluster_id = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if np.dot(embeddings[i], embeddings[j]) >= 0.85:
                old = cluster_id[j]
                new = cluster_id[i]
                cluster_id = [new if x == old else x for x in cluster_id]

    num_clusters = len(set(cluster_id))
    return 1.0 - (num_clusters - 1) / max(n - 1, 1)


# Measures word-level overlap between all response pairs using Jaccard similarity.
def lexical_similarity_score(responses):
    token_sets = [set(r.lower().split()) for r in responses]
    n = len(responses)
    scores = [
        len(token_sets[i] & token_sets[j]) / len(token_sets[i] | token_sets[j])
        for i in range(n)
        for j in range(i + 1, n)
        if token_sets[i] | token_sets[j]
    ]
    return float(np.mean(scores)) if scores else 1.0


# Returns the response closest to the average embedding, i.e. the most central answer.
def pick_best_response(responses):
    embeddings = embedder.encode(responses, normalize_embeddings=True)
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    best_index = np.argmax(np.dot(embeddings, centroid))
    return responses[best_index]


# Averages the four scoring methods into a single confidence score.
def combined_score(f1, cos, cluster, lex):
    return (f1 + cos + cluster + lex) / 4


# Prints a formatted summary for a single question including responses and scores.
def print_result(index, total, prompt, ground_truth, responses, best, scores, votes, correct):
    f1, cos, cluster, lex = scores
    comb = combined_score(f1, cos, cluster, lex)
    tier = "High" if comb >= 0.75 else "Medium" if comb >= 0.5 else "Low"

    print(f"┌─ [{index}/{total}] {'─' * 50}")
    print(f"│ Prompt: {prompt[:82]}{'...' if len(prompt) > 82 else ''}")
    print(f"│ Ground Truth: {ground_truth}")
    print(f"│")
    for j, r in enumerate(responses):
        marker = " ◀ best" if r == best else ""
        print(f"│ [{index}{chr(ord('a') + j)}/{total}] {r}{marker}")
    print(f"│")
    print(f"│ Correct: {'Yes' if correct else 'No'} ({votes}/{NUM_RESPONSES} responses correct)")
    print(f"│ Confidence: {tier} (combined: {comb:.2f})")
    print(f"│ Token F1: {f1:.2f} | Cosine: {cos:.2f} | Cluster: {cluster:.2f} | Lexical: {lex:.2f}")
    print(f"└{'─' * 66}\n")


# Writes all results to a CSV file including scores and AUROC values.
def save_csv(results, filename="results.csv"):
    labels = [r["correct"] for r in results]
    combined_list = [combined_score(r["f1"], r["cos"], r["cluster"], r["lex"]) for r in results]
    has_both = len(set(labels)) >= 2

    def auroc(scores):
        return round(roc_auc_score(labels, scores), 4) if has_both else "N/A"

    auroc_values = {
        "f1": auroc([r["f1"] for r in results]),
        "cos": auroc([r["cos"] for r in results]),
        "cluster": auroc([r["cluster"] for r in results]),
        "lex": auroc([r["lex"] for r in results]),
        "combined": auroc(combined_list),
    }

    fields = [
        "question_id", "prompt", "selected_answer", "ground_truth", "is_correct", "votes",
        "all_responses", "f1_score", "cosine_score", "cluster_score", "lexical_score",
        "combined_score", "confidence_label",
        "auroc_f1", "auroc_cosine", "auroc_cluster", "auroc_lexical", "auroc_combined",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r, comb in zip(results, combined_list):
            confidence = "High" if comb >= 0.75 else "Medium" if comb >= 0.5 else "Low"
            writer.writerow({
                "question_id": r["question_id"],
                "prompt": r["prompt"],
                "selected_answer": r["selected_answer"],
                "ground_truth": r["ground_truth"],
                "is_correct": r["correct"],
                "votes": r["votes"],
                "all_responses": " | ".join(r["all_responses"]),
                "f1_score": round(r["f1"], 4),
                "cosine_score": round(r["cos"], 4),
                "cluster_score": round(r["cluster"], 4),
                "lexical_score": round(r["lex"], 4),
                "combined_score": round(comb, 4),
                "confidence_label": confidence,
                "auroc_f1": auroc_values["f1"],
                "auroc_cosine": auroc_values["cos"],
                "auroc_cluster": auroc_values["cluster"],
                "auroc_lexical": auroc_values["lex"],
                "auroc_combined": auroc_values["combined"],
            })

    print(f"💾 Results saved to {filename}")


# Applies dark theme styling to a matplotlib axis.
def style_axis(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_color("#333")


# Extracts all score lists and labels from results for use in plot functions.
def unpack(results):
    f1s = [r["f1"] for r in results]
    coss = [r["cos"] for r in results]
    clusters = [r["cluster"] for r in results]
    lexs = [r["lex"] for r in results]
    labels = [r["correct"] for r in results]
    combs = [combined_score(f, c, k, l) for f, c, k, l in zip(f1s, coss, clusters, lexs)]
    return f1s, coss, clusters, lexs, combs, labels


# Plots ROC curves for each scoring method so you can compare their ability to predict correctness.
def plot_auroc(ax, results):
    f1s, coss, clusters, lexs, combs, labels = unpack(results)
    style_axis(ax)

    for scores, (name, color) in zip([f1s, coss, clusters, lexs, combs], METHODS):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--", label="Random (0.500)")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=[0, 1], ylim=[0, 1.02])
    ax.set_title("AUROC — UQ Score vs Correctness", fontsize=11)
    ax.legend(facecolor="#111", labelcolor=WHITE, fontsize=7)


# Shows violin plots of each score split by whether the answer was correct or not.
def plot_distributions(ax, results):
    f1s, coss, clusters, lexs, _, labels = unpack(results)
    style_axis(ax)

    labels = np.array(labels)
    metrics = [
        (f1s, "F1", C_F1),
        (coss, "Cosine", C_COS),
        (clusters, "Cluster", C_CLUST),
        (lexs, "Lexical", C_LEX),
    ]

    for idx, (scores, name, color) in enumerate(metrics):
        arr = np.array(scores)
        for vals, pos, alpha in [(arr[labels == 1], idx * 2 + 1, 1.0), (arr[labels == 0], idx * 2 + 2, 0.5)]:
            if len(vals) > 1:
                vp = ax.violinplot(vals, positions=[pos], widths=0.7, showmedians=True)
                for body in vp['bodies']:
                    body.set_facecolor(color)
                    body.set_alpha(alpha)
                    body.set_edgecolor("#333")
                vp['cmedians'].set_color(WHITE)
                for part in ['cbars', 'cmins', 'cmaxes']:
                    vp[part].set_color(color)

    ax.set_xticks([1.5, 3.5, 5.5, 7.5])
    ax.set_xticklabels(["F1", "Cosine", "Cluster", "Lexical"])
    ax.set(ylabel="Score", ylim=[0, 1.05])
    ax.set_title("Score Distributions: Correct vs Incorrect", fontsize=11)
    ax.legend(
        handles=[Patch(facecolor=WHITE, alpha=1.0, label="Correct"), Patch(facecolor=WHITE, alpha=0.4, label="Incorrect")],
        facecolor="#111", labelcolor=WHITE, fontsize=8,
    )


# Shows accuracy per confidence tier (Low/Medium/High) to check if confidence is well calibrated.
def plot_calibration(ax, results):
    _, _, _, _, combs, labels = unpack(results)
    style_axis(ax)

    combs = np.array(combs)
    labels = np.array(labels)
    tiers = [
        ("Low\n(<0.5)", combs < 0.5, C_FAIL),
        ("Medium\n(0.5-0.75)", (combs >= 0.5) & (combs < 0.75), C_COS),
        ("High\n(>=0.75)", combs >= 0.75, C_OK),
    ]

    for i, (label, mask, color) in enumerate(tiers):
        n = mask.sum()
        acc = labels[mask].mean() if n > 0 else 0
        ax.bar(label, acc, color=color, width=0.5, edgecolor="#333", linewidth=0.8)
        ax.text(i, acc + 0.02, f"n={n}\n{acc:.0%}", ha='center', va='bottom', color=WHITE, fontsize=9)

    ax.axhline(labels.mean(), color=DIM, lw=1.2, linestyle="--", label=f"Overall acc ({labels.mean():.0%})")
    ax.set(ylabel="Accuracy", ylim=[0, 1.15])
    ax.set_title("Calibration: Confidence Tier vs Accuracy", fontsize=11)
    ax.legend(facecolor="#111", labelcolor=WHITE, fontsize=8)


# Plots combined confidence for every question in order, with a rolling average line.
def plot_scatter(ax, results):
    _, _, _, _, combs, labels = unpack(results)
    style_axis(ax)

    colors = [C_OK if l == 1 else C_FAIL for l in labels]
    ax.scatter(range(1, len(combs) + 1), combs, c=colors, s=18, alpha=0.8, linewidths=0)

    window = min(10, len(combs))
    rolling = np.convolve(combs, np.ones(window) / window, mode='valid')
    ax.plot(range(window, len(combs) + 1), rolling, color=C_COMB, lw=1.8, label=f"Rolling avg (w={window})")

    ax.axhline(0.75, color="#555", lw=1, linestyle="--")
    ax.axhline(0.50, color="#555", lw=1, linestyle="--")
    ax.text(len(combs) * 1.01, 0.76, "High", color=DIM, fontsize=7)
    ax.text(len(combs) * 1.01, 0.51, "Med", color=DIM, fontsize=7)
    ax.set(xlabel="Question Index", ylabel="Combined Confidence", ylim=[0, 1.05])
    ax.set_title("Confidence Over Questions", fontsize=11)
    ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C_OK, markersize=7, label="Correct"),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C_FAIL, markersize=7, label="Incorrect"),
            Line2D([0], [0], color=C_COMB, lw=2, label="Rolling avg"),
        ],
        facecolor="#111", labelcolor=WHITE, fontsize=8,
    )


# Scatterplots pairs of scoring methods against each other to show how much they agree.
def plot_correlation(ax, results):
    f1s, coss, clusters, lexs, _, _ = unpack(results)
    style_axis(ax)

    f1s = np.array(f1s)
    coss = np.array(coss)
    clusters = np.array(clusters)
    lexs = np.array(lexs)
    x_range = np.linspace(0, 1, 100)

    pairs = [
        (f1s, coss, C_F1, "F1 vs Cosine"),
        (f1s, clusters, C_CLUST, "F1 vs Cluster"),
        (f1s, lexs, C_LEX, "F1 vs Lexical"),
        (coss, lexs, C_COS, "Cosine vs Lexical"),
    ]

    for x, y, color, label in pairs:
        ax.scatter(x, y, s=12, alpha=0.5, color=color, label=label)
        m, b = np.polyfit(x, y, 1)
        ax.plot(x_range, m * x_range + b, color=color, lw=1.2, alpha=0.7)

    ax.set(xlabel="Score (method A)", ylabel="Score (method B)", xlim=[0, 1.05], ylim=[0, 1.05])
    ax.set_title("UQ Method Agreement", fontsize=11)
    ax.legend(facecolor="#111", labelcolor=WHITE, fontsize=7)


# Donut chart showing how many questions fell into each confidence tier.
def plot_pie(ax, results):
    _, _, _, _, combs, labels = unpack(results)
    style_axis(ax)

    combs = np.array(combs)
    high = (combs >= 0.75).sum()
    medium = ((combs >= 0.5) & (combs < 0.75)).sum()
    low = (combs < 0.5).sum()
    overall = np.mean(labels)

    ax.pie(
        [high, medium, low],
        labels=[f"High\n({high})", f"Medium\n({medium})", f"Low\n({low})"],
        colors=[C_OK, C_COS, C_FAIL],
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2),
        textprops=dict(color=WHITE, fontsize=9),
    )
    ax.text(0, 0, f"{overall:.0%}\naccuracy", ha='center', va='center', color=WHITE, fontsize=11, fontweight='bold')
    ax.set_title("Confidence Tier Distribution", fontsize=11)


# Assembles all six plots into a single dashboard figure and saves it.
def plot_dashboard(results):
    has_both = len(set(r["correct"] for r in results)) >= 2

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle("UQ Evaluation Dashboard", color=WHITE, fontsize=16, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def auroc_placeholder(ax):
        ax.text(0.5, 0.5, "Need correct &\nincorrect answers\nfor AUROC",
                ha='center', va='center', color=DIM, transform=ax.transAxes)

    panels = [
        lambda ax: plot_auroc(ax, results) if has_both else auroc_placeholder(ax),
        lambda ax: plot_distributions(ax, results),
        lambda ax: plot_calibration(ax, results),
        lambda ax: plot_scatter(ax, results),
        lambda ax: plot_correlation(ax, results),
        lambda ax: plot_pie(ax, results),
    ]

    positions = [(row, col) for row in range(2) for col in range(3)]
    for (row, col), draw in zip(positions, panels):
        ax = fig.add_subplot(gs[row, col])
        style_axis(ax)
        draw(ax)

    plt.savefig("dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print("📊 Saved dashboard.png")


# Fetches all responses for a single question, scores them, and appends the result.
async def process_question(index, prompt, ground_truth, total, prev_event, my_event, results):
    responses = [await fetch_response(prompt) for _ in range(NUM_RESPONSES)]
    best = pick_best_response(responses)

    f1 = token_f1_score(responses)
    cos = cosine_similarity_score(responses)
    cluster = semantic_cluster_score(responses)
    lex = lexical_similarity_score(responses)

    votes = sum(is_correct(r, ground_truth) for r in responses)
    correct = int(votes > NUM_RESPONSES / 2)

    results.append({
        "question_id": index,
        "prompt": prompt,
        "selected_answer": best,
        "ground_truth": ground_truth,
        "correct": correct,
        "votes": f"{votes}/{NUM_RESPONSES}",
        "all_responses": responses,
        "f1": f1,
        "cos": cos,
        "cluster": cluster,
        "lex": lex,
    })

    await prev_event.wait()
    print_result(index, total, prompt, ground_truth, responses, best, (f1, cos, cluster, lex), votes, correct)
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
        process_question(i + 1, p, gt, total, events[i], events[i + 1], results)
        for i, (p, gt) in enumerate(zip(prompts, ground_truths))
    ])

    return results


# Entry point: loads questions from a CSV, runs evaluation, prints summary, and saves outputs.
def run(csv_file):
    prompts, ground_truths = [], []
    with open(csv_file, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            prompts.append(row['prompt'])
            ground_truths.append(row.get('Answer', row.get('answer', 'N/A')))

    print("=" * 66 + "\nGENERATING & SCORING\n" + "=" * 66 + "\n")
    results = asyncio.run(run_all(prompts, ground_truths))

    labels = [r["correct"] for r in results]
    combined_list = [combined_score(r["f1"], r["cos"], r["cluster"], r["lex"]) for r in results]

    print("=" * 66 + "\nEVALUATION SUMMARY\n" + "=" * 66 + "\n")
    print(f"Correct: {sum(labels)}/{len(labels)} ({sum(labels) / len(labels):.1%})\n")

    if len(set(labels)) >= 2:
        all_scores = [
            [r["f1"] for r in results],
            [r["cos"] for r in results],
            [r["cluster"] for r in results],
            [r["lex"] for r in results],
            combined_list,
        ]
        print("AUROC:")
        for scores, (name, _) in zip(all_scores, METHODS):
            print(f"  {name:<20} {roc_auc_score(labels, scores):.3f}")
        print()

    plot_dashboard(results)
    save_csv(results)
    print("\n" + "=" * 66 + "\nDONE\n" + "=" * 66)


run("prompts.csv")