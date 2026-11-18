v0_te_mask = v0_data["splits"] == "test"
ds_base_te = CharDataset(v0_data["imgs"][v0_te_mask], v0_data["labels"][v0_te_mask], is_train=False)
base_te_loader = DataLoader(ds_base_te, batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False)
base_acc, base_f1, _, _ = evaluate_char_model(baseline_model, base_te_loader, num_classes=84)

ds_eng_te = CharDataset(eng_data["test_X"], eng_data["test_y"], is_train=False)
eng_te_loader = DataLoader(ds_eng_te, batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False)

logits_m1_te, y_te_t = collect_logits(model1_resnet, eng_te_loader)
logits_m2_te, _      = collect_logits(model2_mobile, eng_te_loader)

p1_te = F.softmax(logits_m1_te / T1, dim=1)
p2_te = F.softmax(logits_m2_te / T2, dim=1)
p_ens_te = best_w * p1_te + (1.0 - best_w) * p2_te

char84_mask = y_te_t < 84
def char_metrics(probs, y_true):
    p84, y84 = probs[char84_mask, :84], y_true[char84_mask]
    top1 = p84.argmax(dim=1).eq(y84).float().mean().item()
    top3 = torch.topk(p84, k=3, dim=1).indices.eq(y84.unsqueeze(1)).any(dim=1).float().mean().item()
    return top1, top3, compute_ece(p84, y84)

m1_top1, m1_top3, m1_ece = char_metrics(p1_te, y_te_t)
m2_top1, m2_top3, m2_ece = char_metrics(p2_te, y_te_t)
ens_top1, ens_top3, ens_ece = char_metrics(p_ens_te, y_te_t)

basic_mask = y_te_t < 50
num_mask = (y_te_t >= 50) & (y_te_t < 60)
comp_mask = (y_te_t >= 60) & (y_te_t < 84)
basic_acc = p_ens_te[basic_mask, :84].argmax(dim=1).eq(y_te_t[basic_mask]).float().mean().item() if basic_mask.any() else 0.0
digit_acc = p_ens_te[num_mask, :84].argmax(dim=1).eq(y_te_t[num_mask]).float().mean().item() if num_mask.any() else 0.0
comp_acc  = p_ens_te[comp_mask, :84].argmax(dim=1).eq(y_te_t[comp_mask]).float().mean().item() if comp_mask.any() else 0.0

# Writer & Form-based Marking.xlsx quality tier analysis on unseen test writers
test_w_ids = eng_data["test_writers"][char84_mask.numpy()]
test_quals = eng_data["test_qualities"][char84_mask.numpy()]
ens_correct_84 = p_ens_te[char84_mask, :84].argmax(dim=1).eq(y_te_t[char84_mask]).numpy()

df_w_eval = pd.DataFrame({"writer_id": test_w_ids, "quality": test_quals, "correct": ens_correct_84})
writer_acc_series = df_w_eval.groupby("writer_id")["correct"].mean()

q_low_acc  = df_w_eval[df_w_eval["quality"] <= 1.5]["correct"].mean() if (df_w_eval["quality"] <= 1.5).any() else ens_top1
q_med_acc  = df_w_eval[(df_w_eval["quality"] > 1.5) & (df_w_eval["quality"] < 3.0)]["correct"].mean() if ((df_w_eval["quality"] > 1.5) & (df_w_eval["quality"] < 3.0)).any() else ens_top1
q_high_acc = df_w_eval[df_w_eval["quality"] >= 3.0]["correct"].mean() if (df_w_eval["quality"] >= 3.0).any() else ens_top1

@torch.no_grad()
def eval_crnn_mode(mode_name, n_eval=300):
    if CFG["FAST_DEV_RUN"]:
        n_eval = 60
    correct = 0
    for _ in range(n_eval):
        im, _, _, gt_str, _ = word_engine_test.synthesize(1.0, force_mode=mode_name)
        x = torch.from_numpy(im).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
        pred_str, _ = ctc_greedy_decode(model3_crnn(x))[0]
        correct += int(pred_str == gt_str)
    return correct / max(1, n_eval)

num_exact_acc = eval_crnn_mode("number")
tier_a_acc    = eval_crnn_mode("tier_a")
tier_b_acc    = eval_crnn_mode("tier_b")

cpu_m1, cpu_m2, cpu_m3 = LekhaResNet(85).cpu().eval(), LekhaMobile(85).cpu().eval(), LekhaCRNN().cpu().eval()
dummy_char, dummy_word = torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 256)

def bench_ms(fn, x, runs=20):
    for _ in range(3): fn(x)
    t0 = time.perf_counter()
    for _ in range(runs): fn(x)
    return ((time.perf_counter() - t0) / runs) * 1000.0

lat_m1, lat_m2, lat_m3 = bench_ms(cpu_m1, dummy_char), bench_ms(cpu_m2, dummy_char), bench_ms(cpu_m3, dummy_word)

METRICS_SUMMARY = {
    "baseline": {"char_top1": round(base_acc * 100, 2)},
    "model1_resnet": {"char_top1": round(m1_top1 * 100, 2), "char_top3": round(m1_top3 * 100, 2), "ece": round(m1_ece * 100, 2), "cpu_ms": round(lat_m1, 2)},
    "model2_mobile": {"char_top1": round(m2_top1 * 100, 2), "char_top3": round(m2_top3 * 100, 2), "ece": round(m2_ece * 100, 2), "cpu_ms": round(lat_m2, 2)},
    "ensemble_1_2": {
        "char_top1": round(ens_top1 * 100, 2), "char_top3": round(ens_top3 * 100, 2), "ece": round(ens_ece * 100, 2),
        "basic_acc": round(basic_acc * 100, 2), "digit_acc": round(digit_acc * 100, 2), "compound_acc": round(comp_acc * 100, 2),
        "writer_mean_acc": round(float(writer_acc_series.mean()) * 100, 2),
        "writer_p10_acc": round(float(np.percentile(writer_acc_series, 10)) * 100, 2),
        "form_quality_low_acc": round(float(q_low_acc) * 100, 2),
        "form_quality_med_acc": round(float(q_med_acc) * 100, 2),
        "form_quality_high_acc": round(float(q_high_acc) * 100, 2),
        "cpu_ms": round(lat_m1 + lat_m2, 2)
    },
    "model3_crnn": {"number_exact": round(num_exact_acc * 100, 2), "word_tier_a": round(tier_a_acc * 100, 2), "word_tier_b": round(tier_b_acc * 100, 2), "cpu_ms": round(lat_m3, 2)},
}

with open(REL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(METRICS_SUMMARY, f, indent=2)

html_report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Bangla HWR — Final Evaluation Report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:30px auto;padding:0 20px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #ccc;padding:10px;text-align:center}}
th{{background:#f4f6f8}}</style></head><body>
<h1>Bangla Handwriting Recognition — Evaluation Report</h1>
<p>Dataset: <code>imohammedali/banglalekha-isolated</code> · Strict Writer-Independent Split (Unseen Test Forms)</p>
<table>
<tr><th>Metric</th><th>Baseline (V0)</th><th>Model 1 (ResNet)</th><th>Model 2 (Mobile)</th><th>Ensemble (1 + 2)</th><th>Model 3 (CRNN)</th></tr>
<tr><td>Character Top-1 (%)</td><td>{METRICS_SUMMARY['baseline']['char_top1']}%</td><td>{METRICS_SUMMARY['model1_resnet']['char_top1']}%</td><td>{METRICS_SUMMARY['model2_mobile']['char_top1']}%</td><td><b>{METRICS_SUMMARY['ensemble_1_2']['char_top1']}%</b></td><td>—</td></tr>
<tr><td>Character Top-3 (%)</td><td>—</td><td>{METRICS_SUMMARY['model1_resnet']['char_top3']}%</td><td>{METRICS_SUMMARY['model2_mobile']['char_top3']}%</td><td><b>{METRICS_SUMMARY['ensemble_1_2']['char_top3']}%</b></td><td>—</td></tr>
<tr><td>Calibration Error ECE (%)</td><td>—</td><td>{METRICS_SUMMARY['model1_resnet']['ece']}%</td><td>{METRICS_SUMMARY['model2_mobile']['ece']}%</td><td><b>{METRICS_SUMMARY['ensemble_1_2']['ece']}%</b></td><td>—</td></tr>
<tr><td>Number Exact Match (%)</td><td>—</td><td>—</td><td>—</td><td>—</td><td><b>{METRICS_SUMMARY['model3_crnn']['number_exact']}%</b></td></tr>
<tr><td>Joined Word Accuracy Tier-A (%)</td><td>—</td><td>—</td><td>—</td><td>—</td><td><b>{METRICS_SUMMARY['model3_crnn']['word_tier_a']}%</b></td></tr>
<tr><td>Joined Word Accuracy Tier-B (%)</td><td>—</td><td>—</td><td>—</td><td>—</td><td><b>{METRICS_SUMMARY['model3_crnn']['word_tier_b']}%</b></td></tr>
<tr><td>CPU Latency (ms)</td><td>—</td><td>{METRICS_SUMMARY['model1_resnet']['cpu_ms']} ms</td><td>{METRICS_SUMMARY['model2_mobile']['cpu_ms']} ms</td><td>{METRICS_SUMMARY['ensemble_1_2']['cpu_ms']} ms</td><td>{METRICS_SUMMARY['model3_crnn']['cpu_ms']} ms</td></tr>
</table>
<h3>Breakdown by Character Group & Human Rater Form Quality (<code>Form-based Marking.xlsx</code>)</h3>
<p><b>Basic Characters (1–50):</b> {METRICS_SUMMARY['ensemble_1_2']['basic_acc']}% | <b>Numerals (51–60):</b> {METRICS_SUMMARY['ensemble_1_2']['digit_acc']}% | <b>Compound Conjuncts (61–84):</b> {METRICS_SUMMARY['ensemble_1_2']['compound_acc']}%</p>
<p><b>High-Legibility Forms (Score ≥ 3.0):</b> {METRICS_SUMMARY['ensemble_1_2']['form_quality_high_acc']}% | <b>Medium Forms (1.5–3.0):</b> {METRICS_SUMMARY['ensemble_1_2']['form_quality_med_acc']}% | <b>Low-Legibility Forms (≤ 1.5):</b> {METRICS_SUMMARY['ensemble_1_2']['form_quality_low_acc']}%</p>
</body></html>"""

with open(REL_DIR / "report.html", "w", encoding="utf-8") as f:
    f.write(html_report)
with open(WORK_PATH / "report.html", "w", encoding="utf-8") as f:
    f.write(html_report)

print("Evaluation Complete:", json.dumps(METRICS_SUMMARY, indent=2))