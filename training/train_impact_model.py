# train_impact_model.py
# Fine-tunes FinBERT to predict market "impact" from headlines.
# - --dataset avisheksood  → uses Sentiment_Stock_data*.csv (headline-level)
#   Mapping: positive → 1 (UP), others → 0 (NON_UP)
# - --dataset djia         → uses Combined_News_DJIA*.csv (day-level up/down)
#   Modes: aggregate (Top1..Top25 concatenated) or per_headline_weak (weak supervision)

import os, argparse, json, math, glob
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed

TEXT_CANDS  = ["Sentence","headline","Headlines","headline_text","title","news","text"]
LABEL_CANDS = ["Sentiment","label","Label","target","y"]

def find_col(df, candidates, kind):
    for c in candidates:
        if c in df.columns: return c
    raise ValueError(f"Could not find a {kind} column. Tried: {candidates}. Found: {list(df.columns)}")

def plot_confusion(cm, out_png, labels):
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest'); plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(labels)); plt.xticks(ticks, labels); plt.yticks(ticks, labels)
    thr = cm.max()/2. if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j],'d'), ha="center", color="white" if cm[i,j]>thr else "black")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close(fig)

def default_tokenizer(max_length=160):
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    def fn(batch): return tok(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    return fn

def compute_metrics_binary(p):
    preds = np.argmax(p.predictions, axis=1); y = p.label_ids
    acc = accuracy_score(y, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

def newest(paths):
    paths = [p for p in (paths or []) if os.path.isfile(p)]
    if not paths: return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def resolve_csv_path(dataset, explicit_csv=None):
    if explicit_csv: return explicit_csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [script_dir, os.path.join(script_dir, "data")]
    if dataset == "avisheksood":
        patterns = ["Sentiment_Stock_data*.csv","sentiment_stock_data*.csv","avisheksood*.csv",
                    "stock_news_sentiment*.csv","*avishek*stock*sentiment*.csv"]
    elif dataset == "djia":
        patterns = ["Combined_News_DJIA.csv","combined_news_djia*.csv"]
    else:
        raise ValueError("Unknown dataset")
    cands = []
    for d in search_dirs:
        for pat in patterns:
            cands.extend(glob.glob(os.path.join(d, pat)))
    chosen = newest(cands)
    if not chosen:
        raise FileNotFoundError(
            f"Could not auto-find CSV for dataset '{dataset}'. Place the file in {search_dirs} "
            f"matching one of: {patterns} or pass --csv PATH explicitly."
        )
    print(f"[INFO] Using CSV: {chosen}")
    return chosen

def load_avisheksood(csv_path):
    df = pd.read_csv(csv_path)
    text_col = find_col(df, TEXT_CANDS, "text")
    label_col = find_col(df, LABEL_CANDS, "label")
    df = df[[text_col,label_col]].dropna()
    df["text"] = df[text_col].astype(str)
    def to_up(v):
        if isinstance(v,str):
            s=v.strip().lower()
            if s in ("pos","positive","+","1","true","up","bullish"): return 1
            return 0
        try: return 1 if int(v)==1 else 0
        except: return 0
    df["label"] = df[label_col].apply(to_up).astype(int)
    return df[["text","label"]], {0:"NON_UP", 1:"UP"}

def load_djia(csv_path, mode="aggregate"):
    df = pd.read_csv(csv_path)
    label_col = "Label" if "Label" in df.columns else ("label" if "label" in df.columns else None)
    if label_col is None: raise ValueError("Could not find Label/label column in DJIA CSV.")
    top_cols = [c for c in df.columns if c.lower().startswith("top")]
    if not top_cols: raise ValueError(f"No Top1..TopN columns found. Got: {list(df.columns)}")
    if mode=="aggregate":
        out = pd.DataFrame({"text": df[top_cols].fillna("").agg(". ".join,axis=1).astype(str).str.strip(),
                            "label": df[label_col].astype(int)})
    elif mode=="per_headline_weak":
        sub = df[top_cols+[label_col]].copy()
        out = sub.melt(id_vars=[label_col], value_vars=top_cols, var_name="top", value_name="text")
        out = out.dropna(subset=["text"])
        out["text"]=out["text"].astype(str).str.strip()
        out["label"]=out[label_col].astype(int)
        out = out[["text","label"]]
    else:
        raise ValueError("djia_mode must be 'aggregate' or 'per_headline_weak'")
    out = out.dropna(); out = out[out["text"].str.len()>0]
    return out, {0:"DOWN", 1:"UP"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["avisheksood","djia"])
    ap.add_argument("--csv", default=None, help="Optional CSV path. If omitted, auto-discovers in ./ and ./data/")
    ap.add_argument("--djia_mode", default="aggregate", choices=["aggregate","per_headline_weak"])
    ap.add_argument("--out", default="../models/impact_model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=160)
    args = ap.parse_args()

    set_seed(args.seed); os.makedirs(args.out, exist_ok=True)
    csv_path = resolve_csv_path(args.dataset, args.csv)

    if args.dataset=="avisheksood":
        df,id2label = load_avisheksood(csv_path)
    else:
        df,id2label = load_djia(csv_path, mode=args.djia_mode)

    label2id = {v:k for k,v in id2label.items()}
    n_classes = len(id2label)

    from sklearn.model_selection import train_test_split
    train_df,val_df = train_test_split(df, test_size=0.1, random_state=args.seed, stratify=df["label"])

    tok_fn = default_tokenizer(max_length=args.max_length)
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tok_fn, batched=True).remove_columns(["text"])
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True)).map(tok_fn,   batched=True).remove_columns(["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", num_labels=n_classes, id2label=id2label, label2id=label2id
    )

    steps_per_epoch = max(1, math.ceil(len(train_ds)/max(1,args.batch)))
    training_args = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        learning_rate=args.lr, weight_decay=0.01,
        evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        logging_steps=max(1, steps_per_epoch//10), logging_dir=os.path.join(args.out,"logs"),
        fp16=torch.cuda.is_available()
    )

    compute_metrics = compute_metrics_binary if n_classes==2 else None
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    print(f"Training on {len(train_ds)} samples; Val: {len(val_ds)}; Classes={n_classes}; Labels={id2label}")
    trainer.train()

    preds = trainer.predict(val_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    with open(os.path.join(args.out,"metrics.json"),"w") as f:
        json.dump({"eval": preds.metrics, "classification_report": report, "labels": id2label}, f, indent=2)
    plot_confusion(cm, os.path.join(args.out,"confusion_matrix.png"),
                   labels=[id2label[i] for i in range(n_classes)])

    trainer.save_model(args.out); tok.save_pretrained(args.out)
    with open(os.path.join(args.out,"label_map.json"),"w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()
