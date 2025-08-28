# QKSVM.py – Chapter 2: Quantum Kernel SVM for GoEmotions (Modified for local dataset)
# -------------------------------------------------------------
# - Deterministic & re-startable (caches BERT + kernel matrices)
# - GPU-ready (kernel matrices live on CPU – SVM is CPU-bound)
# - Uses same BERT encoder as MLC-QML-v3.py
# - Hyper-parameter grid-search on validation set
# - Final test score + per-label F1 + serialised model
# -------------------------------------------------------------
import os, json, pickle, random, time, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn
import pennylane as qml
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from transformers import AutoTokenizer, AutoModel
import sklearn.metrics as skm
from tqdm.auto import tqdm
import hashlib
import ast  # For parsing string representations of lists

# Reproducibility
SEED = 42   # Single seed for faster execution with larger subsample
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# qml.disable_warnings()

# Dataset paths - UPDATE THESE PATHS TO MATCH YOUR DIRECTORY
DATA_DIR = r"C:\Users\Admin\.spyder-py3\QvC-3_docs"  # Update this path
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH   = os.path.join(DATA_DIR, "val.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")

#___________________________________________________________________
#      PROGRESS SHOWCASE FUNCTION 
#__________________________________________________________________

def kernel_matrix_with_progress(X, kernel_fn, Y=None, desc="Kernel"):
    """Wrapper around PennyLane kernel_matrix with tqdm progress bar."""
    if Y is None:
        Y = X
    N, M = len(X), len(Y)
    K = np.zeros((N, M))

    for i in tqdm(range(N), desc=desc):
        for j in range(M):
            if i <= j:  # symmetry if square
                K[i, j] = kernel_fn(X[i], Y[j])
                if X is Y:
                    K[j, i] = K[i, j]  # mirror

    return K

# ------------------------------------------------------------------
# 1. Load GoEmotions from local dataset & BERT encoder
# ------------------------------------------------------------------

# Define emotion labels (28 emotions from GoEmotions)
all_labels = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval','disgust',
    'embarrassment','excitement','fear','gratitude','grief','joy','love',
    'nervousness','optimism','pride','realization','relief','remorse','sadness',
    'surprise','neutral'
]

N_LABELS = len(all_labels)
print(f"Found {N_LABELS} emotion labels: {all_labels[:10]}...")

def parse_labels(label_str):
    """Parse string representation of labels list"""
    if isinstance(label_str, str):
        try:
            return ast.literal_eval(label_str)
        except:
            # Fallback parsing if ast.literal_eval fails
            return eval(label_str)
    return label_str

def labels_to_multi_hot(labels_list):
    """Convert list of binary labels to multi-hot vector"""
    vec = np.array(labels_list, dtype=np.float32)
    return vec

# Load the prepared datasets
print("Loading prepared datasets...")
try:
    df_train = pd.read_csv(TRAIN_PATH)
    df_val   = pd.read_csv(VAL_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    
    # Parse labels from string format
    df_train["labels"] = df_train["labels"].apply(parse_labels)
    df_val["labels"]   = df_val["labels"].apply(parse_labels)
    df_test["labels"]  = df_test["labels"].apply(parse_labels)
    
    # Convert to multi-hot format
    df_train["multi_hot"] = df_train["labels"].apply(labels_to_multi_hot)
    df_val["multi_hot"]   = df_val["labels"].apply(labels_to_multi_hot)
    df_test["multi_hot"]  = df_test["labels"].apply(labels_to_multi_hot)
    
    print("Dataset sizes →", len(df_train), len(df_val), len(df_test))
    
except FileNotFoundError as e:
    print(f"Error: Could not find dataset files. Please run the data preparation pipeline first.")
    print(f"Expected files: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")
    print(f"Error details: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

@torch.no_grad()
def bert_embed(texts, batch_size=512):
    texts = texts.tolist() if isinstance(texts, pd.Series) else texts
    embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="BERT embedding"):
        batch_texts = texts[start:start+batch_size]
        enc = tokenizer(batch_texts,
                        truncation=True, padding="max_length",
                        max_length=100, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        embs.append(bert(**enc).last_hidden_state[:, 0, :].cpu())
    return torch.cat(embs).numpy()

# ------------------------------------------------------------------
# 2. Cache BERT embeddings (once)
# ------------------------------------------------------------------
EMB_PATH = "qksvm_bert_embeddings_local.npz"
if not os.path.exists(EMB_PATH):
    print("Pre-computing BERT embeddings (one-off)...")
    X_train = bert_embed(df_train["text"])
    X_val   = bert_embed(df_val["text"])
    X_test  = bert_embed(df_test["text"])
    
    # Convert multi_hot to numpy arrays
    y_train = np.stack(df_train["multi_hot"].values)
    y_val   = np.stack(df_val["multi_hot"].values)
    y_test  = np.stack(df_test["multi_hot"].values)
    
    print(f"Saving embeddings: X_train {X_train.shape}, y_train {y_train.shape}")
    
    np.savez(EMB_PATH,
             X_train=X_train, X_val=X_val, X_test=X_test,
             y_train=y_train, y_val=y_val, y_test=y_test)
    print(f" Saved embeddings to {EMB_PATH}")
else:
    print(f" Loading cached embeddings from {EMB_PATH}")

data = np.load(EMB_PATH)
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

print(f"Loaded embeddings: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"                   X_val {X_val.shape}, y_val {y_val.shape}")
print(f"                   X_test {X_test.shape}, y_test {y_test.shape}")

# ------------------------------------------------------------------
# 3–7. Single seed: sub-sample + train + evaluate
# ------------------------------------------------------------------
SUBSAMPLE = 20_000
print(f"\n Will sub-sample {SUBSAMPLE} training examples")

class QuantumKernel:
    """Thin wrapper to cache matrices per n_qubits."""
    def __init__(self, n_qubits, cache_tag=""):
        self.cache_tag = cache_tag
        self.n_qubits = n_qubits
        self.cache_dir = f"kernel_cache_{n_qubits}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.reducer = nn.Linear(768, n_qubits)
        self.reducer.weight.data.normal_(0, 0.02)
        self.reducer.bias.data.zero_()
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _embed(self, X):
        with torch.no_grad():
            return self.reducer(torch.tensor(X).float()).numpy()

    def _circuit(self):
        @qml.qnode(self.dev)
        def kernel(x1, x2):
            qml.AngleEmbedding(x1, wires=range(self.n_qubits), rotation="Y")
            qml.templates.layers.StronglyEntanglingLayers(
                np.ones((1, self.n_qubits, 3)), wires=range(self.n_qubits)
            )
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(self.n_qubits), rotation="Y")
            return qml.expval(qml.Projector([0]*self.n_qubits, wires=range(self.n_qubits)))  # |<φ(x1)|φ(x2)>|^2
        return kernel

    def square_matrix(self, X):
        cache = os.path.join(self.cache_dir, f"K_train_{len(X)}_{self.cache_tag}.npy")
        if os.path.exists(cache):
            print(f" Loading cached kernel matrix: {cache}")
            return np.load(cache)
        print(f" Computing kernel matrix: {cache}")
        kernel_fn = self._circuit()
        X_emb = self._embed(X)
        K = kernel_matrix_with_progress(X_emb, kernel_fn, desc=f"Train kernel ({len(X_emb)}x{len(X_emb)})")
        np.save(cache, K)
        return K

    def rectangular_matrix(self, X_new, X_train):
        cache = os.path.join(self.cache_dir, f"K_rect_{len(X_new)}_{len(X_train)}_{self.cache_tag}.npy")
        if os.path.exists(cache):
            print(f" Loading cached kernel matrix: {cache}")
            return np.load(cache)
        print(f" Computing kernel matrix: {cache}")
        kernel_fn = self._circuit()
        X_new_emb = self._embed(X_new)
        X_tr_emb  = self._embed(X_train)
        K = kernel_matrix_with_progress(X_new_emb, kernel_fn, Y=X_tr_emb, desc=f"Test kernel ({len(X_new_emb)}x{len(X_tr_emb)})")
        np.save(cache, K)
        return K

# Results storage
results = {}

# Single seed execution
print(f"\n{'='*60}")
print(f" Running with SEED={SEED}")
print(f"{'='*60}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# Sub-sample (larger subsample for better performance)
sub_idx = rng.choice(len(X_train), size=min(SUBSAMPLE, len(X_train)), replace=False)
SUB_TAG = hashlib.sha1(sub_idx.tobytes()).hexdigest()[:8]
X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
print(f" Sub-sampled {len(X_sub)} examples (tag: {SUB_TAG})")

# ------------------------------------------------------------------
# 5. Grid-search over n_qubits and SVM-C
# ------------------------------------------------------------------
N_QUBITS_CAND = [4, 8, 10, 12, 14]
C_CAND = [0.01, 0.1, 1, 10, 100]

best_macro, best_cfg = -1.0, None
grid_results = []

for n_qubits in N_QUBITS_CAND:
    print(f"\n Testing n_qubits = {n_qubits}")
    qk = QuantumKernel(n_qubits, cache_tag=SUB_TAG)
    K_train = qk.square_matrix(X_sub)
    K_val   = qk.rectangular_matrix(X_val, X_sub)

    # Multi-label SVM with pre-computed kernel
    base = SVC(kernel="precomputed", class_weight="balanced")
    ovr  = OneVsRestClassifier(base)
    param_grid = {"estimator__C": C_CAND}
    
    print(f"    Grid searching over C values: {C_CAND}")
    grid = GridSearchCV(ovr, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=0)
    grid.fit(K_train, y_sub)

    y_pred = grid.predict(K_val)
    macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    
    result = {
        'n_qubits': n_qubits,
        'best_C': grid.best_params_['estimator__C'],
        'val_macro_f1': macro,
        'cv_score': grid.best_score_
    }
    grid_results.append(result)
    
    print(f"    best C={grid.best_params_['estimator__C']:.3f}  →  Val Macro-F1={macro:.4f} (CV: {grid.best_score_:.4f})")
    
    if macro > best_macro:
        best_macro = macro
        best_cfg   = {"n_qubits": n_qubits, "C": grid.best_params_["estimator__C"],
                      "model": grid.best_estimator_, "qk": qk}

# Safety check: ensure best_cfg was set
if best_cfg is None:
    raise RuntimeError(f"No best_cfg found for SEED={SEED} — something went wrong in GridSearch.")

print(f"\n Best validation: {best_cfg['n_qubits']} qubits, C = {best_cfg['C']}, Macro-F1 = {best_macro:.4f}")

# ------------------------------------------------------------------
# 6. Retrain on full sub-sample & evaluate on test
# ------------------------------------------------------------------
print(f"\n Final training and testing...")
qk   = best_cfg["qk"]
K_train = qk.square_matrix(X_sub)
clf = OneVsRestClassifier(SVC(kernel="precomputed", C=best_cfg["C"], class_weight="balanced"))
clf.fit(K_train, y_sub)

K_test = qk.rectangular_matrix(X_test, X_sub)
y_pred = clf.predict(K_test)

# ------------------------------------------------------------------
# 7. Report & save artefacts
# ------------------------------------------------------------------
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
per_label = f1_score(y_test, y_pred, average=None, zero_division=0)

print(f"\n === FINAL RESULTS ===")
print(f"Test Macro-F1: {macro_f1:.4f}")
print(f"Test Micro-F1: {micro_f1:.4f}")
print(f"Val Macro-F1:  {best_macro:.4f}")
print(f"Per-label F1 (top 5): {sorted(per_label, reverse=True)[:5]}")

# Store results
results = {
    'seed': SEED,
    'best_n_qubits': best_cfg['n_qubits'],
    'best_C': best_cfg['C'],
    'val_macro_f1': best_macro,
    'test_macro_f1': macro_f1,
    'test_micro_f1': micro_f1,
    'per_label_f1': per_label.tolist(),
    'grid_search_results': grid_results,
    'dataset_sizes': {
        'train_subsample': len(X_sub),
        'val': len(X_val),
        'test': len(X_test),
        'total_train_available': len(X_train)
    }
}

tag = f"seed{SEED}_{best_cfg['n_qubits']}q_12k"

# Save model
model_path = f"qksvm_final_{tag}.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "model": clf,
        "n_qubits": best_cfg["n_qubits"],
        "C": best_cfg["C"],
        "reducer_w": qk.reducer.weight.detach().cpu().numpy(),
        "reducer_b": qk.reducer.bias.detach().cpu().numpy(),
        "sub_idx": sub_idx,
        "val_macro_f1": best_macro,
        "test_macro_f1": macro_f1,
        "subsample_size": SUBSAMPLE,
        "dataset_info": {
            'train_size': len(df_train),
            'val_size': len(df_val),
            'test_size': len(df_test)
        }
    }, f)

# Save results
results_path = f"qksvm_results_{tag}.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f" Saved {model_path} & {results_path}")

# ------------------------------------------------------------------
# 8. Final summary and comprehensive results
# ------------------------------------------------------------------
print(f"\n{'='*60}")
print(f" EXPERIMENT COMPLETED")
print(f"{'='*60}")

print(f" Final Performance:")
print(f"   Test Macro-F1: {results['test_macro_f1']:.4f}")
print(f"   Test Micro-F1: {results['test_micro_f1']:.4f}")
print(f"   Val Macro-F1:  {results['val_macro_f1']:.4f}")

print(f"\n Best Configuration:")
print(f"   n_qubits: {results['best_n_qubits']}")
print(f"   C: {results['best_C']}")
print(f"   Subsample size: {SUBSAMPLE:,}")

print(f"\n Dataset Information:")
print(f"   Total training samples available: {results['dataset_sizes']['total_train_available']:,}")
print(f"   Training subsample used: {results['dataset_sizes']['train_subsample']:,}")
print(f"   Validation samples: {results['dataset_sizes']['val']:,}")
print(f"   Test samples: {results['dataset_sizes']['test']:,}")

# Save comprehensive results
summary_path = "qksvm_comprehensive_results_12k.json"
summary = {
    'experiment_summary': {
        'seed': SEED,
        'subsample_size': SUBSAMPLE,
        'n_qubits_candidates': N_QUBITS_CAND,
        'C_candidates': C_CAND,
        'dataset_sizes': {
            'train': len(df_train),
            'val': len(df_val),
            'test': len(df_test)
        }
    },
    'final_results': results,
    'performance_summary': {
        'test_macro_f1': results['test_macro_f1'],
        'test_micro_f1': results['test_micro_f1'],
        'val_macro_f1': results['val_macro_f1'],
        'best_n_qubits': results['best_n_qubits'],
        'best_C': results['best_C']
    }
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n Comprehensive results saved to: {summary_path}")
print(f" Experiment completed successfully with {len(df_train):,} training samples available!")
print(f" Final test performance: {results['test_macro_f1']:.4f}")
print(f" Used {SUBSAMPLE:,} training samples for optimal performance/speed trade-off")