import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- model deps (LOCAL) ---
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Optional deps for embeddings
EMBEDDINGS_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    EMBEDDINGS_AVAILABLE = False

# Fallback similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="CO₂ Planner — Achats (Prototype)",
    page_icon="🌿",
    layout="wide",
)

DATA_PATH = Path("data/produits_co2.csv")

# LOCAL MODEL PATHS
LOCAL_MODEL_DIR = Path("models_local/carbon_qwen")
LOCAL_ENCODER_DIR = LOCAL_MODEL_DIR / "encoder"
LOCAL_TOKENIZER_DIR = LOCAL_MODEL_DIR / "tokenizer"
LOCAL_HEAD_DIR = LOCAL_MODEL_DIR / "head_repo"
LOCAL_HEAD_WEIGHTS = LOCAL_HEAD_DIR / "best_model.pt"  # ✅ IMPORTANT

# Alternative strictness
SIM_THRESHOLD = 0.80
MIN_SHARED_TOKENS = 2

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "with", "without", "in",
    "use", "one", "single-use", "single", "disposable", "reusable",
    "completion", "including", "reprocessing", "unit", "box", "case", "pack",
    "sterile", "medical", "surgical", "type", "high", "capacity",
    "de", "la", "le", "les", "des", "un", "une", "et", "ou", "pour", "avec", "sans",
}

UNIT_CHOICES = [
    "kgCO2e/unité",
    "kgCO2e/boîte",
    "kgCO2e/pack",
    "kgCO2e/carton",
    "kgCO2e/kg",
    "kgCO2e/litre",
    "kgCO2e/€",
    "kgCO2e/k€",
    "À compléter",
]


# -----------------------------
# STYLE
# -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small { opacity: 0.8; font-size: 0.95rem; }
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 16px 16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
}
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.15);
  background: rgba(0,0,0,0.18);
  font-size: 0.85rem;
}
.hr { height: 1px; background: rgba(255,255,255,0.12); margin: 16px 0; }
.rowcard {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.025);
}
.altcard {
  border: 1px solid rgba(46, 204, 113, 0.35);
  border-radius: 16px;
  padding: 12px;
  background: linear-gradient(180deg, rgba(46, 204, 113, 0.14), rgba(46, 204, 113, 0.05));
}
.warncard {
  border: 1px solid rgba(241, 196, 15, 0.35);
  border-radius: 16px;
  padding: 12px;
  background: linear-gradient(180deg, rgba(241, 196, 15, 0.16), rgba(241, 196, 15, 0.06));
}
.totalcard {
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 18px;
  padding: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# HELPERS (text + formatting)
# -----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_meaningful(s: str) -> set[str]:
    s = normalize_text(s)
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    toks = [t for t in s.split() if len(t) >= 3 and t not in STOPWORDS]
    return set(toks)

def shared_tokens_count(a: str, b: str) -> int:
    return len(tokenize_meaningful(a).intersection(tokenize_meaningful(b)))

def format_kgco2(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    x = float(x)
    if x >= 1000:
        return f"{x/1000:.2f} tCO₂e"
    return f"{x:.2f} kgCO₂e"


# -----------------------------
# DATA
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["product_or_process", "functional_unit", "co2", "unit"])

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["product_or_process", "functional_unit", "co2", "unit"]
    for col in required:
        if col not in df.columns:
            if col in ["functional_unit", "unit"]:
                df[col] = ""
            else:
                raise ValueError(f"Le CSV doit contenir la colonne '{col}'")

    df["product_or_process"] = df["product_or_process"].astype(str)
    df["functional_unit"] = df["functional_unit"].astype(str)
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    df["unit"] = df["unit"].astype(str)

    df = df.dropna(subset=["co2"]).reset_index(drop=True)
    return df


# -----------------------------
# EMBEDDINGS (for alternative suggestion)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    if not EMBEDDINGS_AVAILABLE:
        return None
    try:
        return SentenceTransformer("intfloat/e5-small-v2")
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def compute_embeddings(df: pd.DataFrame):
    texts = ("passage: " + df["product_or_process"].astype(str)).tolist()

    model = load_embedder()
    if model is not None:
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return "embeddings", np.array(emb, dtype=np.float32)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(df["product_or_process"].astype(str).tolist())
    return "tfidf", (vec, mat)

def similarity_search(df: pd.DataFrame, query: str, top_k: int = 40):
    query = (query or "").strip()
    if not query:
        return df.head(0)

    method, store = compute_embeddings(df)

    if method == "embeddings":
        model = load_embedder()
        q = model.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False)
        q = np.array(q, dtype=np.float32)
        sims = (store @ q.T).reshape(-1)
        idx = np.argsort(-sims)[:top_k]
        out = df.iloc[idx].copy()
        out["sim"] = sims[idx]
        return out

    vec, mat = store
    qv = vec.transform([query])
    sims = cosine_similarity(mat, qv).reshape(-1)
    idx = np.argsort(-sims)[:top_k]
    out = df.iloc[idx].copy()
    out["sim"] = sims[idx]
    return out


# -----------------------------
# ALTERNATIVE SUGGESTION
# -----------------------------
def normalize_variant(name: str) -> str:
    n = normalize_text(name)
    n = n.replace("single use", "single-use")
    n = n.replace("singleuse", "single-use")
    n = re.sub(r"\s+", " ", n).strip()
    return n

def variant_swap_candidate(df: pd.DataFrame, product_name: str, current_factor: float):
    if df.empty:
        return None

    src = normalize_variant(product_name)
    swaps = [
        ("disposable", "reusable"),
        ("single-use", "reusable"),
        ("virgin", "recycled"),
        ("new", "remanufactured"),
    ]

    def base_form(s: str) -> str:
        s = normalize_variant(s)
        s = re.sub(r"\b(disposable|reusable|single-use|virgin|recycled|new|remanufactured)\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    src_base = base_form(src)
    if not src_base:
        return None

    for a, b in swaps:
        if re.search(rf"\b{re.escape(a)}\b", src):
            target_query = re.sub(rf"\b{re.escape(a)}\b", b, src).strip()

            exact = df[df["product_or_process"].str.lower().str.strip() == target_query.lower().strip()]
            if len(exact):
                row = exact.iloc[0]
                if float(row["co2"]) < float(current_factor):
                    row = row.copy()
                    row["sim"] = 1.0
                    row["shared_tokens"] = shared_tokens_count(product_name, str(row["product_or_process"]))
                    return row

            df2 = df.copy()
            df2["base"] = df2["product_or_process"].apply(base_form)
            df2 = df2[df2["base"] == src_base].copy()
            if df2.empty:
                continue

            df2["name_norm2"] = df2["product_or_process"].str.lower().str.strip()
            df2 = df2[df2["name_norm2"].str.contains(rf"\b{re.escape(b)}\b", regex=True)]
            df2 = df2[df2["co2"] < float(current_factor)].copy()
            if df2.empty:
                continue

            df2 = df2.sort_values("co2", ascending=True)
            best = df2.iloc[0].copy()
            best["sim"] = 0.99
            best["shared_tokens"] = shared_tokens_count(product_name, str(best["product_or_process"]))
            return best

    return None

def suggest_alternative(df: pd.DataFrame, product_name: str, current_factor: float):
    alt = variant_swap_candidate(df, product_name, current_factor)
    if alt is not None:
        return alt

    if df.empty:
        return None

    candidates = similarity_search(df, product_name, top_k=50)
    if candidates.empty:
        return None

    candidates = candidates[candidates["sim"] >= SIM_THRESHOLD].copy()
    if candidates.empty:
        return None

    candidates["shared_tokens"] = candidates["product_or_process"].apply(
        lambda x: shared_tokens_count(product_name, str(x))
    )
    candidates = candidates[candidates["shared_tokens"] >= MIN_SHARED_TOKENS].copy()
    if candidates.empty:
        return None

    candidates = candidates[candidates["co2"] < current_factor].copy()
    if candidates.empty:
        return None

    candidates = candidates.sort_values(["co2", "sim", "shared_tokens"], ascending=[True, False, False])
    return candidates.iloc[0]


# -----------------------------
# CO2 PREDICTOR (LOCAL)
# -----------------------------
class CarbonAttentionModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.post_encoder_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, ids, mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=ids, attention_mask=mask)
            lhs = outputs.last_hidden_state

        lhs = self.post_encoder_norm(lhs)
        attn_out, _ = self.attention(lhs, lhs, lhs)
        lhs = self.attn_norm(lhs + attn_out)
        pooled = lhs.mean(dim=1)
        return self.regressor(pooled)


def _load_head_state_dict(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Poids du head introuvables: {path}\n"
            f"➡️ Assure-toi d'avoir lancé prepare_local_model.py et que best_model.pt existe."
        )

    sd = torch.load(str(path), map_location="cpu")

    # support: si le fichier contient {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # nettoyage des clés potentielles
    cleaned = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("model."):
            kk = kk[len("model."):]
        cleaned[kk] = v

    return cleaned


@st.cache_resource(show_spinner=True)
def load_carbon_predictor_local():
    # checks
    if not LOCAL_ENCODER_DIR.exists():
        raise FileNotFoundError(f"Encoder local introuvable: {LOCAL_ENCODER_DIR} (lance prepare_local_model.py)")
    if not LOCAL_TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Tokenizer local introuvable: {LOCAL_TOKENIZER_DIR} (lance prepare_local_model.py)")
    if not LOCAL_HEAD_WEIGHTS.exists():
        raise FileNotFoundError(f"Poids head introuvables: {LOCAL_HEAD_WEIGHTS} (doit être best_model.pt)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_TOKENIZER_DIR), trust_remote_code=True)
    model = CarbonAttentionModel(str(LOCAL_ENCODER_DIR))

    sd = _load_head_state_dict(LOCAL_HEAD_WEIGHTS)
    model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_co2_for_new_product(product_name: str, description: str, unit_str: str):
    """
    Le modèle prend en entrée un str = concat des champs Streamlit séparés par '-'
    """
    tokenizer, model, device = load_carbon_predictor_local()

    text_in = f"{product_name.strip()}-{(description or '').strip()}-{(unit_str or '').strip()}"

    inputs = tokenizer(
        text_in,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    )

    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        pred = model(ids, mask).squeeze(-1)
        factor = float(pred.item())

    factor = max(0.0, factor)
    confidence = 0.75  # placeholder
    return factor, (unit_str or "kgCO2e/unité"), confidence


def estimate_co2(product_name: str, quantity: float, df: pd.DataFrame, description: str = "", unit_hint: str = "kgCO2e/unité"):
    exact = df[df["product_or_process"] == product_name]
    if len(exact):
        factor = float(exact.iloc[0]["co2"])
        unit = str(exact.iloc[0]["unit"]).strip() or "kgCO2e/unité"
        return factor * quantity, factor, unit, True, 0.95

    factor, unit, conf = predict_co2_for_new_product(product_name, description, unit_hint)
    return factor * quantity, factor, unit, False, conf


# -----------------------------
# SESSION STATE
# -----------------------------
if "basket" not in st.session_state:
    st.session_state.basket = []
if "add_open" not in st.session_state:
    st.session_state.add_open = False


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## 🌿 CHU — Carbon Toolkit")
    st.caption("Prototype — Cas 1 (Achats)")
    page = st.radio(
        "Navigation",
        ["🧾 Liste d’achats", "📊 Insights (placeholder)", "⚙️ Paramètres (placeholder)"],
        index=0,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### 📁 Dataset")
    st.write(f"`{DATA_PATH}`")
    st.markdown(
        f'<div class="small">Alternative: sim ≥ <b>{SIM_THRESHOLD}</b>, overlap ≥ <b>{MIN_SHARED_TOKENS}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### 🧠 Modèle CO₂ (local)")
    st.caption(f"Poids: {LOCAL_HEAD_WEIGHTS}")
    st.caption(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

df = load_data(DATA_PATH)


# -----------------------------
# PAGE: LISTE D’ACHATS
# -----------------------------
if page.startswith("🧾"):
    st.markdown("# 🧾 Liste d’achats — prévision")
    st.markdown(
        '<span class="badge">Ajout par formulaire</span> '
        '<span class="badge">CO₂ par ligne + total</span> '
        '<span class="badge">Modèle local branché</span>',
        unsafe_allow_html=True
    )
    st.write("")

    top_left, top_right = st.columns([1.2, 0.8], gap="large")
    with top_left:
        st.markdown(
            """
<div class="card">
  <div style="font-size:1.05rem; font-weight:800;">Principe</div>
  <div class="small">
    Mode <b>Oui</b> : produit reconnu dans la base → facteur CO₂ du CSV.<br/>
    Mode <b>Non</b> : produit hors base → facteur CO₂ prédit via le modèle local, sur une entrée concaténée avec <b>-</b>.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with top_right:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Ajouter un produit", use_container_width=True):
                st.session_state.add_open = True
        with c2:
            if st.button("🧹 Vider la liste", use_container_width=True):
                st.session_state.basket = []
                st.toast("Liste vidée.")

    if st.session_state.add_open:
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.subheader("Ajouter un produit")

        mode = st.radio(
            "Produit connu ?",
            ["Oui (sélection)", "Non (saisie libre)"],
            horizontal=True,
            key="mode_outside_form"
        )

        st.write("")

        if mode == "Oui (sélection)":
            if len(df) == 0:
                st.warning("Dataset vide : passe en saisie libre.")
            else:
                selected = st.selectbox(
                    "Produit (base existante)",
                    options=df["product_or_process"].tolist(),
                    key="known_selectbox"
                )

                row = df.loc[df["product_or_process"] == selected].iloc[0]
                desc_default = str(row["functional_unit"])
                unit_default = str(row["unit"]).strip() or "kgCO2e/unité"

                with st.form("known_form", clear_on_submit=True):
                    description = st.text_area(
                        "Description / usage (functional unit)",
                        value=desc_default,
                        height=90
                    )
                    unit_user = st.text_input(
                        "Unité du facteur (modifiable)",
                        value=unit_default
                    )
                    quantity = st.number_input("Quantité", min_value=0.0, value=1.0, step=1.0)

                    submitted = st.form_submit_button("Ajouter à la liste ✅", use_container_width=True)
                    if submitted:
                        unit_user = (unit_user or "").strip() or unit_default

                        total_co2, factor, unit_model, known, conf = estimate_co2(
                            selected, float(quantity), df, description, unit_user
                        )
                        unit_final = unit_user if unit_user else unit_model
                        alt = suggest_alternative(df, selected, current_factor=float(factor)) if len(df) else None

                        st.session_state.basket.append({
                            "name": selected,
                            "description": (description or "").strip(),
                            "quantity": float(quantity),
                            "factor": float(factor),
                            "unit": unit_final,
                            "known": bool(known),
                            "confidence": float(conf),
                            "total_co2": float(total_co2),
                            "alt_name": (alt["product_or_process"] if alt is not None else None),
                            "alt_factor": (float(alt["co2"]) if alt is not None else None),
                            "alt_unit": (str(alt["unit"]) if alt is not None else None),
                        })
                        st.session_state.add_open = False
                        st.toast("Produit ajouté ✅")
                        st.rerun()

        else:
            with st.form("free_form", clear_on_submit=True):
                name = st.text_input(
                    "Nom du produit (libellé complet)",
                    placeholder="Ex: Flexible ureteroscope, reusable"
                )
                desc = st.text_area(
                    "Description / usage (functional unit)",
                    placeholder="Ex: Use of one reusable flexible ureteroscope (including reprocessing)",
                    height=90
                )

                unit_choice = st.selectbox("Unité du facteur", UNIT_CHOICES, index=0)
                if unit_choice == "À compléter":
                    unit_user = st.text_input(
                        "Complète l’unité",
                        placeholder="Ex: kgCO2e/€, kgCO2e/unité, kgCO2e/kg…"
                    )
                else:
                    unit_user = unit_choice

                quantity = st.number_input("Quantité", min_value=0.0, value=1.0, step=1.0)

                submitted = st.form_submit_button("Ajouter à la liste ✅", use_container_width=True)
                if submitted:
                    if not name or not name.strip():
                        st.error("Veuillez renseigner un nom de produit.")
                    else:
                        name = name.strip()
                        unit_user = (unit_user or "").strip() or "kgCO2e/unité"

                        total_co2, factor, unit_model, known, conf = estimate_co2(
                            name, float(quantity), df, desc, unit_user
                        )
                        unit_final = unit_user if unit_user else unit_model
                        alt = suggest_alternative(df, name, current_factor=float(factor)) if len(df) else None

                        st.session_state.basket.append({
                            "name": name,
                            "description": (desc or "").strip(),
                            "quantity": float(quantity),
                            "factor": float(factor),
                            "unit": unit_final,
                            "known": bool(known),
                            "confidence": float(conf),
                            "total_co2": float(total_co2),
                            "alt_name": (alt["product_or_process"] if alt is not None else None),
                            "alt_factor": (float(alt["co2"]) if alt is not None else None),
                            "alt_unit": (str(alt["unit"]) if alt is not None else None),
                        })
                        st.session_state.add_open = False
                        st.toast("Produit ajouté ✅")
                        st.rerun()

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if not st.session_state.basket:
        st.markdown(
            """
<div class="warncard">
  <b>Ta liste est vide.</b><br/>
  Clique sur <b>“Ajouter un produit”</b> pour commencer.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.subheader("Produits ajoutés")
        total_all = 0.0

        for i, item in enumerate(st.session_state.basket):
            total_all += float(item["total_co2"])
            left, right = st.columns([1.25, 0.75], gap="large")

            with left:
                badge = "Reconnu" if item["known"] else f"Prédit (conf. {item['confidence']:.2f})"
                st.markdown(
                    f"""
<div class="rowcard">
  <div style="display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div style="font-weight:850; font-size:1.05rem;">{item['name']}</div>
    <div class="badge">{badge}</div>
  </div>
  <div class="small" style="margin-top:6px;"><b>Usage</b> : {item['description'] if item['description'] else "<i>(non renseigné)</i>"}</div>
  <div class="small" style="margin-top:6px;">
    <b>Quantité</b> : {item['quantity']:g}
  </div>
  <div class="small" style="margin-top:8px;">
    <b>Facteur</b> : {item['factor']:.4f} <span class="small">{item['unit']}</span>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(
                    f"""
<div class="rowcard">
  <div class="small">CO₂ estimé</div>
  <div style="font-size:1.7rem; font-weight:900;">{format_kgco2(item['total_co2'])}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                cdel1, cdel2 = st.columns([0.7, 0.3])
                with cdel2:
                    if st.button("🗑️", key=f"del_{i}", help="Supprimer cette ligne"):
                        st.session_state.basket.pop(i)
                        st.rerun()

                if item.get("alt_name"):
                    gain = (item["factor"] - item["alt_factor"]) * item["quantity"]
                    st.markdown(
                        f"""
<div class="altcard">
  <div style="font-weight:850;">Alternative plus sobre</div>
  <div class="small"><b>{item['alt_name']}</b></div>
  <div class="small">Facteur : <b>{item['alt_factor']:.4f}</b> <span class="small">{item['alt_unit']}</span></div>
  <div class="small">Gain estimé (à quantité égale) : <b>{format_kgco2(gain)}</b></div>
</div>
""",
                        unsafe_allow_html=True,
                    )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="totalcard">
  <div class="small">Total CO₂ — liste d’achats</div>
  <div style="font-size:2.1rem; font-weight:950;">{format_kgco2(total_all)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

elif page.startswith("📊"):
    st.markdown("## 📊 Insights (placeholder)")
    st.info("Page réservée : top émetteurs, couverture, export, etc.")
    if len(st.session_state.basket):
        st.dataframe(pd.DataFrame(st.session_state.basket), use_container_width=True)
    else:
        st.write("Ajoute des lignes pour voir des stats.")

else:
    st.markdown("## ⚙️ Paramètres (placeholder)")
    st.info("Ici tu brancheras ton vrai modèle (LLM/ML) pour prédire le facteur CO₂ des produits non présents dans la base.")
    st.write("Dataset chargé :", len(df), "lignes")
    st.write("Embeddings disponibles :", "Oui (SentenceTransformers)" if EMBEDDINGS_AVAILABLE else "Non (fallback TF-IDF)")
    st.write("Torch device :", "cuda" if torch.cuda.is_available() else "cpu")