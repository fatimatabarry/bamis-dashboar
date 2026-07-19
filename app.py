import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ==============================================================
# CONFIGURATION
# ==============================================================
st.set_page_config(page_title="Dashboard BAMIS - Défi ESP 2026", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size:30px; font-weight:bold; color:#1E3A8A; margin-bottom:10px; }
    .custom-card { background-color:#F0F4F8; padding:18px; border-radius:10px;
                   border-left:5px solid #1E3A8A; text-align:center; }
    .custom-metric { font-size:26px; font-weight:bold; color:#1E3A8A; }
    .custom-label { font-size:13px; color:#555; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================
# CHARGEMENT DES DONNÉES (les vrais fichiers du notebook complet)
# ==============================================================
@st.cache_data(ttl=3600)
def charger_donnees(dossier):
    volet_a = pd.read_parquet(os.path.join(dossier, "volet_a_score_fraude.parquet"))
    volet_b = pd.read_parquet(os.path.join(dossier, "volet_b_seuils_alertes.parquet"))
    volet_b_client = pd.read_parquet(os.path.join(dossier, "volet_b_alertes_par_client.parquet"))
    volet_c = pd.read_parquet(os.path.join(dossier, "volet_c_scoring_client.parquet"))
    return volet_a, volet_b, volet_b_client, volet_c


st.sidebar.markdown("## 🛡️ Dashboard BAMIS")
st.sidebar.markdown("---")

dossier_data = st.sidebar.text_input(
    "Dossier contenant les fichiers .parquet :",
    value="/content/drive/MyDrive/datathon_bamis",
)

if st.sidebar.button("🔄 Recharger les données"):
    st.cache_data.clear()

try:
    df_a, df_b, df_b_client, df_c = charger_donnees(dossier_data)
except FileNotFoundError as e:
    st.error(
        f"Fichier introuvable : {e}\n\n"
        f"Vérifie que le dossier contient bien : volet_a_score_fraude.parquet, "
        f"volet_b_seuils_alertes.parquet, volet_b_alertes_par_client.parquet, "
        f"volet_c_scoring_client.parquet."
    )
    st.stop()

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Vue d'ensemble", "Volet A — Fraude", "Volet B — Seuils", "Volet C — Clients"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"{len(df_c):,} clients · {len(df_a):,} transactions scorées (Volet A) "
    f"· {len(df_b):,} lignes Volet B"
)


# ==============================================================
# PAGE 1 — VUE D'ENSEMBLE
# ==============================================================
if page == "Vue d'ensemble":
    st.markdown("<div class='main-header'>🛡️ BAMIS — Vue d'ensemble</div>", unsafe_allow_html=True)
    st.caption("Score de fraude = moteur de règles (60%) + Isolation Forest (40%). "
               "Segmentation risque sur seuils asymétriques (80e/95e/99e percentile).")

    nb_clients = len(df_c)
    nb_critiques = (df_c["SEGMENT_RISQUE"] == "Critique").sum()
    nb_transactions = len(df_a)
    nb_suspectes = (df_a["SCORE_FRAUDE_FINAL"] >= 0.5).sum()
    nb_depassements = df_b["DEPASSE_SEUIL_DECLARATIF"].sum() if "DEPASSE_SEUIL_DECLARATIF" in df_b.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    for col, valeur, label, couleur in [
        (col1, f"{nb_clients:,}", "Clients analysés", "#1E3A8A"),
        (col2, f"{nb_critiques:,}", "Clients à risque critique", "#DC2626"),
        (col3, f"{nb_suspectes:,}", "Transactions score ≥ 0.5", "#D97706"),
        (col4, f"{nb_depassements:,}", "Dépassements seuil déclaratif", "#DC2626"),
    ]:
        col.markdown(
            f"<div class='custom-card'><div class='custom-metric' style='color:{couleur};'>"
            f"{valeur}</div><div class='custom-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Matrice de décision (Valeur × Risque)")
        fig = px.histogram(
            df_c, x="SEGMENT_VALEUR", color="SEGMENT_RISQUE", barmode="group",
            category_orders={
                "SEGMENT_VALEUR": ["Bronze", "Argent", "Or", "Platine"],
                "SEGMENT_RISQUE": ["Faible", "Modéré", "Élevé", "Critique"],
            },
            color_discrete_map={"Faible": "#16A34A", "Modéré": "#D97706",
                                 "Élevé": "#EA580C", "Critique": "#DC2626"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Décisions recommandées")
        st.dataframe(df_c["DECISION"].value_counts().reset_index(),
                     use_container_width=True, hide_index=True)


# ==============================================================
# PAGE 2 — VOLET A : FRAUDE
# ==============================================================
elif page == "Volet A — Fraude":
    st.markdown("<div class='main-header'>🕵️ Volet A — Détection de fraude</div>", unsafe_allow_html=True)
    st.caption("SCORE_REGLES (moteur de règles, 5 signaux) combiné à SCORE_ANOMALIE "
               "(Isolation Forest, non supervisé) → SCORE_FRAUDE_FINAL.")

    seuil = st.slider("Filtrer par score de fraude minimum :", 0.0, 1.0, 0.5, 0.05)
    tx_suspectes = df_a[df_a["SCORE_FRAUDE_FINAL"] >= seuil]
    st.info(f"🔍 **{len(tx_suspectes):,}** transactions au-dessus du score {seuil:.2f} "
            f"(sur {len(df_a):,}, soit {100*len(tx_suspectes)/len(df_a):.2f}%).")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig = px.histogram(df_a, x="SCORE_FRAUDE_FINAL", nbins=50,
                            title="Distribution du score final")
        fig.add_vline(x=seuil, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with col_g2:
        fig2 = px.scatter(
            df_a.sample(min(20000, len(df_a)), random_state=42),
            x="SCORE_REGLES", y="SCORE_ANOMALIE", color="SCORE_FRAUDE_FINAL",
            title="Règles vs. Isolation Forest (échantillon)",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 50 des transactions les plus suspectes")
    st.dataframe(tx_suspectes.nlargest(50, "SCORE_FRAUDE_FINAL"),
                 use_container_width=True, hide_index=True)


# ==============================================================
# PAGE 3 — VOLET B : SEUILS
# ==============================================================
elif page == "Volet B — Seuils":
    st.markdown("<div class='main-header'>💰 Volet B — Gestion des seuils / budget</div>", unsafe_allow_html=True)

    ordre = ["NORMAL", "VIGILANCE_50", "ALERTE_80", "CRITIQUE_95", "DEPASSEMENT"]
    couleurs = {"NORMAL": "#16A34A", "VIGILANCE_50": "#84CC16", "ALERTE_80": "#D97706",
                "CRITIQUE_95": "#EA580C", "DEPASSEMENT": "#DC2626"}

    st.subheader("Répartition des paliers d'alerte (cumul journalier)")
    fig = px.pie(df_b, names="PALIER_ALERTE_CUMUL", category_orders={"PALIER_ALERTE_CUMUL": ordre},
                 color="PALIER_ALERTE_CUMUL", color_discrete_map=couleurs)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔄 Contournement potentiel (répartition entre services)")
    st.caption("Transactions qui ne dépassent PAS leur seuil unitaire, mais qui dépassent "
               "le seuil déclaratif global du client.")

    if "DEPASSE_SEUIL_UNITAIRE" in df_b.columns and "DEPASSE_SEUIL_DECLARATIF" in df_b.columns:
        contournement = df_b[(~df_b["DEPASSE_SEUIL_UNITAIRE"]) & (df_b["DEPASSE_SEUIL_DECLARATIF"])]
        st.info(f"**{contournement['SOURCE_PHONE'].nunique():,}** clients concernés, "
                f"**{len(contournement):,}** transactions.")
        st.dataframe(contournement.head(100), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Top clients par ratio de cumul maximal atteint")
    st.dataframe(
        df_b_client.nlargest(20, "ratio_cumul_max"),
        use_container_width=True, hide_index=True,
    )


# ==============================================================
# PAGE 4 — VOLET C : CLIENTS
# ==============================================================
elif page == "Volet C — Clients":
    st.markdown("<div class='main-header'>👤 Volet C — Classement client</div>", unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filtre_risque = st.selectbox("Filtrer par risque :", ["Tous"] + sorted(df_c["SEGMENT_RISQUE"].dropna().unique().tolist()))
    with col_f2:
        filtre_decision = st.selectbox("Filtrer par décision :", ["Toutes"] + sorted(df_c["DECISION"].dropna().unique().tolist()))

    df_filtre = df_c.copy()
    if filtre_risque != "Tous":
        df_filtre = df_filtre[df_filtre["SEGMENT_RISQUE"] == filtre_risque]
    if filtre_decision != "Toutes":
        df_filtre = df_filtre[df_filtre["DECISION"] == filtre_decision]

    colonnes_affichees = [c for c in [
        "SOURCE_PHONE", "SEGMENT_VALEUR", "SEGMENT_RISQUE", "DECISION",
        "MONTANT_TOTAL", "SCORE_VALEUR", "SCORE_RISQUE",
    ] if c in df_filtre.columns]
    st.dataframe(df_filtre[colonnes_affichees], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Fiche client individuelle")
    client_choisi = st.selectbox("Rechercher un numéro de téléphone :", df_filtre["SOURCE_PHONE"].unique())

    if client_choisi:
        info = df_c[df_c["SOURCE_PHONE"] == client_choisi].iloc[0]

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### Profil")
            st.markdown(f"**Segment valeur :** `{info['SEGMENT_VALEUR']}`")
            st.markdown(f"**Segment risque :** `{info['SEGMENT_RISQUE']}`")
            if "MONTANT_TOTAL" in info.index:
                st.markdown(f"**Montant total :** {info['MONTANT_TOTAL']:,.0f} MRU")
        with col_p2:
            st.markdown("#### Décision & explication")
            decision = info["DECISION"]
            if decision in ("INVESTIGATION", "SURVEILLANCE PRIORITAIRE"):
                st.error(f"🛑 {decision}")
            elif decision == "VIGILANCE RENFORCÉE":
                st.warning(f"⚠️ {decision}")
            else:
                st.success(f"✅ {decision}")
            if "TOP_FACTEURS_RISQUE" in info.index:
                st.info(f"**Facteurs de risque principaux :**\n\n{info['TOP_FACTEURS_RISQUE']}")

        st.markdown("---")
        st.subheader("Historique Volet B (seuils)")
        historique = df_b[df_b["SOURCE_PHONE"] == client_choisi].sort_values("TRANSACTION_DATE", ascending=False)
        if not historique.empty:
            st.dataframe(historique.head(50), use_container_width=True, hide_index=True)
        else:
            st.caption("Aucune transaction trouvée pour ce client dans le Volet B.")
