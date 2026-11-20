import plotly.express as px
import streamlit as st
from pyspark.sql import SparkSession

st.set_page_config(
    page_title="Lakehouse Analytics — Transactions Financières",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Tableau de Bord Décisionnel — Data Lakehouse (HDFS + Delta Lake)")
st.markdown(
    "Analyse distribuée de **12,6 millions de transactions bancaires** via l'architecture Médaillon "
    "(`Bronze → Silver → Gold`)."
)


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder.appName("StreamlitGoldReader")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@st.cache_data(ttl=60)
def load_gold_table(hdfs_path: str):
    spark = get_spark()
    return spark.read.format("delta").load(hdfs_path).toPandas()


# Chargement des 4 tables Delta Gold depuis HDFS
try:
    df_pareto = load_gold_table("/lakehouse/gold/pareto_mcc")
    df_time = load_gold_table("/lakehouse/gold/evolution_mensuelle")
    df_states = load_gold_table("/lakehouse/gold/top_mcc_par_etat")
    df_segments = load_gold_table("/lakehouse/gold/segments_clients")
except Exception as e:
    st.error(
        f"Erreur lors de la lecture des tables Gold sur HDFS : {e}. "
        "Vérifiez que tous les notebooks J1 à J4 ont bien été exécutés."
    )
    st.stop()

# ==============================================================================
# BARRE LATÉRALE : FILTRES INTERACTIFS
# ==============================================================================
st.sidebar.header("🎛️ Filtres d'exploration")

top_n_mcc = st.sidebar.slider(
    "Nombre de catégories MCC à afficher (Pareto) :",
    min_value=5,
    max_value=min(50, len(df_pareto)),
    value=15,
)

annees_dispo = sorted(df_time["mois"].str[:4].unique())
selected_years = st.sidebar.multiselect(
    "Filtrer la série temporelle par année :",
    options=annees_dispo,
    default=annees_dispo,
)

segments_dispo = sorted(df_segments["segment"].unique())
selected_segments = st.sidebar.multiselect(
    "Filtrer les segments clients (K-Means) :",
    options=segments_dispo,
    default=segments_dispo,
)

# ==============================================================================
# INDICATEURS CLÉS DE PERFORMANCE (KPI ROBUSTES)
# ==============================================================================
total_tx = int(df_time["nb_transactions"].sum())
total_vol = df_time["volume_mensuel_usd"].sum()
med_globale = df_time["mediane_mensuelle_usd"].median()
taux_err_moy = df_time["taux_erreur_pct"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions Analysées (Silver)", f"{total_tx:,}".replace(",", " "))
col2.metric("Volume Financier Total", f"{total_vol / 1e6:,.2f} M$")
col3.metric("Panier Médian (P50)", f"{med_globale:.2f} $")
col4.metric("Taux d'Incident Moyen", f"{taux_err_moy:.2f} %")

st.divider()

# ==============================================================================
# ANALYSE 1 : LOI DE PARETO PAR CATÉGORIE MARCHAND (MCC)
# ==============================================================================
st.subheader("1. Concentration des Volumes par Catégorie Marchand (Loi de Pareto)")

df_pareto_filtered = df_pareto.head(top_n_mcc).copy()
df_pareto_filtered["mcc_str"] = "MCC " + df_pareto_filtered["mcc"].astype(str)

fig_pareto = px.bar(
    df_pareto_filtered,
    x="mcc_str",
    y="volume_total_usd",
    color="mediane_usd",
    text="pct_cumule_pareto",
    labels={
        "mcc_str": "Code Catégorie Marchand (MCC)",
        "volume_total_usd": "Volume Total ($)",
        "mediane_usd": "Panier Médian ($)",
        "pct_cumule_pareto": "Cumul Pareto (%)",
    },
    title=f"Top {top_n_mcc} Catégories Marchands par Volume et Panier Médian",
    color_continuous_scale="Blues",
)
fig_pareto.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig_pareto, use_container_width=True)

with st.expander("Requête SQL & Justification statistique — Pareto & Asymétrie"):
    st.code(
        """
WITH agg_mcc AS (
    SELECT mcc,
           COUNT(*) AS nb_transactions,
           ROUND(SUM(amount_usd), 2) AS volume_total_usd,
           ROUND(PERCENTILE_APPROX(amount_usd, 0.50), 2) AS mediane_usd,
           ROUND(PERCENTILE_APPROX(amount_usd, 0.75) - PERCENTILE_APPROX(amount_usd, 0.25), 2) AS iqr_usd
    FROM v_silver
    GROUP BY mcc
)
SELECT DENSE_RANK() OVER (ORDER BY volume_total_usd DESC) AS rang_volume,
       mcc, nb_transactions, volume_total_usd, mediane_usd, iqr_usd,
       ROUND((SUM(volume_total_usd) OVER (ORDER BY volume_total_usd DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
              / SUM(volume_total_usd) OVER ()) * 100, 2) AS pct_cumule_pareto
FROM agg_mcc;
        """,
        language="sql",
    )
    st.markdown(
        "**Justification statistique :** La distribution des montants présente une forte asymétrie droite "
        "(`SKEWNESS = 8,1`), où la moyenne (50,60 $) est tirée vers le haut par les valeurs extrêmes. "
        "L'utilisation de la **médiane (`PERCENTILE_APPROX`)** et de l'**IQR** garantit une mesure robuste "
        "du panier typique par secteur marchand."
    )

# ==============================================================================
# ANALYSE 2 : DYNAMIQUE TEMPORELLE & CROISSANCE MENSUELLE (LAG)
# ==============================================================================
st.subheader("2. Évolution Mensuelle des Flux Financiers")

df_time_filtered = df_time[df_time["mois"].str[:4].isin(selected_years)]

fig_time = px.line(
    df_time_filtered,
    x="mois",
    y="volume_mensuel_usd",
    markers=True,
    labels={
        "mois": "Période (Année-Mois)",
        "volume_mensuel_usd": "Volume Mensuel ($)",
    },
    title="Évolution du Volume Mensuel des Transactions",
)
st.plotly_chart(fig_time, use_container_width=True)

with st.expander("Requête SQL & Justification statistique — Série temporelle & Test du Chi-2"):
    st.code(
        """
SELECT mois, nb_transactions, volume_mensuel_usd, mediane_mensuelle_usd, taux_erreur_pct,
       LAG(volume_mensuel_usd, 1) OVER (ORDER BY mois) AS volume_mois_precedent,
       ROUND(((volume_mensuel_usd - LAG(volume_mensuel_usd, 1) OVER (ORDER BY mois))
              / NULLIF(LAG(volume_mensuel_usd, 1) OVER (ORDER BY mois), 0)) * 100, 2) AS croissance_mom_pct
FROM mensuel
ORDER BY mois ASC;
        """,
        language="sql",
    )
    st.markdown(
        "**Justification statistique :** La fonction de fenêtrage `LAG()` met en évidence une saisonnalité "
        "calendaire régulière (creux en février suivi d'un rebond en mars). Par ailleurs, le **test du χ²** "
        "(χ² = 5 421,86, p < 0,0001) a démontré que les paiements en ligne présentent un taux d'erreur "
        "significativement plus élevé (2,29 %) que les paiements physiques (1,51 %)."
    )

# ==============================================================================
# ANALYSE 3 & 4 : SEGMENTATION K-MEANS & TOP GÉOGRAPHIQUE
# ==============================================================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("3. Segmentation Comportementale des Clients (K-Means)")
    df_seg_filtered = df_segments[df_segments["segment"].isin(selected_segments)]

    fig_seg = px.scatter(
        df_seg_filtered,
        x="frequence_tx",
        y="volume_total_usd",
        color="segment",
        size="panier_median_usd",
        hover_data=["client_id", "pct_paiement_physique"],
        log_x=True,
        log_y=True,
        labels={
            "frequence_tx": "Fréquence de Transactions (échelle log)",
            "volume_total_usd": "Volume Total Dépensé $ (échelle log)",
            "segment": "Segment Client",
            "panier_median_usd": "Panier Médian ($)",
        },
        title="Typologie des Porteurs de Cartes (4 Clusters)",
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    with st.expander("Méthodologie & Justification statistique — Clustering K-Means"):
        st.markdown(
            "**Justification statistique :** Avant l'entraînement de l'algorithme **K-Means (k=4)**, "
            "les variables fortement asymétriques (`frequence_tx`, `volume_total_usd`, `panier_median_usd`) "
            "ont subi une transformation logarithmique **`log1p`** suivie d'une standardisation **`StandardScaler`**. "
            "Ce prétraitement évite que les variables à grande échelle monétaire n'écrasent le calcul des distances euclidiennes."
        )

with col_right:
    st.subheader("4. Top 15 Zones / États par Volume Dominant")
    df_states_top = df_states.head(15).copy()
    df_states_top["mcc_label"] = "MCC " + df_states_top["top_mcc"].astype(str)

    fig_states = px.bar(
        df_states_top,
        x="volume_usd",
        y="etat",
        color="mcc_label",
        orientation="h",
        labels={
            "volume_usd": "Volume du MCC Dominant ($)",
            "etat": "État / Canal",
            "mcc_label": "Catégorie Dominante",
        },
        title="Catégorie Marchand N°1 par État (ROW_NUMBER)",
    )
    fig_states.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_states, use_container_width=True)

    with st.expander("Requête SQL & Justification statistique — Partitionnement ROW_NUMBER"):
        st.code(
            """
WITH ranked AS (
    SELECT merchant_state AS etat, mcc, COUNT(*) AS nb_transactions,
           ROUND(SUM(amount_usd), 2) AS volume_usd,
           ROW_NUMBER() OVER (PARTITION BY merchant_state ORDER BY SUM(amount_usd) DESC) AS rn
    FROM v_silver
    GROUP BY merchant_state, mcc
)
SELECT etat, mcc AS top_mcc, nb_transactions, volume_usd
FROM ranked
WHERE rn = 1;
            """,
            language="sql",
        )
        st.markdown(
            "**Justification statistique :** Le partitionnement `ROW_NUMBER() OVER (PARTITION BY etat)` "
            "isole le secteur moteur de chaque zone géographique, révélant la prédominance des transactions "
            "en ligne (`ONLINE`) et des transferts de fonds (`MCC 4829`) dans les grands États."
        )
