import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='IMDB Dashboard', layout='wide')

st.markdown("""
<style>
  /* Global */
  .stApp { background-color: #F5F7FA; }
  .stSidebar { background-color: #1E2235; }
  .stSidebar * { color: #CBD5E0 !important; }
  .stSidebar h2 { color: #FFFFFF !important; font-size:16px !important; font-weight:700 !important; letter-spacing:1.5px; }
  .stSidebar .sidebar-label {
    font-size:10px; font-weight:700; letter-spacing:1.2px;
    color:#A0AEC0 !important; text-transform:uppercase; margin-bottom:2px;
  }
  .stSidebar hr { border-color:#2D3748 !important; }

  /* Hide default metric delta */
  [data-testid="stMetricDelta"] { display:none; }

  /* Section headings */
  h1,h2,h3 { color:#1A202C !important; }

  /* KPI cards */
  .kpi-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 20px 22px 14px 22px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    border-top: 4px solid #3B82F6;
    height: 110px;
  }
  .kpi-card.green  { border-top-color: #10B981; }
  .kpi-card.purple { border-top-color: #8B5CF6; }
  .kpi-card.amber  { border-top-color: #F59E0B; }
  .kpi-card.pink   { border-top-color: #EC4899; }
  .kpi-label {
    font-size:10px; font-weight:700; letter-spacing:1.3px;
    text-transform:uppercase; color:#718096; margin-bottom:6px;
  }
  .kpi-value { font-size:26px; font-weight:800; color:#1A202C; line-height:1.1; }
  .kpi-sub   { font-size:11px; color:#A0AEC0; margin-top:4px; }

  /* Chart card */
  .chart-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 20px 10px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 4px;
  }
  .chart-title { font-size:14px; font-weight:700; color:#1A202C; margin-bottom:3px; }
  .chart-desc  { font-size:12px; color:#718096; margin-bottom:10px; line-height:1.5; }

  /* Download button */
  .stDownloadButton > button {
    background:#EBF4FF; color:#2B6CB0;
    border:1px solid #BEE3F8; border-radius:6px;
    font-size:11px; padding:3px 12px; font-weight:600;
  }
  .stDownloadButton > button:hover { background:#BEE3F8; }

  /* Divider */
  hr { border:none; border-top:1px solid #E2E8F0; margin:20px 0; }

  /* Page header band */
  .page-header {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 22px 28px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .page-header-title { font-size:22px; font-weight:800; color:#1A202C; }
  .page-header-sub   { font-size:13px; color:#718096; margin-top:4px; }
  .badge {
    display:inline-block; background:#EBF4FF; color:#2B6CB0;
    border:1px solid #BEE3F8; border-radius:20px;
    font-size:11px; font-weight:600; padding:3px 12px; margin-left:6px;
  }
</style>
""", unsafe_allow_html=True)

# ── Palette & rcParams ────────────────────────────────────────────────────────
PALETTE   = ['#3B82F6','#EF4444','#10B981','#F59E0B','#8B5CF6',
             '#EC4899','#06B6D4','#84CC16','#F97316','#6366F1']
LIGHT_BG  = '#FFFFFF'
GRID_CLR  = '#E2E8F0'
TEXT_CLR  = '#1A202C'
MUTED_CLR = '#718096'
CHART_W, CHART_H = 7.5, 4.8

plt.rcParams.update({
    'figure.facecolor': LIGHT_BG, 'axes.facecolor': LIGHT_BG,
    'axes.grid': True, 'grid.alpha': 0.5, 'grid.color': GRID_CLR,
    'grid.linestyle': '--', 'text.color': TEXT_CLR,
    'axes.labelcolor': TEXT_CLR, 'xtick.color': MUTED_CLR,
    'ytick.color': MUTED_CLR, 'axes.edgecolor': GRID_CLR,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
    'axes.titleweight': 'bold', 'axes.titlecolor': TEXT_CLR,
    'axes.labelsize': 10,
})

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    buf.seek(0)
    return buf.getvalue()

def chart_header(title, desc):
    st.markdown(
        f'<div class="chart-title">{title}</div>'
        f'<div class="chart-desc">{desc}</div>',
        unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_movies.csv')
    df['Votes_clean'] = pd.to_numeric(
        df['Votes'].astype(str).str.replace(',', ''), errors='coerce')
    for col in ['Year','Rating','Duration (min)','Metascore']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h2>IMDB DASHBOARD</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;color:#718096;">Exploratory Data Analysis</p>',
                unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#2D3748;border-radius:7px;padding:8px 14px;
                margin-bottom:12px;font-size:13px;font-weight:600;color:#FFFFFF !important;">
      Dashboard
    </div>
    <div style="padding:6px 14px;font-size:13px;color:#A0AEC0 !important;">
      Analysis
    </div>
    <div style="padding:6px 14px;font-size:13px;color:#A0AEC0 !important;margin-bottom:10px;">
      Reports
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Filters</div>', unsafe_allow_html=True)
    search = st.text_input('Search Movie Title', '')
    all_genres = sorted(df['Primary_Genre'].dropna().unique().tolist())
    selected_genres = st.multiselect('Primary Genre', all_genres, default=[])
    all_certs = sorted(df['Certificate'].dropna().unique().tolist())
    selected_certs = st.multiselect('Certificate', all_certs, default=[])
    min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
    st.markdown('<div style="font-size:13px;color:#CBD5E0;margin-bottom:4px;">Year Range</div>', unsafe_allow_html=True)
    yc1, yc2 = st.columns(2)
    year_from = yc1.number_input('From', min_value=min_year, max_value=max_year, value=min_year, step=1, key='yr_from', label_visibility='visible')
    year_to   = yc2.number_input('To',   min_value=min_year, max_value=max_year, value=max_year, step=1, key='yr_to',   label_visibility='visible')
    year_range = (int(year_from), int(year_to))

    st.markdown('<div style="font-size:13px;color:#CBD5E0;margin-top:8px;margin-bottom:4px;">Rating Range</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    rating_from = rc1.number_input('Min', min_value=0.0, max_value=10.0, value=0.0, step=0.1, format='%.1f', key='rt_from', label_visibility='visible')
    rating_to   = rc2.number_input('Max', min_value=0.0, max_value=10.0, value=10.0, step=0.1, format='%.1f', key='rt_to',   label_visibility='visible')
    rating_range = (float(rating_from), float(rating_to))
    st.markdown('<hr>', unsafe_allow_html=True)
    if st.button('Reset Filters', use_container_width=True):
        st.rerun()
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:11px;color:#4A5568;">Exploratory Data Analysis<br>'
        'Instructor: Ali Hassan Sherazi</p>',
        unsafe_allow_html=True)

# ── Filtering ─────────────────────────────────────────────────────────────────
filtered = df.copy()
if search:
    filtered = filtered[filtered['Title'].str.contains(search, case=False, na=False)]
if selected_genres:
    filtered = filtered[filtered['Primary_Genre'].isin(selected_genres)]
if selected_certs:
    filtered = filtered[filtered['Certificate'].isin(selected_certs)]
filtered = filtered[
    (filtered['Year'] >= year_range[0]) & (filtered['Year'] <= year_range[1]) &
    (filtered['Rating'] >= rating_range[0]) & (filtered['Rating'] <= rating_range[1])
]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <div>
    <div class="page-header-title">IMDB Movies Analytics Dashboard</div>
    <div class="page-header-sub">
      Exploratory Data Analysis &nbsp;·&nbsp;
      {len(filtered):,} Movies &nbsp;·&nbsp;
      Filtered view updates all charts in real time
    </div>
  </div>
  <div>
    <span class="badge">Python</span>
    <span class="badge">Pandas</span>
    <span class="badge">Matplotlib</span>
    <span class="badge">Seaborn</span>
    <span class="badge">Streamlit</span>
    <span class="badge">{len(df):,} Records</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, 'blue',   'TOTAL MOVIES',   f"{len(filtered):,}",           'in filtered dataset'),
    (k2, 'green',  'AVG RATING',     f"{filtered['Rating'].mean():.2f}", 'average imdb score'),
    (k3, 'purple', 'HIGHEST RATED',  f"{filtered['Rating'].max():.1f}", 'max rating in selection'),
    (k4, 'amber',  'AVG DURATION',   f"{filtered['Duration (min)'].mean():.0f} min", 'average runtime'),
    (k5, 'pink',   'AVG VOTES',      f"{filtered['Votes_clean'].mean():,.0f}", 'average vote count'),
]
for col, color, label, value, sub in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# ── Data Table ────────────────────────────────────────────────────────────────
st.markdown('<h3>Movie Data Table</h3>', unsafe_allow_html=True)
st.markdown(
    f'<p style="color:#718096;font-size:13px;margin-bottom:8px;">'
    f'Showing <b>{len(filtered):,}</b> movies — adjust sidebar filters to narrow results.</p>',
    unsafe_allow_html=True)
table_cols = ['Title','Year','Rating','Primary_Genre','Director',
              'Duration (min)','Certificate','Metascore','Cast','Description']
table_cols = [c for c in table_cols if c in filtered.columns]
st.dataframe(filtered[table_cols].reset_index(drop=True),
             use_container_width=True, height=320)
st.markdown('<hr>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Genre Distribution + Rating Histogram
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h3>Genre Overview & Rating Distribution</h3>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

ROW1_W, ROW1_H = 7.0, 5.2   # fixed equal size for both row-1 charts

with col1:
    chart_header("Genre Distribution",
        "Proportional share of the top 8 genres. "
        "Drama and Action typically dominate IMDB's catalogue.")
    fig, ax = plt.subplots(figsize=(ROW1_W, ROW1_H))
    fig.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    genre_counts = filtered['Primary_Genre'].value_counts().head(8)
    wedges, texts, autotexts = ax.pie(
        genre_counts.values, labels=genre_counts.index,
        autopct='%1.1f%%', colors=PALETTE[:len(genre_counts)],
        explode=[0.04]*len(genre_counts), startangle=140,
        textprops={'fontsize': 9, 'color': TEXT_CLR})
    for at in autotexts: at.set_fontweight('bold'); at.set_fontsize(9)
    ax.set_title('Genre Distribution of IMDB Movies', pad=12)
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'genre_distribution.png', 'image/png', key='dl_genre')
    plt.close()

with col2:
    chart_header("Rating Distribution",
        "Frequency histogram of IMDB ratings. "
        "The blue dashed line marks the mean — most films cluster between 6 and 8.")
    fig, ax = plt.subplots(figsize=(ROW1_W, ROW1_H))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.95)
    n, bins, patches = ax.hist(filtered['Rating'].dropna(), bins=25,
                                edgecolor='white', linewidth=0.6, alpha=0.92)
    cmap = plt.cm.Blues
    for i, patch in enumerate(patches):
        patch.set_facecolor(cmap(0.35 + 0.55 * i / len(patches)))
    mean_r = filtered['Rating'].mean()
    ax.axvline(mean_r, color='#3B82F6', linewidth=2, linestyle='--',
               label=f'Mean: {mean_r:.2f}')
    ax.set_title('Distribution of IMDB Movie Ratings', pad=12)
    ax.set_xlabel('Rating'); ax.set_ylabel('Number of Movies')
    ax.legend(fontsize=9)
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'rating_distribution.png', 'image/png', key='dl_rating')
    plt.close()

st.markdown('<hr>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Movies Per Year + Top Directors
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h3>Trends & Top Directors</h3>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    chart_header("Movies Released Per Year",
        "Annual count of movies in the filtered dataset. "
        "A rising trend from the 1990s reflects IMDB's growing catalogue coverage.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    mpy = filtered['Year'].value_counts().sort_index()
    ax.plot(mpy.index, mpy.values, color='#3B82F6', linewidth=2.2,
            marker='o', markersize=3.5, markerfacecolor='#3B82F6')
    ax.fill_between(mpy.index, mpy.values, alpha=0.12, color='#3B82F6')
    ax.set_title('Movies Released Per Year', pad=12)
    ax.set_xlabel('Year'); ax.set_ylabel('Number of Movies')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'movies_per_year.png', 'image/png', key='dl_mpy')
    plt.close()

with col4:
    chart_header("Top 10 Directors by Average Rating",
        "Directors with at least 3 movies, ranked by mean IMDB rating. "
        "Highlights consistently acclaimed filmmakers in the filtered selection.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    top_dir = (filtered.groupby('Director')['Rating']
               .agg(['mean','count']).query('count >= 3')
               .sort_values('mean', ascending=False).head(10))
    bars = ax.barh(top_dir.index, top_dir['mean'],
                   color=PALETTE[:len(top_dir)], edgecolor='white',
                   height=0.55, alpha=0.90)
    for bar, val in zip(bars, top_dir['mean']):
        ax.text(val - 0.06, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', ha='right',
                fontsize=8.5, fontweight='bold', color='white')
    ax.set_title('Top Directors by Avg Rating', pad=12)
    ax.set_xlabel('Average Rating')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'top_directors.png', 'image/png', key='dl_dirs')
    plt.close()

st.markdown('<hr>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Scatter + Box Plot
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h3>Relationships & Distributions</h3>', unsafe_allow_html=True)
col5, col6 = st.columns(2)

with col5:
    chart_header("Rating vs Number of Votes",
        "Each point is a movie coloured by genre. Highly-voted films tend to "
        "cluster around ratings of 7–8, suggesting popularity correlates with quality.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    sample = (filtered.dropna(subset=['Votes_clean','Rating'])
              .sample(n=min(1500, len(filtered)), random_state=42))
    for i, g in enumerate(sample['Primary_Genre'].dropna().unique()[:10]):
        sub = sample[sample['Primary_Genre'] == g]
        ax.scatter(sub['Votes_clean'], sub['Rating'],
                   color=PALETTE[i % len(PALETTE)], label=g,
                   alpha=0.65, s=22, edgecolors='none')
    ax.set_title('Rating vs Number of Votes', pad=12)
    ax.set_xlabel('Number of Votes'); ax.set_ylabel('Rating')
    ax.legend(fontsize=7.5, ncol=2, loc='lower right')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}K'))
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'votes_vs_rating.png', 'image/png', key='dl_scatter')
    plt.close()

with col6:
    chart_header("Rating Distribution by Genre (Box Plot)",
        "Box plots show the spread and median rating for the top 8 genres. "
        "Wider boxes indicate more variation; dots show extreme outlier ratings.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    top_g = filtered['Primary_Genre'].value_counts().head(8).index.tolist()
    filt_g = filtered[filtered['Primary_Genre'].isin(top_g)]
    bp = ax.boxplot(
        [filt_g[filt_g['Primary_Genre'] == g]['Rating'].dropna().values for g in top_g],
        patch_artist=True,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2, color=MUTED_CLR),
        capprops=dict(linewidth=1.2, color=MUTED_CLR),
        flierprops=dict(marker='o', markersize=2.5, alpha=0.4,
                        markerfacecolor=MUTED_CLR, markeredgecolor='none'))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.82)
    ax.set_xticks(range(1, len(top_g)+1))
    ax.set_xticklabels(top_g, rotation=28, ha='right', fontsize=8.5)
    ax.set_title('Rating Distribution by Genre', pad=12)
    ax.set_xlabel('Genre'); ax.set_ylabel('Rating')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'rating_by_genre_box.png', 'image/png', key='dl_box')
    plt.close()

st.markdown('<hr>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Heatmap + Cumulative Production
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h3>Correlations & Cumulative Trends</h3>', unsafe_allow_html=True)
col7, col8 = st.columns(2)

with col7:
    chart_header("Feature Correlation Heatmap",
        "Pearson correlations between numeric features. Metascore and IMDB Rating "
        "show the strongest positive link; vote count weakly correlates with rating.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    num_df = filtered[['Rating','Votes_clean','Duration (min)','Metascore','Year']].dropna()
    sns.heatmap(num_df.corr(), ax=ax, annot=True, fmt='.2f', cmap='Blues',
                linewidths=1.5, linecolor='#F5F7FA', square=True,
                annot_kws={'size': 9, 'weight': 'bold'}, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap', pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color=TEXT_CLR)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=TEXT_CLR)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'correlation_heatmap.png', 'image/png', key='dl_heat')
    plt.close()

with col8:
    chart_header("Movie Production Over Time",
        "Annual releases (blue) overlaid with a normalised cumulative curve (purple). "
        "The steep climb post-2000 reflects digital-era catalogue expansion.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    mpy2 = filtered['Year'].value_counts().sort_index()
    cum  = mpy2.cumsum()
    ax.fill_between(mpy2.index, mpy2.values, alpha=0.75,
                    color='#3B82F6', label='Per Year')
    ax.fill_between(cum.index, cum.values / cum.max() * mpy2.max(),
                    alpha=0.25, color='#8B5CF6', label='Cumulative (scaled)')
    ax.plot(mpy2.index, mpy2.values, color='#1D4ED8', linewidth=1.8)
    ax.set_title('Movie Production Over Time', pad=12)
    ax.set_xlabel('Year'); ax.set_ylabel('Number of Movies')
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'production_over_time.png', 'image/png', key='dl_cum')
    plt.close()

st.markdown('<hr>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Certificate Bar + Violin
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h3>Certificates & Violin Distributions</h3>', unsafe_allow_html=True)
col9, col10 = st.columns(2)

with col9:
    chart_header("Movie Count by Certificate",
        "Number of films per content certificate. Unrated and R-rated films "
        "are frequently the most represented categories on IMDB.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    top_certs = filtered['Certificate'].value_counts().head(10)
    bars = ax.bar(top_certs.index, top_certs.values,
                  color=PALETTE[:len(top_certs)], edgecolor='white',
                  linewidth=0.6, width=0.6, alpha=0.90)
    for bar, val in zip(bars, top_certs.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + top_certs.max()*0.01,
                f'{val:,}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=TEXT_CLR)
    ax.set_title('Movie Count by Certificate', pad=12)
    ax.set_xlabel('Certificate'); ax.set_ylabel('Number of Movies')
    ax.set_xticklabels(top_certs.index, rotation=28, ha='right')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'certificate_count.png', 'image/png', key='dl_cert')
    plt.close()

with col10:
    chart_header("Rating Density by Genre (Violin Plot)",
        "Violin plots reveal the full distribution shape of ratings per genre. "
        "Thicker sections show where most movies cluster; the line marks the median.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    top_g8 = filtered['Primary_Genre'].value_counts().head(8).index.tolist()
    filt_v = filtered[filtered['Primary_Genre'].isin(top_g8)]
    data_v = [filt_v[filt_v['Primary_Genre'] == g]['Rating'].dropna().values for g in top_g8]
    parts = ax.violinplot(data_v, positions=range(len(top_g8)),
                          showmedians=True, widths=0.65)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.78); pc.set_edgecolor('white')
    parts['cmedians'].set_color(TEXT_CLR); parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color(MUTED_CLR)
    parts['cmaxes'].set_color(MUTED_CLR)
    parts['cmins'].set_color(MUTED_CLR)
    ax.set_xticks(range(len(top_g8)))
    ax.set_xticklabels(top_g8, rotation=28, ha='right', fontsize=8.5)
    ax.set_title('Rating Density by Genre', pad=12)
    ax.set_xlabel('Genre'); ax.set_ylabel('Rating')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'violin_genre_rating.png', 'image/png', key='dl_violin')
    plt.close()

st.markdown('<hr>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6 — Bubble + Funnel
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h3>Bonus Charts</h3>', unsafe_allow_html=True)
col11, col12 = st.columns(2)

with col11:
    chart_header("Genre by Duration, Rating & Volume (Bubble)",
        "Each bubble is a genre; size encodes the number of movies. "
        "Genres in the upper-right have both long runtimes and high ratings.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    bubble_df = (filtered.groupby('Primary_Genre')
                 .agg(Avg_Rating=('Rating','mean'),
                      Avg_Duration=('Duration (min)','mean'),
                      Count=('Title','count'))
                 .dropna().sort_values('Count', ascending=False).head(10))
    ax.scatter(bubble_df['Avg_Duration'], bubble_df['Avg_Rating'],
               s=bubble_df['Count'] * 0.85,
               c=PALETTE[:len(bubble_df)], alpha=0.82,
               edgecolors='white', linewidth=1.2)
    for i, row in bubble_df.iterrows():
        ax.annotate(str(i), xy=(row['Avg_Duration'], row['Avg_Rating']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=7.5, fontweight='bold', color=TEXT_CLR)
    ax.set_title('Genre by Duration, Rating & Volume', pad=12)
    ax.set_xlabel('Avg Duration (min)'); ax.set_ylabel('Avg Rating')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'bubble_genre.png', 'image/png', key='dl_bubble')
    plt.close()

with col12:
    chart_header("Movies by Rating Tier (Funnel)",
        "Funnel chart grouping movies into rating buckets. The 6–7 and 7–8 tiers "
        "are typically the largest, confirming IMDB's above-average rating skew.")
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    ax.set_facecolor(LIGHT_BG)
    filtered['Rating_Bucket'] = pd.cut(
        filtered['Rating'], bins=[0,4,5,6,7,8,10],
        labels=['0-4','4-5','5-6','6-7','7-8','8-10'])
    counts = filtered['Rating_Bucket'].value_counts()
    funnel_order  = ['6-7','7-8','5-6','8-10','4-5','0-4']
    funnel_labels = ['6-7 Rating','7-8 Rating','5-6 Rating',
                     '8-10 Rating','4-5 Rating','Below 4']
    funnel_vals   = [counts.get(k, 0) for k in funnel_order]
    max_val = max(funnel_vals) if max(funnel_vals) > 0 else 1
    for i, (label, val) in enumerate(zip(funnel_labels, funnel_vals)):
        left = (max_val - val) / 2
        ax.barh(len(funnel_vals)-i, val, left=left,
                color=PALETTE[i % len(PALETTE)],
                edgecolor='white', linewidth=0.8, height=0.72, alpha=0.90)
        ax.text(max_val/2, len(funnel_vals)-i,
                f'{label}  —  {val:,}',
                ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white')
    ax.set_title('Movies by Rating Tier', pad=12)
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download Chart', fig_to_bytes(fig),
                       'rating_tier_funnel.png', 'image/png', key='dl_funnel')
    plt.close()

st.markdown('<hr>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Pair Plot (full width)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h3>Pair Plot — Feature Relationships</h3>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#718096;font-size:13px;margin-bottom:10px;">'
    'Pairwise scatter plots for all numeric features, coloured by the top 5 genres. '
    'Diagonal KDE curves show each feature\'s distribution shape.</p>',
    unsafe_allow_html=True)

pair_df = filtered[['Rating','Votes_clean','Duration (min)','Metascore','Primary_Genre']].dropna()
top5 = pair_df['Primary_Genre'].value_counts().head(5).index
pair_df = (pair_df[pair_df['Primary_Genre'].isin(top5)]
           .sample(n=min(800, len(pair_df)), random_state=42))
with sns.axes_style('ticks', {'axes.facecolor': LIGHT_BG,
                               'figure.facecolor': LIGHT_BG,
                               'grid.color': GRID_CLR}):
    g = sns.pairplot(pair_df, hue='Primary_Genre', palette=PALETTE[:5],
                     diag_kind='kde', plot_kws={'alpha': 0.55, 's': 18},
                     diag_kws={'alpha': 0.65})
    g.figure.patch.set_facecolor(LIGHT_BG)
    for ax2 in g.axes.flatten():
        if ax2:
            ax2.set_facecolor(LIGHT_BG)
            ax2.tick_params(colors=MUTED_CLR)
            ax2.xaxis.label.set_color(TEXT_CLR)
            ax2.yaxis.label.set_color(TEXT_CLR)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_color(GRID_CLR)
            ax2.spines['left'].set_color(GRID_CLR)
    g.figure.suptitle('Pair Plot — Relationships Between All Numerical Features',
                      y=1.01, fontsize=13, fontweight='bold', color=TEXT_CLR)

buf = io.BytesIO()
g.figure.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=LIGHT_BG)
buf.seek(0)
st.pyplot(g.figure, use_container_width=True)
st.download_button('Download Pair Plot', buf.getvalue(),
                   'pair_plot.png', 'image/png', key='dl_pair')
plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#A0AEC0;font-size:12px;">'
    'IMDB Movies Dashboard &nbsp;·&nbsp; EDA Project &nbsp;·&nbsp; '
    'Instructor: Ali Hassan Sherazi</p>',
    unsafe_allow_html=True)
