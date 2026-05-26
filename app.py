import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='IMDB Dashboard', page_icon='🎬', layout='wide')

st.markdown('''<style>
.stApp { background-color: #0e1117; }
.stSidebar { background-color: #1a1d2e; }
h1, h2, h3 { color: white; }
</style>''', unsafe_allow_html=True)

VIBRANT = ['#FF0000', '#0000FF', '#FFD600', '#FF6F00', '#00C853', '#AA00FF', '#00B0FF', '#FF1744', '#00E676', '#FF9100']
DARK_BG = '#0e1117'
CARD_BG = '#1a1d2e'

plt.rcParams['figure.facecolor'] = DARK_BG
plt.rcParams['axes.facecolor'] = CARD_BG
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams['grid.color'] = '#444444'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['font.family'] = 'DejaVu Sans'

@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\Lenovo\dashboard_project\data\cleaned_movies.csv')
    df['Votes_clean'] = pd.to_numeric(df['Votes'].astype(str).str.replace(',', ''), errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Duration (min)'] = pd.to_numeric(df['Duration (min)'], errors='coerce')
    df['Metascore'] = pd.to_numeric(df['Metascore'], errors='coerce')
    return df

df = load_data()

with st.sidebar:
    st.markdown('<h2 style="color:#E63946;">🎬 IMDB Dashboard</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;">Filter your data below</p>', unsafe_allow_html=True)
    st.markdown('---')
    search = st.text_input('🔍 Search Movie Title', '')
    all_genres = sorted(df['Primary_Genre'].dropna().unique().tolist())
    selected_genres = st.multiselect('🎭 Genre', all_genres, default=[])
    all_certs = sorted(df['Certificate'].dropna().unique().tolist())
    selected_certs = st.multiselect('🎟️ Certificate', all_certs, default=[])
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.slider('📅 Year Range', min_year, max_year, (min_year, max_year))
    rating_range = st.slider('⭐ Rating Range', 0.0, 10.0, (0.0, 10.0), step=0.1)
    st.markdown('---')
    if st.button('🔄 Reset Filters', use_container_width=True):
        st.rerun()
    st.markdown('---')
    st.markdown('<p style="color:#aaa; font-size:12px;">EDA Project<br>Instructor: Ali Hassan Sherazi</p>', unsafe_allow_html=True)

filtered = df.copy()
if search:
    filtered = filtered[filtered['Title'].str.contains(search, case=False, na=False)]
if selected_genres:
    filtered = filtered[filtered['Primary_Genre'].isin(selected_genres)]
if selected_certs:
    filtered = filtered[filtered['Certificate'].isin(selected_certs)]
filtered = filtered[(filtered['Year'] >= year_range[0]) & (filtered['Year'] <= year_range[1])]
filtered = filtered[(filtered['Rating'] >= rating_range[0]) & (filtered['Rating'] <= rating_range[1])]

st.markdown('<h1 style="text-align:center; color:#E63946; font-size:40px;">🎬 IMDB Movies Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#aaaaaa; font-size:15px;">Professional Data Visualization | Exploratory Data Analysis Project</p>', unsafe_allow_html=True)
st.markdown('<hr style="border:1px solid #E63946;">', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric('🎬 Total Movies', f"{len(filtered):,}")
k2.metric('⭐ Avg Rating', f"{filtered['Rating'].mean():.2f}")
k3.metric('🏆 Highest Rated', f"{filtered['Rating'].max():.1f}")
k4.metric('⏱️ Avg Duration', f"{filtered['Duration (min)'].mean():.0f} min")
k5.metric('🗳️ Avg Votes', f"{filtered['Votes_clean'].mean():,.0f}")
st.markdown('---')

st.markdown('<h3 style="color:#00B0FF;">🗃️ Movie Data Table</h3>', unsafe_allow_html=True)
st.markdown(f'<p style="color:#aaa;">Showing {len(filtered):,} movies — use filters or search to narrow down</p>', unsafe_allow_html=True)
table_cols = ['Title', 'Year', 'Rating', 'Primary_Genre', 'Director', 'Duration (min)', 'Certificate', 'Metascore', 'Cast', 'Description']
table_cols = [c for c in table_cols if c in filtered.columns]
st.dataframe(
    filtered[table_cols].reset_index(drop=True),
    use_container_width=True,
    height=350
)
st.markdown('---')

st.markdown('<h3 style="color:#00B0FF;">📊 Genre Overview & Rating Distribution</h3>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    genre_counts = filtered['Primary_Genre'].value_counts().head(8)
    wedges, texts, autotexts = ax.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%',
           colors=VIBRANT[:len(genre_counts)], explode=[0.05]*len(genre_counts),
           startangle=140, textprops={'fontsize': 10, 'fontweight': 'bold', 'color': 'white'})
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
    ax.set_title('Genre Distribution of IMDB Movies', fontsize=14, fontweight='bold', color='white', pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
with col2:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    n, bins, patches = ax.hist(filtered['Rating'].dropna(), bins=25, edgecolor='#0e1117', linewidth=0.8, alpha=0.95)
    for patch, color in zip(patches, plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(patches)))):
        patch.set_facecolor(color)
    mean_r = filtered['Rating'].mean()
    ax.axvline(mean_r, color='#00B0FF', linewidth=2.5, linestyle='--', label=f'Mean: {mean_r:.2f}')
    ax.set_title('Distribution of IMDB Movie Ratings', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Rating', fontsize=11)
    ax.set_ylabel('Number of Movies', fontsize=11)
    ax.legend(fontsize=10, facecolor=CARD_BG, labelcolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
st.markdown('---')

st.markdown('<h3 style="color:#00B0FF;">📈 Trends & Top Directors</h3>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    mpy = filtered['Year'].value_counts().sort_index()
    ax.plot(mpy.index, mpy.values, color='#FF0000', linewidth=2.5, marker='o', markersize=4, markerfacecolor='#FFD600', markeredgecolor='#FF0000')
    ax.fill_between(mpy.index, mpy.values, alpha=0.2, color='#FF0000')
    ax.set_title('Movies Released Per Year', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of Movies', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
with col4:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    top_dir = filtered.groupby('Director')['Rating'].agg(['mean','count']).query('count >= 3').sort_values('mean', ascending=False).head(10)
    bars = ax.barh(top_dir.index, top_dir['mean'], color=VIBRANT[:len(top_dir)], edgecolor='#0e1117', height=0.6)
    for bar, val in zip(bars, top_dir['mean']):
        ax.text(bar.get_width()-0.05, bar.get_y()+bar.get_height()/2, f'{val:.2f}', va='center', ha='right', fontsize=9, fontweight='bold', color='white')
    ax.set_title('Top Directors by Avg Rating', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Average Rating', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
st.markdown('---')

st.markdown('<h3 style="color:#00B0FF;">🔍 Relationships & Distributions</h3>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    sample = filtered.dropna(subset=['Votes_clean','Rating']).sample(n=min(1500,len(filtered)), random_state=42)
    for i, g in enumerate(sample['Primary_Genre'].dropna().unique()[:10]):
        sub = sample[sample['Primary_Genre']==g]
        ax.scatter(sub['Votes_clean'], sub['Rating'], color=VIBRANT[i%len(VIBRANT)], label=g, alpha=0.7, s=35, edgecolors='none')
    ax.set_title('Rating vs Number of Votes', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Number of Votes', fontsize=11)
    ax.set_ylabel('Rating', fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='lower right', facecolor=CARD_BG, labelcolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
with col6:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    top_g = filtered['Primary_Genre'].value_counts().head(8).index.tolist()
    filt_g = filtered[filtered['Primary_Genre'].isin(top_g)]
    bp = ax.boxplot([filt_g[filt_g['Primary_Genre']==g]['Rating'].dropna().values for g in top_g],
                    patch_artist=True, medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5, color='#aaaaaa'),
                    capprops=dict(linewidth=1.5, color='#aaaaaa'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4, markerfacecolor='#aaaaaa'))
    for patch, color in zip(bp['boxes'], VIBRANT):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title('Rating Distribution by Genre', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Genre', fontsize=11)
    ax.set_ylabel('Rating', fontsize=11)
    ax.set_xticklabels(top_g, rotation=30, ha='right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
st.markdown('---')

st.markdown('<h3 style="color:#00B0FF;">🔥 Correlations & Cumulative Trends</h3>', unsafe_allow_html=True)
col7, col8 = st.columns(2)
with col7:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    num_df = filtered[['Rating','Votes_clean','Duration (min)','Metascore','Year']].dropna()
    sns.heatmap(num_df.corr(), ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=2, linecolor=DARK_BG, square=True,
                annot_kws={'size':11,'weight':'bold','color':'white'}, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color='white')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
with col8:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    mpy2 = filtered['Year'].value_counts().sort_index()
    cum = mpy2.cumsum()
    ax.fill_between(mpy2.index, mpy2.values, alpha=0.85, color='#FF0000', label='Per Year')
    ax.fill_between(cum.index, cum.values/cum.max()*mpy2.max(), alpha=0.4, color='#0000FF', label='Cumulative')
    ax.plot(mpy2.index, mpy2.values, color='#FFD600', linewidth=2)
    ax.set_title('Movie Production Over Time', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of Movies', fontsize=11)
    ax.legend(fontsize=10, facecolor=CARD_BG, labelcolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
st.markdown('---')

st.markdown('<h3 style="color:#00B0FF;">🎟️ Certificates & Violin Distributions</h3>', unsafe_allow_html=True)
col9, col10 = st.columns(2)
with col9:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    top_certs = filtered['Certificate'].value_counts().head(10)
    bars = ax.bar(top_certs.index, top_certs.values, color=VIBRANT[:len(top_certs)], edgecolor=DARK_BG, linewidth=0.8, width=0.6)
    for bar, val in zip(bars, top_certs.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
    ax.set_title('Movie Count by Certificate', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Certificate', fontsize=11)
    ax.set_ylabel('Number of Movies', fontsize=11)
    ax.set_xticklabels(top_certs.index, rotation=30, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
with col10:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    top_g8 = filtered['Primary_Genre'].value_counts().head(8).index.tolist()
    filt_v = filtered[filtered['Primary_Genre'].isin(top_g8)]
    parts = ax.violinplot([filt_v[filt_v['Primary_Genre']==g]['Rating'].dropna().values for g in top_g8],
                          positions=range(len(top_g8)), showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(VIBRANT[i%len(VIBRANT)])
        pc.set_alpha(0.85)
        pc.set_edgecolor(DARK_BG)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2.5)
    parts['cbars'].set_color('#aaaaaa')
    parts['cmaxes'].set_color('#aaaaaa')
    parts['cmins'].set_color('#aaaaaa')
    ax.set_title('Rating Density by Genre', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Genre', fontsize=11)
    ax.set_ylabel('Rating', fontsize=11)
    ax.set_xticks(range(len(top_g8)))
    ax.set_xticklabels(top_g8, rotation=30, ha='right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
st.markdown('---')

st.markdown('<h3 style="color:#00B0FF;">🌟 Bonus Charts</h3>', unsafe_allow_html=True)
col11, col12 = st.columns(2)
with col11:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    bubble_df = filtered.groupby('Primary_Genre').agg(Avg_Rating=('Rating','mean'), Avg_Duration=('Duration (min)','mean'), Count=('Title','count')).dropna().sort_values('Count', ascending=False).head(10)
    ax.scatter(bubble_df['Avg_Duration'], bubble_df['Avg_Rating'], s=bubble_df['Count']*0.8, color=VIBRANT[:len(bubble_df)], alpha=0.85, edgecolors='white', linewidth=1.5)
    for i, row in bubble_df.iterrows():
        ax.annotate(str(i), xy=(row['Avg_Duration'], row['Avg_Rating']), xytext=(5,5), textcoords='offset points', fontsize=8, fontweight='bold', color='white')
    ax.set_title('Genre by Duration Rating and Volume', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Avg Duration (min)', fontsize=11)
    ax.set_ylabel('Avg Rating', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
with col12:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    filtered['Rating_Bucket'] = pd.cut(filtered['Rating'], bins=[0,4,5,6,7,8,10], labels=['0-4','4-5','5-6','6-7','7-8','8-10'])
    counts = filtered['Rating_Bucket'].value_counts()
    funnel_order = ['6-7','7-8','5-6','8-10','4-5','0-4']
    funnel_labels = ['6-7 Rating','7-8 Rating','5-6 Rating','8-10 Rating','4-5 Rating','Below 4']
    funnel_vals = [counts.get(k,0) for k in funnel_order]
    max_val = max(funnel_vals) if max(funnel_vals) > 0 else 1
    for i, (label, val) in enumerate(zip(funnel_labels, funnel_vals)):
        left = (max_val - val) / 2
        ax.barh(len(funnel_vals)-i, val, left=left, color=VIBRANT[i%len(VIBRANT)], edgecolor=DARK_BG, linewidth=1.2, height=0.75, alpha=0.92)
        ax.text(max_val/2, len(funnel_vals)-i, f'{label} - {val:,}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.set_title('Movies by Rating Tier', fontsize=14, fontweight='bold', color='white', pad=15)
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown('<h3 style="color:#00B0FF;">🔗 Pair Plot — Feature Relationships</h3>', unsafe_allow_html=True)
pair_df = filtered[['Rating','Votes_clean','Duration (min)','Metascore','Primary_Genre']].dropna()
top5 = pair_df['Primary_Genre'].value_counts().head(5).index
pair_df = pair_df[pair_df['Primary_Genre'].isin(top5)].sample(n=min(800,len(pair_df)), random_state=42)
VIBRANT_5 = ['#FF0000','#0000FF','#FFD600','#00C853','#AA00FF']
with sns.axes_style('dark'):
    g = sns.pairplot(pair_df, hue='Primary_Genre', palette=VIBRANT_5, diag_kind='kde',
                     plot_kws={'alpha':0.6,'s':25}, diag_kws={'alpha':0.7})
    g.figure.patch.set_facecolor(DARK_BG)
    for ax2 in g.axes.flatten():
        if ax2:
            ax2.set_facecolor(CARD_BG)
            ax2.tick_params(colors='white')
            ax2.xaxis.label.set_color('white')
            ax2.yaxis.label.set_color('white')
    g.figure.suptitle('Pair Plot - Relationships Between All Numerical Features', y=1.02, fontsize=14, fontweight='bold', color='white')
    st.pyplot(g.figure)
    plt.close()

st.markdown('---')
st.markdown('<p style="text-align:center; color:#555; font-size:13px;">IMDB Movies Dashboard | EDA Project | Instructor: Ali Hassan Sherazi</p>', unsafe_allow_html=True)