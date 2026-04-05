"""
2026 OOS test:
- Train on historical 2020-2024 data (atp_clean.csv + atp_elo.csv)
- Update rolling serve history with full 2025 Flashscore data
- Test on 2026 Jan-Apr data (FS serve stats + tennis-data odds)
"""
import sys, io, re, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from unidecode import unidecode
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. HISTORICAL PIPELINE (same as tennis_2025_full_oos.py)
# ============================================================
clean = pd.read_csv("C:/Users/User/Downloads/tennis_data/atp_clean.csv")
elo   = pd.read_csv("C:/Users/User/Downloads/tennis_data/atp_elo.csv")

bad = [c for c in clean.columns if '\t' in c]
if bad: clean = clean.drop(columns=bad)

clean['Date'] = pd.to_datetime(clean['Date'])
elo['Date']   = pd.to_datetime(elo['Date'])

clean['_w'] = clean['Winner'].str.strip().str.lower()
clean['_l'] = clean['Loser'].str.strip().str.lower()
elo['_w']   = elo['winner'].str.strip().str.lower()
elo['_l']   = elo['loser'].str.strip().str.lower()

df = clean.merge(
    elo[['Date','_w','_l','elo_w','elo_l','elo_diff',
         'elo_w_surf','elo_l_surf','elo_diff_surf','exp_w','exp_w_surf']],
    on=['Date','_w','_l'], how='inner'
)
df = df.drop(columns=['_w','_l']).sort_values('Date').reset_index(drop=True)
print(f"Historical: {len(df)} matches ({df['Date'].min().date()} - {df['Date'].max().date()})")

# Form ELO
player_elo_history = {}
form_elo_w, form_elo_l = [], []
for _, row in df.iterrows():
    w, l = row['Winner'], row['Loser']
    hw = player_elo_history.get(w, [])
    hl = player_elo_history.get(l, [])
    form_elo_w.append(np.mean([x[1] for x in hw[-5:]]) if len(hw) >= 2 else row['elo_w'])
    form_elo_l.append(np.mean([x[1] for x in hl[-5:]]) if len(hl) >= 2 else row['elo_l'])
    player_elo_history.setdefault(w, []).append((row['Date'], row['elo_w']))
    player_elo_history.setdefault(l, []).append((row['Date'], row['elo_l']))

df['form_elo_w'] = form_elo_w
df['form_elo_l'] = form_elo_l
df['form_elo_diff'] = df['form_elo_w'] - df['form_elo_l']

# Serve features
for col in ['w_1stIn','w_svpt','w_1stWon','w_2ndWon','w_bpFaced','w_bpSaved','w_ace','w_df',
            'l_1stIn','l_svpt','l_1stWon','l_2ndWon','l_bpFaced','l_bpSaved','l_ace','l_df']:
    df[col] = pd.to_numeric(df.get(col, 0), errors='coerce')

df['w_1st_pct']     = df['w_1stIn'] / df['w_svpt']
df['l_1st_pct']     = df['l_1stIn'] / df['l_svpt']
df['w_1stWon_pct']  = df['w_1stWon'] / df['w_1stIn']
df['l_1stWon_pct']  = df['l_1stWon'] / df['l_1stIn']
df['w_2ndWon_pct']  = df['w_2ndWon'] / (df['w_svpt'] - df['w_1stIn'])
df['l_2ndWon_pct']  = df['l_2ndWon'] / (df['l_svpt'] - df['l_1stIn'])
df['w_bp_save_pct'] = np.where(df['w_bpFaced'] > 0, df['w_bpSaved']/df['w_bpFaced'], np.nan)
df['l_bp_save_pct'] = np.where(df['l_bpFaced'] > 0, df['l_bpSaved']/df['l_bpFaced'], np.nan)
df['w_ace_rate']    = df['w_ace'] / df['w_svpt']
df['l_ace_rate']    = df['l_ace'] / df['l_svpt']
df['w_df_rate']     = df['w_df']  / df['w_svpt']
df['l_df_rate']     = df['l_df']  / df['l_svpt']

serve_cols_base = ['1st_pct','1stWon_pct','2ndWon_pct','bp_save_pct','ace_rate','df_rate']
player_serve_hist = {}

def get_rolling_serve(player, n=10):
    hist = player_serve_hist.get(player, [])
    if len(hist) < 2:
        return {c: np.nan for c in serve_cols_base}
    return {c: np.nanmean([h[c] for h in hist[-n:]]) for c in serve_cols_base}

for _, row in df.iterrows():
    w, l = row['Winner'], row['Loser']
    for player, prefix in [(w, 'w_'), (l, 'l_')]:
        player_serve_hist.setdefault(player, []).append({
            '1st_pct':     row.get(f'{prefix}1st_pct', np.nan),
            '1stWon_pct':  row.get(f'{prefix}1stWon_pct', np.nan),
            '2ndWon_pct':  row.get(f'{prefix}2ndWon_pct', np.nan),
            'bp_save_pct': row.get(f'{prefix}bp_save_pct', np.nan),
            'ace_rate':    row.get(f'{prefix}ace_rate', np.nan),
            'df_rate':     row.get(f'{prefix}df_rate', np.nan),
        })

# Build rolling serve features for historical training set
roll_w_records, roll_l_records = [], []
temp_hist = {}
for _, row in df.iterrows():
    w, l = row['Winner'], row['Loser']
    def _roll(p, n=10):
        h = temp_hist.get(p, [])
        if len(h) < 2: return {c: np.nan for c in serve_cols_base}
        return {c: np.nanmean([x[c] for x in h[-n:]]) for c in serve_cols_base}
    roll_w_records.append(_roll(w))
    roll_l_records.append(_roll(l))
    for player, prefix in [(w, 'w_'), (l, 'l_')]:
        temp_hist.setdefault(player, []).append({
            '1st_pct':     row.get(f'{prefix}1st_pct', np.nan),
            '1stWon_pct':  row.get(f'{prefix}1stWon_pct', np.nan),
            '2ndWon_pct':  row.get(f'{prefix}2ndWon_pct', np.nan),
            'bp_save_pct': row.get(f'{prefix}bp_save_pct', np.nan),
            'ace_rate':    row.get(f'{prefix}ace_rate', np.nan),
            'df_rate':     row.get(f'{prefix}df_rate', np.nan),
        })

roll_w_df = pd.DataFrame(roll_w_records).add_prefix('roll_w_')
roll_l_df = pd.DataFrame(roll_l_records).add_prefix('roll_l_')
df = pd.concat([df.reset_index(drop=True), roll_w_df, roll_l_df], axis=1)
for c in serve_cols_base:
    df[f'roll_diff_{c}'] = df[f'roll_w_{c}'] - df[f'roll_l_{c}']

df['WRank'] = df['WRank'].fillna(500)
df['LRank'] = df['LRank'].fillna(500)
df['rank_diff'] = df['LRank'] - df['WRank']

surface_dummies = pd.get_dummies(df['Surface'], prefix='surf')
df = pd.concat([df, surface_dummies], axis=1)
surf_cols = [c for c in df.columns if c.startswith('surf_')]
print(f"Surface cols: {surf_cols}")

FEATURES = [
    'elo_diff', 'elo_diff_surf', 'form_elo_diff', 'rank_diff',
    'roll_diff_1st_pct', 'roll_diff_1stWon_pct', 'roll_diff_2ndWon_pct',
    'roll_diff_bp_save_pct', 'roll_diff_ace_rate', 'roll_diff_df_rate',
] + surf_cols

# ============================================================
# 2. SYMMETRIZE + TRAIN (pre-2025, same as before)
# ============================================================
np.random.seed(42)
flip = np.random.rand(len(df)) < 0.5
sym_rows = []
for i, (_, row) in enumerate(df.iterrows()):
    sign  = 1 if flip[i] else -1
    label = 1 if flip[i] else 0
    p1_odds = row['PSW'] if flip[i] else row['PSL']
    base = {
        'Date': row['Date'], 'label': label,
        'player1_odds': p1_odds,
        'PSW': row['PSW'], 'PSL': row['PSL'],
        'elo_diff':       sign * row['elo_diff'],
        'elo_diff_surf':  sign * row['elo_diff_surf'],
        'form_elo_diff':  sign * row['form_elo_diff'],
        'rank_diff':      sign * row['rank_diff'],
        'roll_diff_1st_pct':     sign * row.get('roll_diff_1st_pct', np.nan),
        'roll_diff_1stWon_pct':  sign * row.get('roll_diff_1stWon_pct', np.nan),
        'roll_diff_2ndWon_pct':  sign * row.get('roll_diff_2ndWon_pct', np.nan),
        'roll_diff_bp_save_pct': sign * row.get('roll_diff_bp_save_pct', np.nan),
        'roll_diff_ace_rate':    sign * row.get('roll_diff_ace_rate', np.nan),
        'roll_diff_df_rate':     sign * row.get('roll_diff_df_rate', np.nan),
    }
    for sc in surf_cols:
        base[sc] = row.get(sc, 0)
    sym_rows.append(base)

mdf = pd.DataFrame(sym_rows).sort_values('Date').reset_index(drop=True)

# Train on pre-2023 (same as tennis_2025_full_oos.py — unchanged model)
train = mdf[mdf['Date'] < '2023-01-01']
train_c = train.dropna(subset=FEATURES)
X_tr = train_c[FEATURES]
y_tr = train_c['label']

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_tr_s, y_tr)
prob_train = model.predict_proba(X_tr_s)[:,1]

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(prob_train, y_tr)

# Quick sanity: hist 2023-2024
test_h = mdf[(mdf['Date'] >= '2023-01-01') & (mdf['Date'] < '2025-01-01')].dropna(subset=FEATURES).copy()
X_th = scaler.transform(test_h[FEATURES])
test_h['model_prob'] = model.predict_proba(X_th)[:,1]
test_h['model_prob_cal'] = iso.predict(test_h['model_prob'])
test_h['overround'] = 1/test_h['PSW'] + 1/test_h['PSL']
test_h['pinnacle_prob'] = (1/test_h['player1_odds']) / test_h['overround']
test_h['value_cal'] = test_h['model_prob_cal'] - test_h['pinnacle_prob']
gpt_hist = test_h[(test_h['value_cal'] <= -0.03) &
                  (test_h['player1_odds'] >= 1.5) & (test_h['player1_odds'] < 2.5)]
wins_h = gpt_hist['label'].sum()
profit_h = (gpt_hist['label']*(gpt_hist['player1_odds']-1)).sum() - (len(gpt_hist)-wins_h)
print(f"Hist 2023-2024 GPT: {len(gpt_hist)} bets | ROI={profit_h/len(gpt_hist):+.1%}")

# ============================================================
# 3. UPDATE ROLLING SERVE WITH FULL 2025 FS DATA
#    (so 2026 predictions have proper serve history)
# ============================================================
fs25 = pd.read_csv('C:/Users/User/atp_2025_serve_stats.csv')
extra = pd.read_csv('C:/Users/User/atp_2025_2026_extra.csv')
extra25 = extra[extra['season'] == 2025].copy()

# Combine all 2025 FS serve data
fs25['date_dt'] = pd.to_datetime(fs25['date'], errors='coerce')
extra25['date_dt'] = pd.to_datetime(extra25['date'], errors='coerce')

all_25 = pd.concat([
    fs25[['date_dt','player1','player2','winner',
          'p1_1stin_pct','p1_1stwon','p1_1sttotal','p1_2ndwon','p1_2ndtotal',
          'p1_bpsaved','p1_bpfaced','p1_aces','p1_df',
          'p2_1stin_pct','p2_1stwon','p2_1sttotal','p2_2ndwon','p2_2ndtotal',
          'p2_bpsaved','p2_bpfaced','p2_aces','p2_df']],
    extra25[['date_dt','player1','player2','winner',
             'p1_1stin_pct','p1_1stwon','p1_1sttotal','p1_2ndwon','p1_2ndtotal',
             'p1_bpsaved','p1_bpfaced','p1_aces','p1_df',
             'p2_1stin_pct','p2_1stwon','p2_1sttotal','p2_2ndwon','p2_2ndtotal',
             'p2_bpsaved','p2_bpfaced','p2_aces','p2_df']],
], ignore_index=True).sort_values('date_dt')

def fs_row_to_serve(row, idx):
    """Convert FS serve stats to serve_cols_base dict. idx=1 or 2 (player number)."""
    def n(x):
        try: v = float(x); return v if np.isfinite(v) else np.nan
        except: return np.nan

    p = f'p{idx}_'
    fst_in_pct  = n(row.get(f'{p}1stin_pct')) / 100.0 if pd.notna(row.get(f'{p}1stin_pct')) else np.nan
    fst_won     = n(row.get(f'{p}1stwon'))
    fst_tot     = n(row.get(f'{p}1sttotal'))
    snd_won     = n(row.get(f'{p}2ndwon'))
    snd_tot     = n(row.get(f'{p}2ndtotal'))
    bps         = n(row.get(f'{p}bpsaved'))
    bpf         = n(row.get(f'{p}bpfaced'))
    aces        = n(row.get(f'{p}aces'))
    dfs         = n(row.get(f'{p}df'))

    fst_won_pct = fst_won / fst_tot if (fst_tot and fst_tot > 0) else np.nan
    snd_won_pct = snd_won / snd_tot if (snd_tot and snd_tot > 0) else np.nan
    bp_save     = bps / bpf if (bpf and bpf > 0) else np.nan
    # Approximate svpt from 1stIn and 1stIn%
    svpt_approx = fst_tot / fst_in_pct if (fst_in_pct and fst_in_pct > 0) else np.nan
    ace_rate    = aces / svpt_approx if (svpt_approx and svpt_approx > 0) else np.nan
    df_rate     = dfs  / svpt_approx if (svpt_approx and svpt_approx > 0) else np.nan

    return {
        '1st_pct': fst_in_pct, '1stWon_pct': fst_won_pct,
        '2ndWon_pct': snd_won_pct, 'bp_save_pct': bp_save,
        'ace_rate': ace_rate, 'df_rate': df_rate,
    }

updated_25 = 0
for _, row in all_25.iterrows():
    p1 = str(row.get('player1', '')).strip()
    p2 = str(row.get('player2', '')).strip()
    if p1 and p1 != 'nan':
        player_serve_hist.setdefault(p1, []).append(fs_row_to_serve(row, 1))
        updated_25 += 1
    if p2 and p2 != 'nan':
        player_serve_hist.setdefault(p2, []).append(fs_row_to_serve(row, 2))

print(f"\n2025 serve history updated: {updated_25} p1 entries, {len(all_25)} matches")
print(f"Total players with serve history: {len(player_serve_hist)}")

# ============================================================
# 4. LOAD 2026 DATA
# ============================================================

# --- 4a. Tennis-data 2026 odds ---
import io as _io
r2026 = requests.get('http://www.tennis-data.co.uk/2026/2026.xlsx', timeout=20)
td26 = pd.read_excel(_io.BytesIO(r2026.content))
td26['Date'] = pd.to_datetime(td26['Date'], dayfirst=True, errors='coerce')
td26 = td26.sort_values('Date').reset_index(drop=True)
print(f"\n2026 TD odds: {len(td26)} rows, {td26['Tournament'].nunique()} tournaments")
print(f"Date range: {td26['Date'].min().date()} - {td26['Date'].max().date()}")

# --- 4b. FS 2026 serve stats ---
fs26 = extra[extra['season'] == 2026].copy()
fs26['date_dt'] = pd.to_datetime(fs26['date'], errors='coerce')
print(f"2026 FS serve stats: {len(fs26)} rows")
print(f"Tournaments: {fs26['tournament'].value_counts().to_dict()}")

# ============================================================
# 5. MERGE 2026 ODDS + SERVE STATS
# ============================================================

def normalize_name(name):
    if pd.isna(name): return ''
    s = unidecode(str(name)).strip().lower()
    s = re.sub(r'[-.]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    parts = s.split()
    if not parts: return ''
    surname, initials = [], []
    for p in parts:
        if len(p) == 1 or (len(p) == 2 and p[1] not in 'aeiou'):
            initials.append(p[0])
        else:
            surname.append(p)
    return (' '.join(surname) + ' ' + initials[0]) if initials else ' '.join(surname)

def raw_norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[^a-z0-9]', '', str(t).lower())

TOURN_MAP_26 = {
    'brisbaneinternational':          'brisbane',
    'hongkongtennisopen':             'hong kong',
    'adelaideinternational':          'adelaide',
    'asbclassic':                     'auckland',
    'australianopen':                 'australian open',
    'opensuddefrance':                'montpellier',
    'argentinaopen':                  'buenos aires',
    'dallasopen':                     'dallas',
    'abnamroworldtennistournament':   'rotterdam',
    'delraybeachopen':                'delray beach',
    'qatarexxonmobilopen':            'doha',
    'rioopen':                        'rio de janeiro',
    'abiertoexicano':                 'acapulco',
    'abiertomexicano':                'acapulco',
    'dubaitennischampionships':       'dubai',
    'chileopen':                      'santiago',
    'bnpparibasopen':                 'indian wells',
    'miamiopen':                      'miami',
}

def map_tourn_26(t):
    r = raw_norm(t)
    if r in TOURN_MAP_26:
        return TOURN_MAP_26[r]
    for k, v in TOURN_MAP_26.items():
        if k in r or r in k:
            return v
    return r

td26['tourn_mapped'] = td26['Tournament'].apply(map_tourn_26)
td26['winner_norm'] = td26['Winner'].apply(normalize_name)
td26['loser_norm']  = td26['Loser'].apply(normalize_name)

# Normalize FS player names for lookup
fs26['p1_norm'] = fs26['player1'].apply(normalize_name)
fs26['p2_norm'] = fs26['player2'].apply(normalize_name)
fs26['tourn_norm'] = fs26['tournament'].apply(raw_norm)

# Build FS lookup (tourn_norm, date, frozenset(p1_norm, p2_norm))
fs26_lookup = {}
for _, row in fs26.iterrows():
    t_n = raw_norm(row['tournament'])
    d_n = str(row['date'])
    key = (t_n, d_n, frozenset([row['p1_norm'], row['p2_norm']]))
    fs26_lookup[key] = row

# Check which FS tournament names match TD mapped names
td_tourns = td26['tourn_mapped'].unique()
fs_tourns_rn = set(fs26['tournament'].apply(raw_norm).unique())
print("\nTournament match check (raw_norm):")
for tt in sorted(td_tourns):
    tt_rn = raw_norm(tt)
    matched = any(tt_rn == ft_rn or tt_rn in ft_rn or ft_rn in tt_rn for ft_rn in fs_tourns_rn)
    print(f"  {'OK' if matched else 'MISS'} TD:{tt!r} ({tt_rn!r})")

# Merge
stat_cols_w = {
    'p1_aces': 'W_aces', 'p2_aces': 'L_aces',
    'p1_df': 'W_df', 'p2_df': 'L_df',
    'p1_1stin_pct': 'W_1stin', 'p2_1stin_pct': 'L_1stin',
    'p1_1stwon': 'W_1stwon', 'p1_1sttotal': 'W_1sttot',
    'p2_1stwon': 'L_1stwon', 'p2_1sttotal': 'L_1sttot',
    'p1_2ndwon': 'W_2ndwon', 'p1_2ndtotal': 'W_2ndtot',
    'p2_2ndwon': 'L_2ndwon', 'p2_2ndtotal': 'L_2ndtot',
    'p1_bpsaved': 'W_bpsaved', 'p1_bpfaced': 'W_bpfaced',
    'p2_bpsaved': 'L_bpsaved', 'p2_bpfaced': 'L_bpfaced',
}

matched_26 = 0
rows_26 = []

for _, td_row in td26.iterrows():
    tourn = td_row['tourn_mapped']
    date_str = td_row['Date'].strftime('%Y-%m-%d') if pd.notna(td_row['Date']) else ''
    winner_n = td_row['winner_norm']
    loser_n  = td_row['loser_norm']
    pset = frozenset([winner_n, loser_n])

    fs_row = None
    tourn_rn = raw_norm(tourn)  # ensure no-space form matches FS lookup keys
    for delta in [0, 1, -1, 2, -2]:
        try:
            d = (pd.Timestamp(date_str) + pd.Timedelta(days=delta)).strftime('%Y-%m-%d')
            key = (tourn_rn, d, pset)
            if key in fs26_lookup:
                fs_row = fs26_lookup[key]
                break
        except:
            pass

    out_row = td_row.to_dict()
    if fs_row is not None:
        matched_26 += 1
        p1_is_winner = (normalize_name(fs_row['player1']) == winner_n)
        # Map FS columns to TD-style columns
        for fs_col, out_col in stat_cols_w.items():
            val = fs_row.get(fs_col, np.nan)
            if not p1_is_winner:
                # swap p1/p2
                swap = fs_col.replace('p1_', 'TEMP_').replace('p2_', 'p1_').replace('TEMP_', 'p2_')
                out_col_swap = stat_cols_w.get(swap, out_col)
                val = fs_row.get(swap, np.nan)
                out_col = stat_cols_w.get(fs_col.replace('p1_','TEMP_').replace('p2_','p1_').replace('TEMP_','p2_'), out_col)
            out_row[out_col] = val
        out_row['fs_match_id'] = fs_row.get('match_id', np.nan)
    else:
        for v in list(stat_cols_w.values()) + ['fs_match_id']:
            out_row[v] = np.nan
    rows_26.append(out_row)

td26_merged = pd.DataFrame(rows_26)
# Fill missing Pinnacle odds with B365 before feature building
td26_merged['PSW'] = pd.to_numeric(td26_merged['PSW'], errors='coerce')
td26_merged['PSL'] = pd.to_numeric(td26_merged['PSL'], errors='coerce')
td26_merged['B365W'] = pd.to_numeric(td26_merged['B365W'], errors='coerce')
td26_merged['B365L'] = pd.to_numeric(td26_merged['B365L'], errors='coerce')
td26_merged['PSW'] = td26_merged['PSW'].fillna(td26_merged['B365W'])
td26_merged['PSL'] = td26_merged['PSL'].fillna(td26_merged['B365L'])
print(f"\n2026 merge: {matched_26}/{len(td26)} matched ({100*matched_26/len(td26):.1f}%)")
print(f"Rows with odds (PS or B365): {(td26_merged['PSW'].notna() & td26_merged['PSL'].notna()).sum()}")

# ============================================================
# 6. BUILD 2026 FEATURES
# ============================================================

td26_merged['PSW'] = pd.to_numeric(td26_merged['PSW'], errors='coerce')
td26_merged['PSL'] = pd.to_numeric(td26_merged['PSL'], errors='coerce')

feats_26 = []
for _, row in td26_merged.iterrows():
    w = str(row.get('Winner', '')).strip()
    l = str(row.get('Loser', '')).strip()
    date = row['Date']

    rw = get_rolling_serve(w)
    rl = get_rolling_serve(l)

    # ELO from last known historical
    elo_w_hist = player_elo_history.get(w, [])
    elo_l_hist = player_elo_history.get(l, [])
    elo_w = elo_w_hist[-1][1] if elo_w_hist else 1500.0
    elo_l = elo_l_hist[-1][1] if elo_l_hist else 1500.0
    elo_d = elo_w - elo_l

    fw = np.mean([x[1] for x in elo_w_hist[-5:]]) if len(elo_w_hist) >= 2 else elo_w
    fl = np.mean([x[1] for x in elo_l_hist[-5:]]) if len(elo_l_hist) >= 2 else elo_l
    form_elo_d = fw - fl

    wr = float(row['WRank']) if pd.notna(row.get('WRank')) else 300.0
    lr = float(row['LRank']) if pd.notna(row.get('LRank')) else 300.0

    surf_raw = str(row.get('Surface', 'Hard') or 'Hard').strip()
    surf_map = {'Hard': 'Hard', 'Clay': 'Clay', 'Grass': 'Grass', 'Carpet': 'Carpet',
                'hard': 'Hard', 'clay': 'Clay', 'grass': 'Grass'}
    surf = surf_map.get(surf_raw, surf_raw)

    feat = {
        'Date': date,
        'PSW': row.get('PSW'), 'PSL': row.get('PSL'),
        'Winner': w, 'Loser': l,
        'Surface': surf,
        'Tournament': row.get('Tournament', ''),
        'Series': row.get('Series', ''),
        'elo_diff':       elo_d,
        'elo_diff_surf':  elo_d,  # simplified
        'form_elo_diff':  form_elo_d,
        'rank_diff':      lr - wr,
        'roll_diff_1st_pct':     rw['1st_pct'] - rl['1st_pct'],
        'roll_diff_1stWon_pct':  rw['1stWon_pct'] - rl['1stWon_pct'],
        'roll_diff_2ndWon_pct':  rw['2ndWon_pct'] - rl['2ndWon_pct'],
        'roll_diff_bp_save_pct': rw['bp_save_pct'] - rl['bp_save_pct'],
        'roll_diff_ace_rate':    rw['ace_rate'] - rl['ace_rate'],
        'roll_diff_df_rate':     rw['df_rate'] - rl['df_rate'],
        'W_aces': row.get('W_aces'), 'L_aces': row.get('L_aces'),
        'W_1stin': row.get('W_1stin'), 'L_1stin': row.get('L_1stin'),
        'B365W': row.get('B365W'), 'B365L': row.get('B365L'),
    }
    feats_26.append(feat)

    # Update rolling serve with 2026 actual stats
    def n(x):
        try: v = float(x); return v if np.isfinite(v) else np.nan
        except: return np.nan

    for player, pref in [(w, 'W_'), (l, 'L_')]:
        fi = n(row.get(f'{pref}1stin'))
        fw_v = n(row.get(f'{pref}1stwon'))
        ft = n(row.get(f'{pref}1sttot'))
        sw = n(row.get(f'{pref}2ndwon'))
        st = n(row.get(f'{pref}2ndtot'))
        bps = n(row.get(f'{pref}bpsaved'))
        bpf = n(row.get(f'{pref}bpfaced'))
        aces = n(row.get(f'{pref}aces'))
        dfs  = n(row.get(f'{pref}df'))

        fi_pct = fi / 100.0 if pd.notna(row.get(f'{pref}1stin')) else np.nan
        svpt_approx = ft / fi_pct if (fi_pct and fi_pct > 0) else np.nan
        player_serve_hist.setdefault(player, []).append({
            '1st_pct': fi_pct,
            '1stWon_pct': fw_v / ft if (ft and ft > 0) else np.nan,
            '2ndWon_pct': sw / st if (st and st > 0) else np.nan,
            'bp_save_pct': bps / bpf if (bpf and bpf > 0) else np.nan,
            'ace_rate': aces / svpt_approx if (svpt_approx and svpt_approx > 0) else np.nan,
            'df_rate':  dfs  / svpt_approx if (svpt_approx and svpt_approx > 0) else np.nan,
        })

df26 = pd.DataFrame(feats_26)
print(f"2026 feature rows: {len(df26)}")

# Surface dummies
for sc in surf_cols:
    surf_name = sc.replace('surf_', '')
    df26[sc] = (df26['Surface'] == surf_name).astype(float)

# ============================================================
# 7. SYMMETRIZE 2026
# ============================================================
np.random.seed(456)
flip26 = np.random.rand(len(df26)) < 0.5
sym26_rows = []
for i, (_, row) in enumerate(df26.iterrows()):
    sign  = 1 if flip26[i] else -1
    label = 1 if flip26[i] else 0
    p1_odds = row['PSW'] if flip26[i] else row['PSL']
    base = {
        'Date': row['Date'], 'label': label,
        'player1_odds': p1_odds,
        'PSW': row['PSW'], 'PSL': row['PSL'],
        'B365W': row.get('B365W'), 'B365L': row.get('B365L'),
        'Winner': row['Winner'], 'Loser': row['Loser'],
        'Surface': row['Surface'],
        'Tournament': row['Tournament'],
        'elo_diff':       sign * row['elo_diff'],
        'elo_diff_surf':  sign * row['elo_diff_surf'],
        'form_elo_diff':  sign * row['form_elo_diff'],
        'rank_diff':      sign * row['rank_diff'],
        'roll_diff_1st_pct':     sign * row.get('roll_diff_1st_pct', np.nan),
        'roll_diff_1stWon_pct':  sign * row.get('roll_diff_1stWon_pct', np.nan),
        'roll_diff_2ndWon_pct':  sign * row.get('roll_diff_2ndWon_pct', np.nan),
        'roll_diff_bp_save_pct': sign * row.get('roll_diff_bp_save_pct', np.nan),
        'roll_diff_ace_rate':    sign * row.get('roll_diff_ace_rate', np.nan),
        'roll_diff_df_rate':     sign * row.get('roll_diff_df_rate', np.nan),
    }
    for sc in surf_cols:
        base[sc] = row.get(sc, 0)
    sym26_rows.append(base)

sym26 = pd.DataFrame(sym26_rows)

# ============================================================
# 8. PREDICT + EVALUATE 2026
# ============================================================
sym26_odds = sym26.copy()
sym26_odds['PSW'] = pd.to_numeric(sym26_odds['PSW'], errors='coerce')
sym26_odds['PSL'] = pd.to_numeric(sym26_odds['PSL'], errors='coerce')
sym26_odds = sym26_odds[sym26_odds['PSW'].notna() & sym26_odds['PSL'].notna()]

X26 = sym26_odds[FEATURES].fillna(0)
sym26_odds['model_prob'] = model.predict_proba(scaler.transform(X26))[:,1]
sym26_odds['model_prob_cal'] = iso.predict(sym26_odds['model_prob'])
sym26_odds['overround'] = 1/sym26_odds['PSW'] + 1/sym26_odds['PSL']
sym26_odds['pinnacle_prob'] = (1/sym26_odds['player1_odds']) / sym26_odds['overround']
sym26_odds['value_cal'] = sym26_odds['model_prob_cal'] - sym26_odds['pinnacle_prob']

print(f"\n2026 with odds: {len(sym26_odds)} sym rows ({len(sym26_odds)//2} matches)")
print(f"Model accuracy: {(sym26_odds['model_prob_cal'] > 0.5).mean():.3f}")
print(f"Actual win rate: {sym26_odds['label'].mean():.3f}")

serve_feats = ['roll_diff_1st_pct', 'roll_diff_1stWon_pct', 'roll_diff_2ndWon_pct']
cov = (sym26_odds[serve_feats[0]] != 0).sum()
print(f"Serve feature coverage (non-zero): {cov}/{len(sym26_odds)} ({100*cov/len(sym26_odds):.0f}%)")

# ============================================================
# 9. GPT STRATEGY
# ============================================================
print("\n" + "="*60)
print("GPT STRATEGY: anti-value <= -0.03, odds 1.5-2.5")
print("="*60)
gpt26 = sym26_odds[(sym26_odds['value_cal'] <= -0.03) &
                   (sym26_odds['player1_odds'] >= 1.5) &
                   (sym26_odds['player1_odds'] < 2.5)]
if len(gpt26) > 0:
    wins = gpt26['label'].sum()
    profit = (gpt26['label']*(gpt26['player1_odds']-1)).sum() - (len(gpt26)-wins)
    roi = profit / len(gpt26)
    print(f"Bets:    {len(gpt26)}")
    print(f"Wins:    {wins} ({wins/len(gpt26):.1%})")
    print(f"Profit:  {profit:.2f} units")
    print(f"ROI:     {roi:+.1%}")
    print(f"Avg odds:{gpt26['player1_odds'].mean():.2f}")

    print("\nBy surface:")
    for surf, g in gpt26.groupby('Surface'):
        if len(g) < 3: continue
        p = (g['label']*(g['player1_odds']-1)).sum() - (len(g) - g['label'].sum())
        print(f"  {surf:7s}: {len(g):3d} bets | WR={g['label'].mean():.1%} | ROI={p/len(g):+.1%}")

    print("\nBy tournament:")
    for tourn, g in gpt26.groupby('Tournament'):
        if len(g) < 2: continue
        p = (g['label']*(g['player1_odds']-1)).sum() - (len(g) - g['label'].sum())
        print(f"  {tourn:35s}: {len(g):3d} bets | WR={g['label'].mean():.1%} | ROI={p/len(g):+.1%}")
else:
    print("No bets!")

# Threshold sweep
print("\n=== THRESHOLD SWEEP (odds 1.5-2.5) ===")
print(f"{'Thr':>6} | {'Bets':>5} | {'WR':>6} | {'ROI':>8}")
for thr in [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.10]:
    g = sym26_odds[(sym26_odds['value_cal'] <= thr) &
                   (sym26_odds['player1_odds'] >= 1.5) &
                   (sym26_odds['player1_odds'] < 2.5)]
    if len(g) < 3: continue
    p = (g['label']*(g['player1_odds']-1)).sum() - (len(g) - g['label'].sum())
    print(f"{thr:>6.2f} | {len(g):>5d} | {g['label'].mean():>6.1%} | {p/len(g):>+8.1%}")

# Wider odds range sweep
print("\n=== ODDS RANGE SWEEP (threshold=-0.03) ===")
for lo, hi in [(1.3,2.0),(1.5,2.5),(1.3,3.0),(1.5,3.5),(2.0,4.0)]:
    g = sym26_odds[(sym26_odds['value_cal'] <= -0.03) &
                   (sym26_odds['player1_odds'] >= lo) &
                   (sym26_odds['player1_odds'] < hi)]
    if len(g) < 3: continue
    p = (g['label']*(g['player1_odds']-1)).sum() - (len(g) - g['label'].sum())
    print(f"  [{lo:.1f}-{hi:.1f}): {len(g):3d} bets | WR={g['label'].mean():.1%} | ROI={p/len(g):+.1%}")

print(f"\nValue_cal distribution:")
print(sym26_odds['value_cal'].describe())
