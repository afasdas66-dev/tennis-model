"""
2025 OOS test s plnymi serve stats z Flashscore.
Replikuje presne ten samy pipeline jako tennis_cal.py.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. HISTORICAL PIPELINE (stejny jako tennis_cal.py)
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
print(f"Po merge: {len(df)} zapasu")

# Form ELO (rolling 5)
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

# Serve features from Jeff Sackmann columns
for col in ['w_1stIn','w_svpt','w_1stWon','w_2ndWon','w_bpFaced','w_bpSaved','w_ace','w_df',
            'l_1stIn','l_svpt','l_1stWon','l_2ndWon','l_bpFaced','l_bpSaved','l_ace','l_df']:
    df[col] = pd.to_numeric(df.get(col, 0), errors='coerce')

df['w_1st_pct']    = df['w_1stIn'] / df['w_svpt']
df['l_1st_pct']    = df['l_1stIn'] / df['l_svpt']
df['w_1stWon_pct'] = df['w_1stWon'] / df['w_1stIn']
df['l_1stWon_pct'] = df['l_1stWon'] / df['l_1stIn']
df['w_2ndWon_pct'] = df['w_2ndWon'] / (df['w_svpt'] - df['w_1stIn'])
df['l_2ndWon_pct'] = df['l_2ndWon'] / (df['l_svpt'] - df['l_1stIn'])
df['w_bp_save_pct'] = np.where(df['w_bpFaced'] > 0, df['w_bpSaved']/df['w_bpFaced'], np.nan)
df['l_bp_save_pct'] = np.where(df['l_bpFaced'] > 0, df['l_bpSaved']/df['l_bpFaced'], np.nan)
df['w_ace_rate'] = df['w_ace'] / df['w_svpt']
df['l_ace_rate'] = df['l_ace'] / df['l_svpt']
df['w_df_rate']  = df['w_df']  / df['w_svpt']
df['l_df_rate']  = df['l_df']  / df['l_svpt']

serve_cols_base = ['1st_pct','1stWon_pct','2ndWon_pct','bp_save_pct','ace_rate','df_rate']
player_serve_hist = {}

def get_rolling_serve(player, n=10):
    hist = player_serve_hist.get(player, [])
    if len(hist) < 2:
        return {c: np.nan for c in serve_cols_base}
    return {c: np.nanmean([h[c] for h in hist[-n:]]) for c in serve_cols_base}

roll_w_records, roll_l_records = [], []
for _, row in df.iterrows():
    w, l = row['Winner'], row['Loser']
    roll_w_records.append(get_rolling_serve(w))
    roll_l_records.append(get_rolling_serve(l))
    for player, prefix in [(w, 'w_'), (l, 'l_')]:
        player_serve_hist.setdefault(player, []).append({
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

# Rank + surface
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
# 2. SYMMETRIZE HISTORICAL
# ============================================================
np.random.seed(42)
flip = np.random.rand(len(df)) < 0.5
sym_rows = []
for i, (_, row) in enumerate(df.iterrows()):
    sign  = 1 if flip[i] else -1
    label = 1 if flip[i] else 0
    p1_odds = row['PSW'] if flip[i] else row['PSL']
    base = {
        'Date': row['Date'],
        'label': label,
        'player1_odds': p1_odds,
        'PSW': row['PSW'], 'PSL': row['PSL'],
        'elo_diff': sign * row['elo_diff'],
        'elo_diff_surf': sign * row['elo_diff_surf'],
        'form_elo_diff': sign * row['form_elo_diff'],
        'rank_diff': sign * row['rank_diff'],
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
print(f"Symetrizovany: {len(mdf)}, label_mean={mdf['label'].mean():.3f}")

# ============================================================
# 3. TRAIN + CALIBRATE
# ============================================================
train = mdf[mdf['Date'] < '2023-01-01']
train_c = train.dropna(subset=FEATURES)
X_tr = train_c[FEATURES]
y_tr = train_c['label']

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
prob_train = model_lr = None

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_tr_s, y_tr)
prob_train = model.predict_proba(X_tr_s)[:,1]

# Calibrate on TRAINING data (same as tennis_cal.py)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(prob_train, y_tr)

# Historical test 2023-2024
test_h = mdf[(mdf['Date'] >= '2023-01-01') & (mdf['Date'] < '2025-01-01')].dropna(subset=FEATURES).copy()
X_th = scaler.transform(test_h[FEATURES])
test_h['model_prob'] = model.predict_proba(X_th)[:,1]
test_h['model_prob_cal'] = iso.predict(test_h['model_prob'])
test_h['overround'] = 1/test_h['PSW'] + 1/test_h['PSL']
test_h['pinnacle_prob'] = (1/test_h['player1_odds']) / test_h['overround']
test_h['value_cal'] = test_h['model_prob_cal'] - test_h['pinnacle_prob']

gpt_hist = test_h[(test_h['value_cal'] <= -0.03) &
                  (test_h['player1_odds'] >= 1.5) &
                  (test_h['player1_odds'] < 2.5)]
wins_h = gpt_hist['label'].sum()
profit_h = (gpt_hist['label']*(gpt_hist['player1_odds']-1)).sum() - (len(gpt_hist)-wins_h)
hist_acc = (test_h['model_prob_cal'] > 0.5).mean()
print(f"\nHist 2023-2024: {len(test_h)//2} matches | acc={hist_acc:.3f} | GPT: {len(gpt_hist)} bets | ROI={profit_h/len(gpt_hist):+.1%}")
print(f"Train: {len(train_c)} | Test-hist: {len(test_h)}")

# ============================================================
# 4. 2025 FEATURES
# ============================================================
td25 = pd.read_csv('C:/Users/User/atp_2025_with_serve.csv', encoding='utf-8-sig')
bad25 = [c for c in td25.columns if '\t' in c]
if bad25: td25 = td25.drop(columns=bad25)

td25['Date'] = pd.to_datetime(td25['Date'], dayfirst=True, errors='coerce')
td25 = td25.sort_values('Date').reset_index(drop=True)

# Surface + level
surf_map_td = {'Hard': 'Hard', 'Clay': 'Clay', 'Grass': 'Grass', 'Carpet': 'Carpet'}
td25['Surface'] = td25['Surface'].map(surf_map_td).fillna(td25['Surface'])

def safe_div(a, b):
    try:
        a = float(a); b = float(b)
        return a/b if b > 0 else np.nan
    except: return np.nan

# Build 2025 serve stats — using ACTUAL Flashscore stats
# Compute per-match actual serve percentages for updating rolling hist
def get_pct(row, w_col, tot_col):
    try:
        w = float(row[w_col]); t = float(row[tot_col])
        return w/t if t > 0 else np.nan
    except: return np.nan

# Get rolling serve for 2025 players by continuing from historical
# player_serve_hist is already populated from historical data

feats_25 = []
for _, row in td25.iterrows():
    w = str(row.get('Winner', '')).strip()
    l = str(row.get('Loser', '')).strip()
    date = row['Date']

    rw = get_rolling_serve(w)
    rl = get_rolling_serve(l)

    # ELO from historical (last known)
    elo_w_hist = player_elo_history.get(w, [])
    elo_l_hist = player_elo_history.get(l, [])
    elo_w = elo_w_hist[-1][1] if elo_w_hist else 1500.0
    elo_l = elo_l_hist[-1][1] if elo_l_hist else 1500.0
    elo_d = elo_w - elo_l

    # Form ELO
    fw = np.mean([x[1] for x in elo_w_hist[-5:]]) if len(elo_w_hist) >= 2 else elo_w
    fl = np.mean([x[1] for x in elo_l_hist[-5:]]) if len(elo_l_hist) >= 2 else elo_l
    form_elo_d = fw - fl

    # Rank (flip: winner has lower rank = rank_diff positive = winner advantage)
    wr = float(row['WRank']) if pd.notna(row.get('WRank')) else 300.0
    lr = float(row['LRank']) if pd.notna(row.get('LRank')) else 300.0

    surf = row.get('Surface', 'Hard') or 'Hard'
    elo_d_surf = elo_d  # simplified (no surface-specific ELO in 2025 data)

    feat = {
        'Date': date,
        'PSW': row.get('PSW'), 'PSL': row.get('PSL'),
        'Winner': w, 'Loser': l,
        'Surface': surf,
        'elo_diff': elo_d,
        'elo_diff_surf': elo_d_surf,
        'form_elo_diff': form_elo_d,
        'rank_diff': lr - wr,  # positive = winner advantage
        'roll_diff_1st_pct':     rw['1st_pct'] - rl['1st_pct'],
        'roll_diff_1stWon_pct':  rw['1stWon_pct'] - rl['1stWon_pct'],
        'roll_diff_2ndWon_pct':  rw['2ndWon_pct'] - rl['2ndWon_pct'],
        'roll_diff_bp_save_pct': rw['bp_save_pct'] - rl['bp_save_pct'],
        'roll_diff_ace_rate':    rw['ace_rate'] - rl['ace_rate'],
        'roll_diff_df_rate':     rw['df_rate'] - rl['df_rate'],
    }
    feats_25.append(feat)

    # Update rolling serve from ACTUAL 2025 stats
    def calc_serve_stats(row, prefix):
        # Actual stats from Flashscore (W_ or L_ prefix)
        p = 'W_' if prefix == 'w' else 'L_'
        fst_in = row.get(f'{p}1stin')    # 1st serve %
        fst_won = row.get(f'{p}1stwon')
        fst_tot = row.get(f'{p}1sttot')
        snd_won = row.get(f'{p}2ndwon')
        snd_tot = row.get(f'{p}2ndtot')
        bps = row.get(f'{p}bpsaved')
        bpf = row.get(f'{p}bpfaced')
        aces = row.get(f'{p}aces')
        dfs = row.get(f'{p}df')

        def n(x):
            try: return float(x)
            except: return np.nan

        fst_in_pct  = n(fst_in) / 100.0 if pd.notna(row.get(f'{p}1stin')) else np.nan
        fst_won_pct = n(fst_won)/n(fst_tot) if n(fst_tot) > 0 else np.nan
        snd_won_pct = n(snd_won)/n(snd_tot) if n(snd_tot) > 0 else np.nan
        bp_save     = n(bps)/n(bpf) if n(bpf) > 0 else np.nan

        # ace_rate and df_rate need svpt — approximate from 1sttot+2ndtot
        svpt_approx = n(fst_tot) / fst_in_pct if (fst_in_pct and fst_in_pct > 0) else np.nan
        ace_rate = n(aces)/svpt_approx if svpt_approx and svpt_approx > 0 else np.nan
        df_rate  = n(dfs)/svpt_approx  if svpt_approx and svpt_approx > 0 else np.nan

        return {
            '1st_pct': fst_in_pct, '1stWon_pct': fst_won_pct,
            '2ndWon_pct': snd_won_pct, 'bp_save_pct': bp_save,
            'ace_rate': ace_rate, 'df_rate': df_rate,
        }

    w_stats = calc_serve_stats(row, 'w')
    l_stats = calc_serve_stats(row, 'l')
    player_serve_hist.setdefault(w, []).append(w_stats)
    player_serve_hist.setdefault(l, []).append(l_stats)

df25 = pd.DataFrame(feats_25)
print(f"\n2025 features: {len(df25)} rows")

# Surface dummies for 2025 (same as training)
for sc in surf_cols:
    surf_name = sc.replace('surf_','')
    df25[sc] = (df25['Surface'] == surf_name).astype(float)

# ============================================================
# 5. SYMMETRIZE 2025
# ============================================================
np.random.seed(123)
flip25 = np.random.rand(len(df25)) < 0.5
sym25_rows = []
for i, (_, row) in enumerate(df25.iterrows()):
    sign  = 1 if flip25[i] else -1
    label = 1 if flip25[i] else 0
    p1_odds = row['PSW'] if flip25[i] else row['PSL']
    base = {
        'Date': row['Date'],
        'label': label,
        'player1_odds': p1_odds,
        'PSW': row['PSW'], 'PSL': row['PSL'],
        'Winner': row['Winner'], 'Loser': row['Loser'],
        'Surface': row['Surface'],
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
    sym25_rows.append(base)

sym25 = pd.DataFrame(sym25_rows)

# ============================================================
# 6. PREDICT + EVALUATE 2025
# ============================================================
sym25_odds = sym25[sym25['PSW'].notna() & sym25['PSL'].notna()].copy()
sym25_odds['PSW'] = pd.to_numeric(sym25_odds['PSW'], errors='coerce')
sym25_odds['PSL'] = pd.to_numeric(sym25_odds['PSL'], errors='coerce')
sym25_odds = sym25_odds[sym25_odds['PSW'].notna() & sym25_odds['PSL'].notna()]

X25 = sym25_odds[FEATURES].fillna(0)
sym25_odds['model_prob'] = model.predict_proba(scaler.transform(X25))[:,1]
sym25_odds['model_prob_cal'] = iso.predict(sym25_odds['model_prob'])
sym25_odds['overround'] = 1/sym25_odds['PSW'] + 1/sym25_odds['PSL']
sym25_odds['pinnacle_prob'] = (1/sym25_odds['player1_odds']) / sym25_odds['overround']
sym25_odds['value_cal'] = sym25_odds['model_prob_cal'] - sym25_odds['pinnacle_prob']

print(f"\n2025 with odds: {len(sym25_odds)} symmetrized rows ({len(sym25_odds)//2} matches)")
print(f"Model accuracy: {(sym25_odds['model_prob_cal'] > 0.5).mean():.3f}")
print(f"Actual win rate: {sym25_odds['label'].mean():.3f}")

# Serve feature coverage
serve_feats = ['roll_diff_1st_pct','roll_diff_1stWon_pct','roll_diff_2ndWon_pct']
cov = sym25_odds[serve_feats[0]].notna().sum()
print(f"Serve feature coverage: {cov}/{len(sym25_odds)} ({100*cov/len(sym25_odds):.0f}%)")

# ============================================================
# 7. GPT STRATEGY
# ============================================================
print("\n=== GPT STRATEGY: anti-value <= -0.03, odds 1.5-2.5 ===")
gpt25 = sym25_odds[(sym25_odds['value_cal'] <= -0.03) &
                   (sym25_odds['player1_odds'] >= 1.5) &
                   (sym25_odds['player1_odds'] < 2.5)]
if len(gpt25) > 0:
    wins = gpt25['label'].sum()
    profit = (gpt25['label'] * (gpt25['player1_odds'] - 1)).sum() - (len(gpt25) - wins)
    roi = profit / len(gpt25)
    print(f"Bets: {len(gpt25)}")
    print(f"Wins: {wins} ({wins/len(gpt25):.1%})")
    print(f"Profit: {profit:.2f} units")
    print(f"ROI: {roi:+.1%}")
    print(f"Avg odds: {gpt25['player1_odds'].mean():.2f}")

    # By surface
    print("\nBy surface:")
    for surf, g in gpt25.groupby('Surface'):
        if len(g) < 3: continue
        p = (g['label']*(g['player1_odds']-1)).sum() - (len(g) - g['label'].sum())
        print(f"  {surf:7s}: {len(g):3d} bets | WR={g['label'].mean():.1%} | ROI={p/len(g):+.1%}")
else:
    print("Zadne bety!")

# Threshold sweep
print("\n=== THRESHOLD SWEEP (odds 1.5-2.5) ===")
print(f"{'Thr':>6} | {'Bets':>5} | {'WR':>6} | {'ROI':>8}")
for thr in [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.10]:
    g = sym25_odds[(sym25_odds['value_cal'] <= thr) &
                   (sym25_odds['player1_odds'] >= 1.5) &
                   (sym25_odds['player1_odds'] < 2.5)]
    if len(g) < 3: continue
    p = (g['label']*(g['player1_odds']-1)).sum() - (len(g) - g['label'].sum())
    print(f"{thr:>6.2f} | {len(g):>5d} | {g['label'].mean():>6.1%} | {p/len(g):>+8.1%}")

# Value distribution
print(f"\nValue_cal distribution:")
print(sym25_odds['value_cal'].describe())
