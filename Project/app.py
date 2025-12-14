import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FPL AI Manager", page_icon="⚽", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .player-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: black;
    }
    .player-pos { font-size: 10px; color: #888; text-transform: uppercase; font-weight: bold; }
    .player-name { font-size: 14px; font-weight: 800; margin: 4px 0; color: #000; }
    .player-team { font-size: 11px; color: #555; }
    .metric-value { font-size: 16px; font-weight: 800; color: #2e7d32; }
    .metric-label { font-size: 10px; color: #666; }
    
    /* Roles */
    .starter { border-left: 5px solid #00cc44; }
    .captain { border-left: 5px solid #ff2b2b; background-color: #fffafa; }
    .bench { border-left: 5px solid #999; opacity: 0.7; background-color: #f4f4f4; }
    
    .pitch-container {
        background-color: #e8f5e9;
        border: 2px solid #2e7d32;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("models/rf_model.pkl")
        return model
    except FileNotFoundError:
        st.error("⚠️ Model missing! Run the notebook to generate 'models/rf_model.pkl'.")
        return None

# --- 2. DATA PIPELINE (Cleaned - No K-Means) ---
@st.cache_data
def fetch_and_process_data(season="2024-25"):
    base_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/gw{{}}.csv"
    all_gws = []
    
    progress = st.progress(0, text="Fetching Season Data...")
    
    for i in range(1, 39):
        try:
            # Error Handling for Missing/Corrupt Files
            df_gw = pd.read_csv(base_url.format(i))
            
            # Standardization
            req_cols = ['name', 'position', 'team', 'minutes', 'total_points', 'value', 'was_home', 'influence']
            available = [c for c in req_cols if c in df_gw.columns]
            
            if len(available) < len(req_cols):
                continue 
                
            df_gw = df_gw[available]
            df_gw['GW'] = i
            all_gws.append(df_gw)
            
            progress.progress(int(i / 38 * 100))
        except Exception:
            continue
            
    progress.empty()
    
    if not all_gws: return pd.DataFrame()

    full_df = pd.concat(all_gws, ignore_index=True)
    
    # Feature Engineering
    full_df = full_df.sort_values(by=['name', 'GW'])
    grouped = full_df.groupby('name')
    
    full_df['form_last_3'] = grouped['total_points'].transform(lambda x: x.rolling(3).mean().shift(1))
    full_df['avg_minutes_last_3'] = grouped['minutes'].transform(lambda x: x.rolling(3).mean().shift(1))
    full_df['avg_influence_last_3'] = grouped['influence'].transform(lambda x: x.rolling(3).mean().shift(1))
    
    # Clean
    df_clean = full_df.dropna(subset=['form_last_3', 'avg_minutes_last_3']).copy()
    df_clean['avg_influence_last_3'] = df_clean['avg_influence_last_3'].fillna(0)
    
    # Removed K-Means clustering here as requested
        
    return df_clean

# --- 3. OPTIMIZER (Corrected Captain Logic) ---
def solve_fpl_strong_xi(pool, budget=100.0):
    pool = pool.copy().reset_index(drop=True)
    pool['cost'] = pool['value'] / 10.0
    
    # No Minutes = No Points
    pool['xMins_factor'] = (pool['avg_minutes_last_3'] / 90).clip(upper=1.0)
    pool['adjusted_points'] = pool['predicted_points'] * pool['xMins_factor']
    
    starters_pool = pool.sort_values(by='adjusted_points', ascending=False)
    fodder_pool = pool.sort_values(by='cost', ascending=True)
    
    squad = []
    team_counts = {}
    pos_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    current_cost = 0.0
    
    # Phase 1: Bench
    bench_reqs = {'GK': 1, 'DEF': 1, 'MID': 1, 'FWD': 1}
    for _, p in fodder_pool.iterrows():
        pos, team = p['position'], p['team']
        if bench_reqs.get(pos, 0) > 0 and team_counts.get(team, 0) < 3:
            squad.append({**p.to_dict(), 'role': 'Bench', 'is_captain': 0})
            bench_reqs[pos] -= 1
            pos_counts[pos] += 1
            team_counts[team] = team_counts.get(team, 0) + 1
            current_cost += p['cost']
            starters_pool = starters_pool[starters_pool['name'] != p['name']]

    # Phase 2: Starters
    for _, p in starters_pool.iterrows():
        if len(squad) == 15: break
        pos, team, cost = p['position'], p['team'], p['cost']
        limit_map = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        if pos_counts[pos] < limit_map[pos] and team_counts.get(team, 0) < 3:
            if current_cost + cost <= budget:
                squad.append({**p.to_dict(), 'role': 'Starter', 'is_captain': 0})
                pos_counts[pos] += 1
                team_counts[team] = team_counts.get(team, 0) + 1
                current_cost += cost

    # Phase 3: Fallback
    if len(squad) < 15:
        remaining = pool[~pool['name'].isin([x['name'] for x in squad])].sort_values('cost')
        for _, p in remaining.iterrows():
            if len(squad) == 15: break
            pos = p['position']
            limit_map = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            if pos_counts[pos] < limit_map[pos]:
                squad.append({**p.to_dict(), 'role': 'Bench', 'is_captain': 0})
                pos_counts[pos] += 1
                current_cost += p['cost']

    squad_df = pd.DataFrame(squad)
    
    if not squad_df.empty:
        # Assign Captain
        starters = squad_df[squad_df['role'] == 'Starter']
        if not starters.empty:
            cap_idx = starters['adjusted_points'].idxmax()
            squad_df.at[cap_idx, 'is_captain'] = 1
            squad_df.at[cap_idx, 'role'] = 'Captain'
            
            # --- FIXED LOGIC ---
            # 1. Do NOT touch 'adjusted_points' (Predicted). Keep it raw.
            # 2. ONLY double 'total_points' (Actual) for the final score.
            squad_df.at[cap_idx, 'total_points'] *= 2
            
    return squad_df

# --- 4. MAIN APP ---
st.title("⚽ FPL AI Manager")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    target_gw = st.slider("📅 Gameweek", 31, 38, 35)
    budget = st.slider("💰 Budget (£m)", 80.0, 105.0, 100.0, 0.5)
    st.success("System Ready")

# Load
model = load_resources()
df_full = fetch_and_process_data()

if model and not df_full.empty:
    
    # 1. PREDICT
    target_data = df_full[df_full['GW'] == target_gw].copy()
    
    if target_data.empty:
        st.warning(f"⚠️ No data for GW {target_gw}. Please select another week.")
    else:
        features = ['form_last_3', 'avg_minutes_last_3', 'avg_influence_last_3', 'value', 'was_home']
        target_data['predicted_points'] = model.predict(target_data[features])
        
        # Tabs
        tab_team, tab_ml = st.tabs(["🏆 Squad Optimization", "📊 AI Insights"])
        
        # --- TAB 1: SQUAD ---
        with tab_team:
            if st.button("🚀 Generate Squad"):
                squad_df = solve_fpl_strong_xi(target_data, budget)
                
                if not squad_df.empty:
                    # Metrics
                    starters = squad_df[squad_df['role'].isin(['Starter', 'Captain'])]
                    
                    # Totals Calculation
                    xp = starters['adjusted_points'].sum() # Pure raw sum (Captain NOT doubled)
                    real = starters['total_points'].sum()  # Captain ALREADY doubled in solve func
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔮 AI Prediction (Raw)", f"{xp:.1f} pts")
                    c2.metric("📺 Actual Points (Cap x2)", f"{int(real)} pts", delta=f"{int(real - xp)}")
                    c3.metric("💰 Cost", f"£{squad_df['cost'].sum():.1f}m")
                    
                    # Pitch View
                    st.markdown('<div class="pitch-container">', unsafe_allow_html=True)
                    
                    def draw_player(p):
                        role = "captain" if p['is_captain'] else "starter"
                        icon = "©️" if p['is_captain'] else ""
                        
                        # Display Points: 
                        # Predicted: Just show raw. 
                        # Actual: Show the doubled value for captain.
                        
                        pred_display = f"{p['adjusted_points']:.1f}"
                        actual_display = f"{p['total_points']}" # Already doubled for captain
                            
                        return f"""
                        <div class="player-card {role}">
                            <div class="player-pos">{p['position']}</div>
                            <div class="player-name">{p['name']} {icon}</div>
                            <div class="player-team">{p['team']}</div>
                            <div class="metric-value">{pred_display}</div>
                            <div class="metric-label">Actual: {actual_display}</div>
                        </div>"""
                    
                    for pos in ['GK', 'DEF', 'MID', 'FWD']:
                        rows = starters[starters['position'] == pos]
                        cols = st.columns(len(rows) or 1)
                        for i, (_, p) in enumerate(rows.iterrows()):
                            cols[i].markdown(draw_player(p), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 2: ML INSIGHTS (Fixed & Improved) ---
        with tab_ml:
            st.subheader("Model Validation")
            
            # Filter for only the selected squad for cleaner visualization
            if 'squad_df' in locals() and not squad_df.empty:
                viz_df = squad_df[squad_df['role'].isin(['Starter', 'Captain'])].copy()
                
                st.markdown("### 1. Predicted vs. Actual Performance")
                st.markdown("This scatter plot reveals model bias. The green line represents perfect prediction ($y=x$). Points above the line are underperformers (overestimated), and points below are overperformers (underestimated).")
                
                # Visualization A: Predicted vs Actual Scatter
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
                sns.set_theme(style="whitegrid")
                
                # Plot
                sns.scatterplot(
                    data=viz_df, 
                    x='adjusted_points', 
                    y='total_points', # Uses doubled actuals for captain
                    hue='is_captain', 
                    palette={0: 'blue', 1: 'red'},
                    s=150,
                    ax=ax_scatter
                )
                
                # Perfect Prediction Line
                max_val = max(viz_df['adjusted_points'].max(), viz_df['total_points'].max()) + 2
                ax_scatter.plot([0, max_val], [0, max_val], color='green', linestyle='--', label='Perfect Prediction')
                
                ax_scatter.set_title("Accuracy Check: Predicted vs Actual")
                ax_scatter.set_xlabel("Predicted Points (Raw)")
                ax_scatter.set_ylabel("Actual Points (with Cap x2)")
                ax_scatter.legend(title="Captain")
                st.pyplot(fig_scatter)
                
                st.markdown("---")

                # Visualization B: Residual Analysis
                st.markdown("### 2. Residual Analysis (Where did we lose points?)")
                st.markdown("This chart isolates the error per player. Positive bars mean the player performed better than expected.")
                
                viz_df['error'] = viz_df['total_points'] - viz_df['adjusted_points']
                
                fig_res, ax_res = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    x='name', 
                    y='error', 
                    data=viz_df.sort_values('error'), 
                    palette='RdBu', # Red for negative error, Blue for positive
                    ax=ax_res
                )
                ax_res.axhline(0, color='black', linewidth=1)
                ax_res.set_title("Prediction Error by Player")
                ax_res.set_ylabel("Error (Actual - Predicted)")
                plt.xticks(rotation=45)
                st.pyplot(fig_res)
                
            else:
                st.info("👉 Please generate a squad in the first tab to see the specific performance analysis.")