import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import hashlib
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from itertools import combinations
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify)
from werkzeug.utils import secure_filename
import google.genai as genai

warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score,
                              classification_report)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'clarix-retail-2024-dev-secret')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, 'clarix_retail.db')
DATA_DIR   = os.path.join(BASE_DIR, '8451_The_Complete_Journey_2_Sample-2')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Jinja2 filter ───────────────────────────────────────────────────────────
@app.template_filter('format_number')
def format_number(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return value

# ─── Database helpers ─────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT UNIQUE NOT NULL,
            email        TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS households (
            hshd_num         INTEGER PRIMARY KEY,
            loyalty_flag     TEXT,
            age_range        TEXT,
            marital          TEXT,
            income_range     TEXT,
            homeowner        TEXT,
            hshd_composition TEXT,
            hh_size          TEXT,
            children         TEXT
        );
        CREATE TABLE IF NOT EXISTS products (
            product_num          TEXT PRIMARY KEY,
            department           TEXT,
            commodity            TEXT,
            brand_type           TEXT,
            natural_organic_flag TEXT
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            basket_num    TEXT,
            hshd_num      INTEGER,
            purchase_date TEXT,
            product_num   TEXT,
            spend         REAL,
            units         INTEGER,
            store_region  TEXT,
            week_num      INTEGER,
            year          INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_tx_hshd    ON transactions(hshd_num);
        CREATE INDEX IF NOT EXISTS idx_tx_product ON transactions(product_num);
        CREATE INDEX IF NOT EXISTS idx_tx_basket  ON transactions(basket_num);
    """)
    conn.commit()
    conn.close()


def _clean_df(df):
    """Strip whitespace from all column names and string values; replace 'null' with None."""
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    return df.replace({'null': None, 'NULL': None, 'nan': None, 'None': None})


def load_csv_to_db(hh_path, tx_path, prod_path):
    conn = get_db()
    conn.execute("DELETE FROM transactions")
    conn.execute("DELETE FROM households")
    conn.execute("DELETE FROM products")
    conn.commit()

    # ── Households ──
    hh = _clean_df(pd.read_csv(hh_path, skipinitialspace=True))
    hh_map = {
        'HSHD_NUM': 'hshd_num', 'L': 'loyalty_flag', 'AGE_RANGE': 'age_range',
        'MARITAL': 'marital', 'INCOME_RANGE': 'income_range',
        'HOMEOWNER': 'homeowner', 'HSHD_COMPOSITION': 'hshd_composition',
        'HH_SIZE': 'hh_size', 'CHILDREN': 'children'
    }
    hh = hh.rename(columns=hh_map)
    keep = [c for c in ['hshd_num','loyalty_flag','age_range','marital','income_range',
                         'homeowner','hshd_composition','hh_size','children'] if c in hh.columns]
    hh[keep].to_sql('households', conn, if_exists='append', index=False)

    # ── Products ──
    prod = _clean_df(pd.read_csv(prod_path, skipinitialspace=True))
    prod_map = {
        'PRODUCT_NUM': 'product_num', 'DEPARTMENT': 'department',
        'COMMODITY': 'commodity', 'BRAND_TY': 'brand_type',
        'NATURAL_ORGANIC_FLAG': 'natural_organic_flag'
    }
    prod = prod.rename(columns=prod_map)
    keep = [c for c in ['product_num','department','commodity','brand_type',
                         'natural_organic_flag'] if c in prod.columns]
    prod[keep].drop_duplicates(subset=['product_num']).to_sql(
        'products', conn, if_exists='append', index=False)

    # ── Transactions (chunked) ──
    total = 0
    tx_map = {
        'BASKET_NUM': 'basket_num', 'HSHD_NUM': 'hshd_num',
        'PURCHASE_': 'purchase_date', 'PRODUCT_NUM': 'product_num',
        'SPEND': 'spend', 'UNITS': 'units', 'STORE_R': 'store_region',
        'WEEK_NUM': 'week_num', 'YEAR': 'year'
    }
    for chunk in pd.read_csv(tx_path, skipinitialspace=True, chunksize=50_000):
        chunk = _clean_df(chunk)
        chunk = chunk.rename(columns=tx_map)
        keep = [c for c in ['basket_num','hshd_num','purchase_date','product_num',
                              'spend','units','store_region','week_num','year']
                if c in chunk.columns]
        chunk[keep].to_sql('transactions', conn, if_exists='append', index=False)
        total += len(chunk)

    conn.close()
    return total


# ─── Auth helpers ─────────────────────────────────────────────────────────────
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ─── Auth routes ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username=? AND password_hash=?',
            (username, hash_pw(password))
        ).fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        if not all([username, email, password]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        try:
            conn = get_db()
            conn.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?,?,?)',
                (username, email, hash_pw(password))
            )
            conn.commit()
            conn.close()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'danger')
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ─── Data loading routes ──────────────────────────────────────────────────────
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        files = {
            'households':   request.files.get('households'),
            'transactions': request.files.get('transactions'),
            'products':     request.files.get('products'),
        }
        if not all(files.values()):
            flash('All three CSV files are required.', 'danger')
            return redirect(url_for('upload'))
        paths = {}
        for key, f in files.items():
            path = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
            f.save(path)
            paths[key] = path
        try:
            total = load_csv_to_db(paths['households'], paths['transactions'], paths['products'])
            flash(f'Data loaded successfully — {total:,} transaction records.', 'success')
        except Exception as e:
            flash(f'Error loading data: {e}', 'danger')
        return redirect(url_for('upload'))

    if request.args.get('load_default') == '1':
        hh   = os.path.join(DATA_DIR, '400_households.csv')
        tx   = os.path.join(DATA_DIR, '400_transactions.csv')
        prod = os.path.join(DATA_DIR, '400_products.csv')
        if all(os.path.exists(p) for p in [hh, tx, prod]):
            try:
                total = load_csv_to_db(hh, tx, prod)
                flash(f'Default data loaded — {total:,} transaction records.', 'success')
            except Exception as e:
                flash(f'Error: {e}', 'danger')
        else:
            flash('Default CSV files not found on server.', 'danger')
        return redirect(url_for('upload'))

    conn = get_db()
    counts = {
        'households':   conn.execute('SELECT COUNT(*) FROM households').fetchone()[0],
        'transactions': conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0],
        'products':     conn.execute('SELECT COUNT(*) FROM products').fetchone()[0],
    }
    conn.close()
    return render_template('upload.html', counts=counts)


# ─── Data pull ────────────────────────────────────────────────────────────────
@app.route('/data-pull')
@login_required
def data_pull():
    raw = request.args.get('hshd_num', '10')
    try:
        hshd_num = int(raw)
    except ValueError:
        hshd_num = 10

    conn = get_db()
    rows = conn.execute("""
        SELECT  t.hshd_num, t.basket_num, t.purchase_date, t.product_num,
                p.department, p.commodity, t.spend, t.units,
                t.store_region, t.week_num, t.year,
                h.loyalty_flag, h.age_range, h.marital, h.income_range,
                h.homeowner, h.hshd_composition, h.hh_size, h.children
        FROM   transactions t
        LEFT JOIN households h ON t.hshd_num  = h.hshd_num
        LEFT JOIN products   p ON t.product_num = p.product_num
        WHERE  t.hshd_num = ?
        ORDER  BY t.hshd_num, t.basket_num, t.purchase_date,
                  t.product_num, p.department, p.commodity
        LIMIT  1000
    """, (hshd_num,)).fetchall()
    conn.close()
    return render_template('data_pull.html', rows=rows, hshd_num=hshd_num)


# ─── Dashboard ────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    stats = {
        'households':   conn.execute('SELECT COUNT(*) FROM households').fetchone()[0],
        'transactions': conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0],
        'products':     conn.execute('SELECT COUNT(*) FROM products').fetchone()[0],
        'total_spend':  conn.execute('SELECT ROUND(SUM(spend),2) FROM transactions').fetchone()[0] or 0,
    }
    conn.close()
    return render_template('dashboard.html', stats=stats)


@app.route('/api/spending-over-time')
@login_required
def api_spending_over_time():
    conn = get_db()
    rows = conn.execute("""
        SELECT year, week_num, ROUND(SUM(spend),2) AS total
        FROM   transactions
        GROUP  BY year, week_num
        ORDER  BY year, week_num
    """).fetchall()
    conn.close()
    buckets = {}
    for r in rows:
        key = f"{r['year']}-M{(r['week_num']-1)//4+1:02d}"
        buckets[key] = round(buckets.get(key, 0) + r['total'], 2)
    return jsonify({'labels': list(buckets.keys()), 'values': list(buckets.values())})


@app.route('/api/top-commodities')
@login_required
def api_top_commodities():
    conn = get_db()
    rows = conn.execute("""
        SELECT p.commodity, ROUND(SUM(t.spend),2) AS total
        FROM   transactions t
        JOIN   products p ON t.product_num = p.product_num
        WHERE  p.commodity IS NOT NULL AND p.commodity NOT IN ('None','')
        GROUP  BY p.commodity
        ORDER  BY total DESC LIMIT 15
    """).fetchall()
    conn.close()
    return jsonify({'labels': [r['commodity'] for r in rows],
                    'values': [r['total']     for r in rows]})


@app.route('/api/demographics')
@login_required
def api_demographics():
    conn = get_db()
    inc = conn.execute("""
        SELECT h.income_range, ROUND(AVG(s.total),2) AS avg_spend
        FROM   households h
        JOIN   (SELECT hshd_num, SUM(spend) AS total FROM transactions GROUP BY hshd_num) s
               ON h.hshd_num = s.hshd_num
        WHERE  h.income_range IS NOT NULL AND h.income_range NOT IN ('None','')
        GROUP  BY h.income_range ORDER BY avg_spend DESC
    """).fetchall()
    sz = conn.execute("""
        SELECT h.hh_size, ROUND(AVG(s.total),2) AS avg_spend
        FROM   households h
        JOIN   (SELECT hshd_num, SUM(spend) AS total FROM transactions GROUP BY hshd_num) s
               ON h.hshd_num = s.hshd_num
        WHERE  h.hh_size IS NOT NULL AND h.hh_size NOT IN ('None','')
        GROUP  BY h.hh_size ORDER BY h.hh_size
    """).fetchall()
    conn.close()
    return jsonify({
        'income': {'labels': [r['income_range'] for r in inc], 'values': [r['avg_spend'] for r in inc]},
        'size':   {'labels': [r['hh_size']      for r in sz],  'values': [r['avg_spend'] for r in sz]},
    })


@app.route('/api/brand-preference')
@login_required
def api_brand_preference():
    conn = get_db()
    rows = conn.execute("""
        SELECT p.brand_type, ROUND(SUM(t.spend),2) AS total
        FROM   transactions t
        JOIN   products p ON t.product_num = p.product_num
        WHERE  p.brand_type IS NOT NULL AND p.brand_type NOT IN ('None','')
        GROUP  BY p.brand_type
    """).fetchall()
    conn.close()
    return jsonify({'labels': [r['brand_type'] for r in rows],
                    'values': [r['total']       for r in rows]})


@app.route('/api/regional-stats')
@login_required
def api_regional_stats():
    conn = get_db()
    rows = conn.execute("""
        SELECT store_region,
               ROUND(SUM(spend),2)          AS total_spend,
               COUNT(DISTINCT hshd_num)     AS hh_count
        FROM   transactions
        WHERE  store_region IS NOT NULL AND store_region NOT IN ('None','')
        GROUP  BY store_region ORDER BY total_spend DESC
    """).fetchall()
    conn.close()
    return jsonify({'labels':     [r['store_region'] for r in rows],
                    'spend':      [r['total_spend']  for r in rows],
                    'households': [r['hh_count']     for r in rows]})


@app.route('/api/loyalty-stats')
@login_required
def api_loyalty_stats():
    conn = get_db()
    rows = conn.execute("""
        SELECT h.loyalty_flag,
               ROUND(AVG(s.total),2)    AS avg_spend,
               COUNT(DISTINCT h.hshd_num) AS hh_count
        FROM   households h
        JOIN   (SELECT hshd_num, SUM(spend) AS total FROM transactions GROUP BY hshd_num) s
               ON h.hshd_num = s.hshd_num
        WHERE  h.loyalty_flag IS NOT NULL AND h.loyalty_flag NOT IN ('None','')
        GROUP  BY h.loyalty_flag
    """).fetchall()
    conn.close()
    return jsonify({'labels': [r['loyalty_flag'] for r in rows],
                    'values': [r['avg_spend']    for r in rows],
                    'counts': [r['hh_count']     for r in rows]})


@app.route('/api/children-spend')
@login_required
def api_children_spend():
    conn = get_db()
    rows = conn.execute("""
        SELECT h.children,
               ROUND(AVG(s.total),2) AS avg_spend,
               COUNT(DISTINCT h.hshd_num) AS hh_count
        FROM   households h
        JOIN   (SELECT hshd_num, SUM(spend) AS total FROM transactions GROUP BY hshd_num) s
               ON h.hshd_num = s.hshd_num
        WHERE  h.children IS NOT NULL AND h.children NOT IN ('None','')
        GROUP  BY h.children
    """).fetchall()
    conn.close()
    return jsonify({'labels': [r['children']  for r in rows],
                    'values': [r['avg_spend'] for r in rows]})


# ─── ML — Write-up & CLV ──────────────────────────────────────────────────────
@app.route('/ml-writeup')
@login_required
def ml_writeup():
    conn = get_db()
    df = pd.read_sql("""
        SELECT t.hshd_num, t.purchase_date, t.spend, t.basket_num,
               h.income_range, h.hh_size, h.children, h.loyalty_flag
        FROM   transactions t
        LEFT JOIN households h ON t.hshd_num = h.hshd_num
    """, conn)
    conn.close()

    clv_results = None
    if not df.empty:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
        df = df.dropna(subset=['purchase_date'])
        max_date = df['purchase_date'].max()

        rfm = df.groupby('hshd_num').agg(
            recency=('purchase_date', lambda x: (max_date - x.max()).days),
            frequency=('basket_num', 'nunique'),
            monetary=('spend', 'sum'),
        ).reset_index()

        demo = df[['hshd_num','income_range','hh_size','children','loyalty_flag']].drop_duplicates('hshd_num')
        rfm = rfm.merge(demo, on='hshd_num', how='left')

        for col in ['income_range','hh_size','children','loyalty_flag']:
            rfm[col] = rfm[col].fillna('Unknown').astype(str)
            rfm[col] = LabelEncoder().fit_transform(rfm[col])

        X = rfm[['recency','frequency','income_range','hh_size','children','loyalty_flag']]
        y = rfm['monetary']

        if len(X) > 30:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            models = {
                'Linear Regression':  LinearRegression(),
                'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
            }
            clv_results = []
            for name, mdl in models.items():
                fit_X = X_tr_s if name == 'Linear Regression' else X_tr
                prd_X = X_te_s if name == 'Linear Regression' else X_te
                mdl.fit(fit_X, y_tr)
                preds = mdl.predict(prd_X)
                rmse = round(np.sqrt(mean_squared_error(y_te, preds)), 2)
                r2   = round(r2_score(y_te, preds), 4)
                clv_results.append({'model': name, 'rmse': rmse, 'r2': r2})

    return render_template('ml_writeup.html', clv_results=clv_results)


# ─── ML — Basket Analysis ─────────────────────────────────────────────────────
@app.route('/basket-analysis')
@login_required
def basket_analysis():
    conn = get_db()
    df = pd.read_sql("""
        SELECT t.basket_num, p.commodity
        FROM   transactions t
        JOIN   products p ON t.product_num = p.product_num
        WHERE  p.commodity IS NOT NULL AND p.commodity NOT IN ('None','')
    """, conn)
    conn.close()

    if df.empty:
        return render_template('basket_analysis.html',
                               rules=[], rf_results=None, total_baskets=0,
                               error="No data loaded yet.")

    baskets = df.groupby('basket_num')['commodity'].apply(set)
    total_baskets = len(baskets)

    comm_counts = {}
    pair_counts = {}
    for comms in baskets:
        lst = list(comms)
        for c in lst:
            comm_counts[c] = comm_counts.get(c, 0) + 1
        for pair in combinations(sorted(lst), 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    rules = []
    MIN_SUPPORT = 0.005
    for (A, B), cnt in pair_counts.items():
        sup = cnt / total_baskets
        if sup < MIN_SUPPORT:
            continue
        conf_ab = cnt / comm_counts[A]
        conf_ba = cnt / comm_counts[B]
        lift    = sup / ((comm_counts[A] / total_baskets) * (comm_counts[B] / total_baskets))
        rules.append({'item_a': A, 'item_b': B,
                      'support': round(sup, 4),
                      'conf_ab': round(conf_ab, 4),
                      'conf_ba': round(conf_ba, 4),
                      'lift':    round(lift, 4),
                      'count':   cnt})
    rules.sort(key=lambda x: x['lift'], reverse=True)
    top_rules = rules[:30]

    rf_results = None
    top_comms = df['commodity'].value_counts().head(25).index.tolist()
    df_top = df[df['commodity'].isin(top_comms)]
    bm = df_top.groupby(['basket_num','commodity']).size().unstack(fill_value=0)
    bm = (bm > 0).astype(int)

    if len(bm) > 200 and len(bm.columns) > 3:
        target = bm.columns[0]
        X = bm.drop(columns=[target])
        y = bm[target]
        if y.sum() > 20:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            acc = round(accuracy_score(y_te, rf.predict(X_te)) * 100, 1)
            imp = (pd.DataFrame({'commodity': X.columns, 'importance': rf.feature_importances_})
                   .sort_values('importance', ascending=False)
                   .head(10)
                   .to_dict('records'))
            rf_results = {'target': target, 'accuracy': acc,
                          'importances': imp, 'n_baskets': len(bm)}

    return render_template('basket_analysis.html',
                           rules=top_rules,
                           rf_results=rf_results,
                           total_baskets=total_baskets,
                           error=None)


# ─── ML — Churn Prediction ────────────────────────────────────────────────────
@app.route('/churn-prediction')
@login_required
def churn_prediction():
    conn = get_db()
    df = pd.read_sql("""
        SELECT t.hshd_num, t.purchase_date, t.spend, t.basket_num,
               h.loyalty_flag, h.age_range, h.income_range,
               h.hh_size, h.children, h.marital, h.homeowner
        FROM   transactions t
        LEFT JOIN households h ON t.hshd_num = h.hshd_num
    """, conn)
    conn.close()

    if df.empty:
        return render_template('churn.html', error="No data loaded yet.")

    df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
    df = df.dropna(subset=['purchase_date'])
    max_date = df['purchase_date'].max()
    CHURN_DAYS = 90

    rfm = df.groupby('hshd_num').agg(
        recency=('purchase_date', lambda x: (max_date - x.max()).days),
        frequency=('basket_num', 'nunique'),
        monetary=('spend', 'sum'),
    ).reset_index()
    rfm['monetary'] = rfm['monetary'].round(2)
    rfm['churned']  = (rfm['recency'] > CHURN_DAYS).astype(int)

    demo_cols = ['hshd_num','loyalty_flag','age_range','income_range',
                 'hh_size','children','marital','homeowner']
    demo = df[demo_cols].drop_duplicates('hshd_num')
    rfm  = rfm.merge(demo, on='hshd_num', how='left')

    for c in ['loyalty_flag','age_range','income_range','hh_size','children','marital','homeowner']:
        rfm[c] = rfm[c].fillna('Unknown').astype(str).str.strip()

    churn_rate = round(rfm['churned'].mean() * 100, 1)
    n_churned  = int(rfm['churned'].sum())
    n_active   = len(rfm) - n_churned

    seg = (rfm.groupby('churned')
              .agg(avg_recency=('recency','mean'),
                   avg_frequency=('frequency','mean'),
                   avg_monetary=('monetary','mean'))
              .round(2).reset_index())
    seg['label'] = seg['churned'].map({0:'Active', 1:'Churned'})

    def churn_by(col):
        g = rfm.groupby(col)['churned'].agg(['mean','count']).reset_index()
        g.columns = [col, 'churn_rate', 'count']
        g['churn_rate'] = (g['churn_rate'] * 100).round(1)
        g = g[g[col] != 'Unknown'].sort_values('churn_rate', ascending=False)
        return g.to_dict('records')

    seg_income   = churn_by('income_range')
    seg_hh_size  = churn_by('hh_size')
    seg_children = churn_by('children')
    seg_loyalty  = churn_by('loyalty_flag')
    seg_age      = churn_by('age_range')

    freq_bins = list(range(0, int(rfm['frequency'].max()) + 15, 5))
    mon_bins  = [0,200,400,600,800,1000,1500,2000,3000,5000,
                 int(rfm['monetary'].max())+1]

    def hist_data(col, bins):
        active  = rfm[rfm['churned']==0][col]
        churned = rfm[rfm['churned']==1][col]
        labels  = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        a_counts = pd.cut(active,  bins=bins).value_counts().reindex(
                       pd.IntervalIndex.from_breaks(bins), fill_value=0).tolist()
        c_counts = pd.cut(churned, bins=bins).value_counts().reindex(
                       pd.IntervalIndex.from_breaks(bins), fill_value=0).tolist()
        return {'labels': labels, 'active': a_counts, 'churned': c_counts}

    freq_dist = hist_data('frequency', freq_bins)
    mon_dist  = hist_data('monetary',  mon_bins)

    cat_cols = ['loyalty_flag','age_range','income_range','hh_size','children']
    num_cols = ['frequency','monetary']
    ml = rfm[num_cols + cat_cols + ['churned']].copy()
    le = LabelEncoder()
    for c in cat_cols:
        ml[c] = le.fit_transform(ml[c])

    corr_full = ml.corr()['churned'].drop('churned').round(4)
    corr_data = {'labels': list(corr_full.index),
                 'values': list(corr_full.values)}

    model_results = None
    X, y = ml[num_cols + cat_cols], ml['churned']
    if len(X) > 30 and y.nunique() > 1:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)

        gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                        max_depth=4, random_state=42)
        gb.fit(X_tr, y_tr)
        gb_pred = gb.predict(X_te)
        gb_acc  = round(accuracy_score(y_te, gb_pred) * 100, 1)
        gb_imp  = (pd.DataFrame({'feature': X.columns,
                                 'importance': gb.feature_importances_})
                   .sort_values('importance', ascending=False)
                   .to_dict('records'))

        sc = StandardScaler()
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(sc.fit_transform(X_tr), y_tr)
        lr_acc  = round(accuracy_score(y_te, lr.predict(sc.transform(X_te))) * 100, 1)
        lr_coef = [{'feature': f, 'coef': round(float(c), 4)}
                   for f, c in zip(X.columns, lr.coef_[0])]
        lr_coef.sort(key=lambda x: abs(x['coef']), reverse=True)

        rfm_ml = rfm[num_cols + cat_cols].copy()
        for c in cat_cols:
            rfm_ml[c] = le.fit_transform(rfm_ml[c].astype(str))
        rfm['churn_prob'] = (gb.predict_proba(rfm_ml[num_cols + cat_cols])[:, 1] * 100).round(1)

        at_risk = (rfm[rfm['churned']==0]
                   .sort_values('churn_prob', ascending=False)
                   .head(20)[['hshd_num','churn_prob','frequency',
                               'monetary','recency','income_range','hh_size']]
                   .to_dict('records'))

        model_results = {
            'gb_accuracy': gb_acc,
            'lr_accuracy': lr_acc,
            'gb_imp':      gb_imp,
            'lr_coef':     lr_coef,
            'n_train':     len(X_tr),
            'n_test':      len(X_te),
            'at_risk':     at_risk,
        }

    return render_template('churn.html',
                           churn_rate=churn_rate,
                           n_churned=n_churned,
                           n_active=n_active,
                           total_hh=len(rfm),
                           churn_days=CHURN_DAYS,
                           segments=seg.to_dict('records'),
                           seg_income=seg_income,
                           seg_hh_size=seg_hh_size,
                           seg_children=seg_children,
                           seg_loyalty=seg_loyalty,
                           seg_age=seg_age,
                           freq_dist=freq_dist,
                           mon_dist=mon_dist,
                           corr_data=corr_data,
                           model_results=model_results,
                           error=None)


# ─── Natural Language Query (Gemini 2.5 Flash) ───────────────────────────────
DB_SCHEMA = """
Tables in the SQLite database:

households(hshd_num INTEGER, loyalty_flag TEXT, age_range TEXT, marital TEXT,
           income_range TEXT, homeowner TEXT, hshd_composition TEXT,
           hh_size TEXT, children TEXT)

transactions(id INTEGER, basket_num TEXT, hshd_num INTEGER, purchase_date TEXT,
             product_num TEXT, spend REAL, units INTEGER, store_region TEXT,
             week_num INTEGER, year INTEGER)

products(product_num TEXT, department TEXT, commodity TEXT,
         brand_type TEXT, natural_organic_flag TEXT)

Relationships:
  transactions.hshd_num  → households.hshd_num
  transactions.product_num → products.product_num

Notes:
  - purchase_date is stored as text e.g. '17-AUG-18'
  - loyalty_flag values: 'Y' or 'N'
  - brand_type values: 'PRIVATE' or 'NATIONAL'
  - natural_organic_flag values: 'Y' or 'N'
  - store_region examples: 'CENTRAL', 'EAST', 'SOUTH', 'WEST'
"""

BLOCKED = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|PRAGMA)\b',
    re.IGNORECASE
)

def nl_to_sql(question: str) -> str:
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    prompt = f"""You are an expert SQL assistant for a retail analytics database.

{DB_SCHEMA}

Convert the following plain-English question into a single, valid SQLite SELECT query.
Rules:
- Return ONLY the raw SQL query, no markdown, no explanation, no code fences.
- Use only SELECT statements. No INSERT, UPDATE, DELETE, DROP or DDL.
- Always use LEFT JOIN when joining tables so missing data doesn't drop rows.
- Limit results to 200 rows unless the user specifies otherwise.
- Use ROUND(..., 2) for monetary values.
- Use UPPER() for case-insensitive text comparisons where appropriate.

Question: {question}

SQL:"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    sql = response.text.strip()
    sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'^```\s*',    '', sql)
    sql = re.sub(r'```$',       '', sql).strip()
    return sql


@app.route('/nl-query', methods=['GET', 'POST'])
@login_required
def nl_query():
    result = None
    sql    = None
    question = ''
    error  = None

    api_key_set = bool(os.environ.get('GEMINI_API_KEY', ''))

    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            error = "Please enter a question."
        elif not api_key_set:
            error = "GEMINI_API_KEY is not configured. Set it as an environment variable and restart the server."
        else:
            try:
                sql = nl_to_sql(question)

                if BLOCKED.search(sql) or not sql.upper().lstrip().startswith('SELECT'):
                    error = "Generated query was not a SELECT statement and was blocked for safety."
                    sql   = None
                else:
                    conn = get_db()
                    try:
                        rows = conn.execute(sql).fetchall()
                        if rows:
                            cols   = rows[0].keys()
                            result = {'columns': list(cols),
                                      'rows':    [list(r) for r in rows],
                                      'count':   len(rows)}
                        else:
                            result = {'columns': [], 'rows': [], 'count': 0}
                    except Exception as db_err:
                        error = f"SQL execution error: {db_err}"
                    finally:
                        conn.close()

            except ValueError as ve:
                error = str(ve)
            except Exception as e:
                error = f"Gemini API error: {e}"

    examples = [
        "Which 10 households have the highest total spend?",
        "What are the top 5 commodities by total revenue?",
        "How does average spend differ between loyal and non-loyal customers?",
        "Show churn risk: households with no purchase in the last 90 days",
        "What is the average basket size by store region?",
        "Which income range spends the most on organic products?",
        "List households with children and their average weekly spend",
        "What are the most commonly bought product pairs (commodity level)?",
        "Show monthly revenue trend for year 2019",
        "Which department has the highest private label vs national brand split?",
    ]

    return render_template('nl_query.html',
                           question=question,
                           sql=sql,
                           result=result,
                           error=error,
                           examples=examples,
                           api_key_set=api_key_set)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    print("Database initialised.")
    print("Open http://localhost:8000 in your browser.")
    print("Register a new account, then go to Load Data → 'Load Default CSVs'.")
    app.run(debug=True, host='0.0.0.0', port=8000)
