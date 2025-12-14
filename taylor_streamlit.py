"""泰勒展开可视化 - 优化版 v3 (暗色/亮色自适应)"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import factorial
from typing import Callable, Optional, Tuple
import re

# ==================== 配置 ====================
st.set_page_config(page_title="泰勒展开可视化", page_icon="🧮", layout="wide")

# 检测主题
try:
    is_dark = st.get_option("theme.base") == "dark"
except:
    is_dark = False

# 自适应样式
st.markdown(f"""<style>
.main-header {{
    text-align: center;
    background: linear-gradient(120deg, #00d4ff, #7b68ee, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: bold;
}}
.subtitle {{
    text-align: center;
    color: {'#b0b0b0' if is_dark else '#666'};
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}}
.formula-box {{
    background: {'linear-gradient(135deg, #2a2a4e, #1a3050)' if is_dark else 'linear-gradient(135deg, #1a1a2e, #16213e)'};
    padding: 1rem;
    border-radius: 10px;
    color: #ffffff;
    text-align: center;
}}
.range-box {{
    background: {'#1a3a1a' if is_dark else '#e8f4e8'};
    padding: 0.8rem;
    border-radius: 8px;
    border-left: 4px solid {'#66bb6a' if is_dark else '#4CAF50'};
    color: {'#fafafa' if is_dark else '#1a1a1a'};
}}
[data-testid="stMetricValue"] {{
    color: {'#fafafa' if is_dark else '#1a1a1a'} !important;
}}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧮 泰勒展开可视化</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">探索泰勒级数如何逼近各种函数</p>', unsafe_allow_html=True)

# ==================== 核心函数 ====================
@st.cache_data
def get_factorials(n: int) -> np.ndarray:
    return np.array([factorial(i) for i in range(n+1)], dtype=float)

@st.cache_data
def binomial_coeffs(n: int) -> np.ndarray:
    return np.array([((-1)**k)*factorial(n)/(factorial(k)*factorial(n-k)) for k in range(n+1)])

def gaussian_deriv(x0, n):
    if n == 0: return np.exp(-x0**2)
    H = [1.0, 2*x0]
    for k in range(2, n+1): H.append(2*x0*H[-1] - 2*(k-1)*H[-2])
    return ((-1)**n) * H[n] * np.exp(-x0**2)

def super_gaussian_deriv(x0, n):
    if x0 != 0: return None
    if n % 4 != 0: return 0.0
    return ((-1)**(n//4)) * factorial(n) / factorial(n//4)

def arctan_deriv(x0, n):
    if x0 != 0: return None
    if n == 0: return 0.0
    if n % 2 == 0: return 0.0
    return ((-1)**((n-1)//2)) * factorial(n-1)

def binomial_deriv(alpha):
    def d(x0, n):
        if n == 0: return (1+x0)**alpha
        c = 1.0
        for k in range(n): c *= (alpha - k)
        return c * ((1+x0)**(alpha-n))
    return d

def numerical_deriv(func, x0, n, h=0.01):
    if n == 0:
        v = func(np.array([x0]))
        return float(v[0]) if hasattr(v, '__len__') else float(v)
    c = binomial_coeffs(n)
    pts = x0 + (n/2 - np.arange(n+1)) * h
    try:
        vals = func(pts)
        if not hasattr(vals, '__len__'): vals = np.array([func(p) for p in pts])
    except: vals = np.array([func(p) for p in pts])
    return float(np.sum(c * vals) / (h**n))

def taylor_expand(x, x0, terms, func, deriv):
    dx, facts = x - x0, get_factorials(terms)
    derivs = np.zeros(terms)
    for n in range(terms):
        if deriv == "numerical": derivs[n] = numerical_deriv(func, x0, n)
        else:
            v = deriv(x0, n)
            derivs[n] = numerical_deriv(func, x0, n) if v is None else v
    result, power = np.zeros_like(x, dtype=float), np.ones_like(x)
    for n in range(terms):
        result += derivs[n] * power / facts[n]
        power *= dx
    return result

def estimate_range(func, deriv, x0, terms, threshold=0.01):
    """估算泰勒展开的有效拟合范围，使用绝对误差"""
    def taylor_at(xv):
        r, facts = 0.0, get_factorials(terms)
        for n in range(terms):
            if deriv == "numerical":
                d = numerical_deriv(func, x0, n)
            else:
                d = deriv(x0, n)
                if d is None:  # 只有 None 才回退，0 是有效值
                    d = numerical_deriv(func, x0, n)
            r += d * (xv - x0)**n / facts[n]
        return r
    
    def find_bound(direction):
        lo, hi = 0.0, 15.0
        for _ in range(25):
            mid = (lo + hi) / 2
            xt = x0 + direction * mid
            try:
                fv = func(np.array([xt]))[0] if hasattr(func(np.array([xt])), '__len__') else func(xt)
                tv = taylor_at(xt)
                if not np.isfinite(fv) or not np.isfinite(tv): 
                    hi = mid
                else:
                    abs_err = abs(fv - tv)
                    if abs_err > threshold:
                        hi = mid
                    else:
                        lo = mid
            except: 
                hi = mid
        return lo
    
    return (x0 - find_bound(-1), x0 + find_bound(1))

def format_expr(x0, terms, func, deriv):
    parts = []
    for n in range(min(terms, 6)):
        if deriv == "numerical":
            d = numerical_deriv(func, x0, n)
        else:
            d = deriv(x0, n)
            if d is None: d = numerical_deriv(func, x0, n)
        c = d / factorial(n) if n else d
        if abs(c) < 1e-10: continue
        cs = str(int(round(c))) if abs(c-round(c))<1e-6 else f"{c:.3f}"
        if n == 0: t = cs
        elif x0 == 0: t = f"{cs}x^{n}" if n>1 else f"{cs}x"
        else: t = f"{cs}(x-{x0:.1f})^{n}"
        parts.append((" + " if parts and c>0 else " ") + t if parts else t)
    if terms > 6: parts.append(" + ...")
    return "".join(parts) or "0"

# ==================== 函数库 ====================
FUNCTIONS = {
    "sin(x)": {"f": np.sin, "d": lambda x0,n: [np.sin(x0),np.cos(x0),-np.sin(x0),-np.cos(x0)][n%4],
               "formula": "sin(x) = x - x³/3! + x⁵/5! - ...", "desc": "正弦函数", "y_range": (-2,2), "conv": "全局"},
    "cos(x)": {"f": np.cos, "d": lambda x0,n: [np.cos(x0),-np.sin(x0),-np.cos(x0),np.sin(x0)][n%4],
               "formula": "cos(x) = 1 - x²/2! + x⁴/4! - ...", "desc": "余弦函数", "y_range": (-2,2), "conv": "全局"},
    "exp(x)": {"f": np.exp, "d": lambda x0,n: np.exp(x0),
               "formula": "eˣ = 1 + x + x²/2! + x³/3! + ...", "desc": "指数函数", "y_range": (-1,10), "conv": "全局"},
    "exp(-x²) 高斯": {"f": lambda x: np.exp(-x**2), "d": gaussian_deriv,
                      "formula": "e^(-x²) Hermite多项式", "desc": "高斯函数", "y_range": (-0.5,1.5), "conv": "全局"},
    "exp(-x⁴) 超高斯": {"f": lambda x: np.exp(-x**4), "d": super_gaussian_deriv,
                        "formula": "e^(-x⁴) 只有4n次项", "desc": "超高斯 (x₀=0)", "y_range": (-0.5,1.5), "conv": "全局"},
    "arctan(x)": {"f": np.arctan, "d": arctan_deriv,
                  "formula": "arctan(x) = x - x³/3 + x⁵/5 - ...", "desc": "反正切 (x₀=0)", "y_range": (-2,2), "conv": "|x|≤1"},
    "sqrt(1+x)": {"f": lambda x: np.sqrt(1+x), "d": binomial_deriv(0.5),
                  "formula": "√(1+x) 广义二项式", "desc": "平方根", "y_range": (0,3), "x_range": (-0.9,3), "conv": "|x|<1"},
    "ln(1+x)": {"f": lambda x: np.log(1+x), "d": lambda x0,n: 0 if n==0 else ((-1)**(n+1)*factorial(n-1))/((1+x0)**n),
                "formula": "ln(1+x) = x - x²/2 + x³/3 - ...", "desc": "自然对数", "y_range": (-3,3), "x_range": (-0.9,2), "conv": "|x|<1"},
    "sinh(x)": {"f": np.sinh, "d": lambda x0,n: np.sinh(x0) if n%2==0 else np.cosh(x0),
                "formula": "sinh(x) = x + x³/3! + x⁵/5! + ...", "desc": "双曲正弦", "y_range": (-5,5), "conv": "全局"},
    "cosh(x)": {"f": np.cosh, "d": lambda x0,n: np.cosh(x0) if n%2==0 else np.sinh(x0),
                "formula": "cosh(x) = 1 + x²/2! + x⁴/4! + ...", "desc": "双曲余弦", "y_range": (-1,10), "conv": "全局"},
}

PATTERNS = [(r"^exp\(-x\*\*2\)$","exp(-x²) 高斯"), (r"^exp\(-x\*\*4\)$","exp(-x⁴) 超高斯"),
            (r"^sin\(x\)$","sin(x)"), (r"^cos\(x\)$","cos(x)"), (r"^exp\(x\)$","exp(x)"),
            (r"^arctan\(x\)$","arctan(x)"), (r"^sqrt\(1\+x\)$","sqrt(1+x)"),
            (r"^sinh\(x\)$","sinh(x)"), (r"^cosh\(x\)$","cosh(x)")]

SAFE = {"sin":np.sin,"cos":np.cos,"tan":np.tan,"exp":np.exp,"log":np.log,"ln":np.log,
        "sqrt":np.sqrt,"abs":np.abs,"arcsin":np.arcsin,"arccos":np.arccos,"arctan":np.arctan,
        "sinh":np.sinh,"cosh":np.cosh,"tanh":np.tanh,"pi":np.pi,"e":np.e}

def parse_custom(expr):
    e = expr.replace(" ","").replace("^","**").lower()
    for p, name in PATTERNS:
        if re.match(p, e) and name in FUNCTIONS:
            info = FUNCTIONS[name]
            return info["f"], True, name, info["d"]
    es = expr.replace("^","**")
    try:
        def fn(x): return eval(es, {"__builtins__":{}, "x":x, **SAFE})
        fn(np.linspace(-1,1,5))
        return fn, True, None, "numerical"
    except: return None, False, None, None

# ==================== 侧边栏 ====================
st.sidebar.header("📐 函数设置")
sel = st.sidebar.selectbox("选择函数", list(FUNCTIONS.keys()) + ["自定义函数"])

expr, matched, cust_d = "", None, None
if sel == "自定义函数":
    st.sidebar.markdown("---")
    expr = st.sidebar.text_input("函数表达式", "exp(-x**4)")
    func, valid, matched, cust_d = parse_custom(expr)
    if valid: st.sidebar.success(f"✅ {'识别: '+matched if matched else '数值微分'}")
    else: st.sidebar.error("❌ 无效"); st.stop()

st.sidebar.markdown("---")
terms = st.sidebar.slider("📊 展开项数 n", 1, 25, 5)
x0 = st.sidebar.slider("📍 展开点 x₀", -3.14, 3.14, 0.0, 0.1)
show_err = st.sidebar.checkbox("🔬 显示误差", False)

# 函数配置
if sel == "自定义函数":
    deriv, formula, desc = cust_d, f"f(x) = {expr}", matched or "自定义"
    y_range, conv = None, "需分析"
else:
    info = FUNCTIONS[sel]
    func, deriv, formula, desc = info["f"], info["d"], info["formula"], info["desc"]
    y_range, conv = info.get("y_range"), info.get("conv", "")

# 推荐范围
rec_min, rec_max = estimate_range(func, deriv, x0, terms, 0.01)
rec_min, rec_max = max(rec_min, -10), min(rec_max, 10)

st.sidebar.markdown("---")
st.sidebar.markdown(f'<div class="range-box"><b>🎯 推荐范围</b> (误差<1%)<br><code>[{rec_min:.2f}, {rec_max:.2f}]</code><br><small>收敛性: {conv}</small></div>', unsafe_allow_html=True)

use_rec = st.sidebar.checkbox("使用推荐范围", True)
if use_rec: x_min, x_max = rec_min, rec_max
else:
    c1, c2 = st.sidebar.columns(2)
    x_min, x_max = c1.number_input("最小", value=-6.28, step=0.5), c2.number_input("最大", value=6.28, step=0.5)
if x_min >= x_max: x_max = x_min + 1

# ==================== 计算 ====================
x = np.linspace(x_min, x_max, 500)
try: y_orig = np.where(np.isfinite(func(x)), func(x), np.nan)
except: y_orig = np.full_like(x, np.nan)
try: y_taylor = np.where(np.isfinite(taylor_expand(x, x0, terms, func, deriv)), taylor_expand(x, x0, terms, func, deriv), np.nan)
except: y_taylor = np.full_like(x, np.nan)
y_err = np.where(np.isfinite(np.abs(y_orig - y_taylor)), np.abs(y_orig - y_taylor), np.nan)

# ==================== 绘图 ====================
# 主题颜色
line_color = "#ffffff" if is_dark else "#000000"
bg_color = "#0e1117" if is_dark else "#ffffff"
grid_color = "#444444" if is_dark else "#e5e5e5"
text_color = "#fafafa" if is_dark else "#1a1a1a"
axis_color = "#fafafa" if is_dark else "#1a1a1a"
template = "plotly_dark" if is_dark else "plotly_white"

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_orig, mode='lines', name='f(x)', line=dict(color=line_color, width=3)))
fig.add_trace(go.Scatter(x=x, y=y_taylor, mode='lines', name=f'T_{terms}(x)', line=dict(color='#ff6b6b', width=2.5)))
if show_err:
    fig.add_trace(go.Scatter(x=x, y=y_err, mode='lines', name='误差', fill='tozeroy', 
                              fillcolor='rgba(255,107,107,0.2)', line=dict(color='rgba(255,107,107,0.5)', width=1)))

fig.add_vrect(x0=rec_min, x1=rec_max, fillcolor="rgba(76,175,80,0.15)" if is_dark else "rgba(76,175,80,0.1)", 
              line_width=0, annotation_text="推荐范围", annotation_position="top left",
              annotation=dict(font=dict(color=text_color)))
fig.add_vline(x=x0, line_dash="dash", line_color="#888", annotation_text=f"x₀={x0:.2f}",
              annotation=dict(font=dict(color=text_color)))

fig.update_layout(
    title=dict(text=f"{sel if sel != '自定义函数' else 'f(x)'} · n={terms} · 推荐范围[{rec_min:.1f}, {rec_max:.1f}]", font=dict(color=text_color)),
    xaxis=dict(title=dict(text="x", font=dict(color=axis_color)), gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=axis_color)),
    yaxis=dict(title=dict(text="y", font=dict(color=axis_color)), gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=axis_color)),
    template=template, height=450, paper_bgcolor=bg_color, plot_bgcolor=bg_color,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(128,128,128,0.3)", font=dict(color=text_color))
)
if y_range: fig.update_yaxes(range=list(y_range))
st.plotly_chart(fig, use_container_width=True)

# ==================== 展开式和统计 ====================
col1, col2 = st.columns([2, 1])
with col1:
    with st.expander("📝 泰勒展开式", expanded=True):
        st.latex(f"f(x) = {sel if sel != '自定义函数' else expr}")
        st.latex(f"T_n(x) = {format_expr(x0, terms, func, deriv)}")
        data = []
        for n in range(min(terms, 8)):
            if deriv == "numerical":
                dv = numerical_deriv(func, x0, n)
            else:
                dv = deriv(x0, n)
                if dv is None: dv = numerical_deriv(func, x0, n)
            data.append({"n": n, "f⁽ⁿ⁾(x₀)": f"{dv:.4f}", "系数": f"{dv/factorial(n) if n else dv:.4f}"})
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

with col2:
    st.markdown(f'<div class="formula-box"><b>{desc}</b><br><small>{formula}</small></div>', unsafe_allow_html=True)
    st.markdown("")
    valid_e = y_err[np.isfinite(y_err)]
    st.metric("📈 最大误差", f"{np.max(valid_e):.4f}" if len(valid_e) else "N/A")
    st.metric("📊 平均误差", f"{np.mean(valid_e):.4f}" if len(valid_e) else "N/A")
    st.metric("✅ 收敛比例", f"{100*np.sum(valid_e<0.01)/len(valid_e):.1f}%" if len(valid_e) else "N/A")

# ==================== 范围分析 ====================
with st.expander("📊 范围分析"):
    st.markdown(f"**推荐范围**: `[{rec_min:.2f}, {rec_max:.2f}]` (误差<1%) | 收敛性: {conv}")
    if st.checkbox("查看不同 n 的推荐范围"):
        rd = []
        for n in [3, 5, 10, 15, 20, 25]:
            rm, rx = estimate_range(func, deriv, x0, n, 0.01)
            rd.append({"n": n, "推荐范围": f"[{max(rm,-10):.2f}, {min(rx,10):.2f}]", "宽度": f"{min(rx,10)-max(rm,-10):.2f}"})
        st.dataframe(pd.DataFrame(rd), use_container_width=True, hide_index=True)

with st.expander("💡 使用说明"):
    st.markdown("**智能识别**: `exp(-x**2)`, `exp(-x**4)`, `sin(x)`, `cos(x)`, `arctan(x)` 等自动使用解析导数\n\n**注意**: 超高斯、arctan 的解析导数仅支持 x₀=0")
