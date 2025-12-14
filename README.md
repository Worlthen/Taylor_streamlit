# 🧮 泰勒展开可视化 (Taylor Series Visualizer)

一个交互式的泰勒级数可视化工具，帮助你直观理解泰勒展开如何逼近各种数学函数。

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://worlthen-taylor-streamlit-taylor-streamlit-xxxxxx.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **在线预览**: (https://taylorapp-wvlre9rqezva4umqopdjlb.streamlit.app/) 

## ✨ 功能特性

- **10+ 预设函数**: sin, cos, exp, 高斯函数, 超高斯函数, arctan, 双曲函数等
- **自定义函数**: 输入任意函数表达式进行泰勒展开
- **智能模式识别**: 自动识别常见函数并使用解析导数（更精确）
- **推荐范围计算**: 根据展开项数自动计算有效拟合范围
- **暗色/亮色主题**: 自动适配系统主题
- **实时交互**: 调整参数立即看到结果变化

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行应用

```bash
streamlit run taylor_streamlit.py
```

然后在浏览器中打开 `http://localhost:8501`

## 📖 使用说明

### 预设函数

| 函数 | 收敛性 | 特点 |
|------|--------|------|
| sin(x), cos(x) | 全局 | 周期函数 |
| exp(x) | 全局 | 指数增长 |
| exp(-x²) 高斯 | 全局 | 使用 Hermite 多项式 |
| exp(-x⁴) 超高斯 | 全局 | 仅 x₀=0 处解析 |
| arctan(x) | \|x\|≤1 | 仅 x₀=0 处解析 |
| ln(1+x) | \|x\|<1 | 收敛域有限 |
| sqrt(1+x) | \|x\|<1 | 广义二项式展开 |

### 自定义函数

支持的语法：
- 基本运算: `+`, `-`, `*`, `/`, `**` (幂)
- 函数: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`
- 反三角: `arcsin`, `arccos`, `arctan`
- 双曲: `sinh`, `cosh`, `tanh`
- 常数: `pi`, `e`

示例：
```
exp(-x**2)     # 自动识别为高斯函数
x**2 * sin(x)  # 使用数值微分
1/(1 + x**2)   # 洛伦兹函数
```

### 智能功能

- **函数识别**: 输入 `exp(-x**4)` 会自动识别并使用解析导数
- **推荐范围**: 显示误差 <1% 的有效范围
- **范围分析**: 查看不同展开项数对应的推荐范围

## 📁 文件结构

```
Taylor_streamlit/
├── taylor_streamlit.py  # 主应用代码
├── requirements.txt     # 依赖列表
└── README.md           # 说明文档
```

## 🛠️ 技术栈

- **Streamlit**: Web 应用框架
- **Plotly**: 交互式图表
- **NumPy**: 数值计算
- **Pandas**: 数据展示

## 📝 License

MIT License
