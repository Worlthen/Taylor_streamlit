import { useState, useMemo } from "react";
import { PlotArea } from "./PlotArea";
import "./App.css";

// 计算阶乘
function factorial(n: number): number {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

// 计算 sin(x) 的泰勒展开
function sinTaylor(x: number, x0: number, terms: number): number {
  let result = 0;
  const sinVals = [Math.sin(x0), Math.cos(x0), -Math.sin(x0), -Math.cos(x0)];

  for (let n = 0; n < terms; n++) {
    const derivative = sinVals[n % 4];
    result += derivative * Math.pow(x - x0, n) / factorial(n);
  }
  return result;
}

function App() {
  const [terms, setTerms] = useState(5);  // 泰勒展开项数
  const [x0, setX0] = useState(0);         // 展开点
  const [mode, setMode] = useState<"teaching" | "research">("teaching");

  // 生成数据
  const data = useMemo(() => {
    const xValues: number[] = [];
    const originalValues: number[] = [];
    const taylorValues: number[] = [];
    const errorValues: number[] = [];

    // 生成 x 范围: -2π 到 2π
    for (let i = -200; i <= 200; i++) {
      const x = (i / 100) * Math.PI;
      xValues.push(parseFloat(x.toFixed(3)));

      const orig = Math.sin(x);
      const taylor = sinTaylor(x, x0, terms);

      originalValues.push(parseFloat(orig.toFixed(6)));
      taylorValues.push(parseFloat(taylor.toFixed(6)));
      errorValues.push(parseFloat(Math.abs(orig - taylor).toFixed(6)));
    }

    return {
      x: xValues,
      original: originalValues,
      taylor: taylorValues,
      error: errorValues,
      x0: x0
    };
  }, [terms, x0]);

  return (
    <div className="app-container">
      <header className="header">
        <h1>🧮 泰勒展开可视化 - sin(x)</h1>
        <p className="subtitle">探索泰勒级数如何逼近正弦函数</p>
      </header>

      <div className="controls">
        <div className="control-group">
          <label>
            <span className="label-text">展开项数 (n):</span>
            <input
              type="range"
              min="1"
              max="20"
              value={terms}
              onChange={(e) => setTerms(parseInt(e.target.value))}
            />
            <span className="value-display">{terms}</span>
          </label>
        </div>

        <div className="control-group">
          <label>
            <span className="label-text">展开点 (x₀):</span>
            <input
              type="range"
              min="-3.14"
              max="3.14"
              step="0.1"
              value={x0}
              onChange={(e) => setX0(parseFloat(e.target.value))}
            />
            <span className="value-display">{x0.toFixed(2)}</span>
          </label>
        </div>

        <div className="control-group">
          <label>
            <span className="label-text">模式:</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as "teaching" | "research")}
            >
              <option value="teaching">教学模式</option>
              <option value="research">科研模式 (显示误差)</option>
            </select>
          </label>
        </div>
      </div>

      <div className="chart-container">
        <PlotArea data={data} mode={mode} />
      </div>

      <div className="formula-display">
        <h3>泰勒公式</h3>
        <p className="formula">
          sin(x) ≈ sin(x₀) + cos(x₀)(x-x₀) - sin(x₀)(x-x₀)²/2! - cos(x₀)(x-x₀)³/3! + ...
        </p>
        <p className="info">
          当 x₀ = 0 时, 即为麦克劳林级数: sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
        </p>
      </div>

      <div className="legend">
        <div className="legend-item">
          <span className="color-box original"></span>
          <span>原函数 f(x) = sin(x)</span>
        </div>
        <div className="legend-item">
          <span className="color-box taylor"></span>
          <span>泰勒近似 T<sub>n</sub>(x)</span>
        </div>
      </div>
    </div>
  );
}

export default App;
