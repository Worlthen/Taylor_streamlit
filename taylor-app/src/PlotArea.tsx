import React from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ReferenceLine,
    Area
} from "recharts";

type PlotAreaProps = {
    data: {
        x: number[];
        original: number[];
        taylor: number[];
        error?: number[];
        x0: number;
    };
    mode: "teaching" | "research";
};

export const PlotArea: React.FC<PlotAreaProps> = ({ data, mode }) => {
    const chartData = data.x.map((xVal, i) => ({
        x: xVal,
        original: data.original[i],
        taylor: data.taylor[i],
        error: data.error ? data.error[i] : undefined
    }));

    return (
        <div style={{ width: "100%", height: 420 }}>
            <LineChart
                width={900}
                height={420}
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" />
                <YAxis />
                <Tooltip />

                {/* 原函数 */}
                <Line
                    type="monotone"
                    dataKey="original"
                    stroke="#000"
                    strokeWidth={2.5}
                    dot={false}
                    name="f(x)"
                />

                {/* Taylor 近似 */}
                <Line
                    type="monotone"
                    dataKey="taylor"
                    stroke="#ff6b6b"
                    strokeWidth={2}
                    dot={false}
                    name="Tₙ(x)"
                />

                {/* 展开点 x₀ */}
                <ReferenceLine
                    x={data.x0}
                    stroke="#555"
                    strokeDasharray="4 4"
                    label={{ value: "x₀", position: "top" }}
                />

                {/* 误差区域（科研模式） */}
                {mode === "research" && data.error && (
                    <Area
                        type="monotone"
                        dataKey="error"
                        stroke="none"
                        fill="rgba(255, 107, 107, 0.25)"
                        name="|f - Tₙ|"
                    />
                )}
            </LineChart>
        </div>
    );
};
