"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  CartesianGrid,
} from "recharts";

interface ActivationHistogramProps {
  data: { bin_center: number; count: number }[];
  title?: string;
}

export default function ActivationHistogram({
  data,
  title,
}: ActivationHistogramProps) {
  const chartData = useMemo(() => {
    if (data.length === 0) return [];
    const maxCount = Math.max(...data.map((d) => d.count));
    return data.map((d) => ({
      bin_center: d.bin_center,
      count: d.count,
      pct: (d.count / maxCount) * 100,
    }));
  }, [data]);

  if (chartData.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-muted-foreground text-sm">
        No data
      </div>
    );
  }

  return (
    <div>
      {title && (
        <h3 className="text-sm font-medium text-white mb-2">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(0, 0%, 15%)" />
          <XAxis
            dataKey="bin_center"
            stroke="hsl(0, 0%, 50%)"
            tick={{ fontSize: 10 }}
            tickFormatter={(v) => v.toFixed(0)}
          />
          <YAxis
            stroke="hsl(0, 0%, 50%)"
            tick={{ fontSize: 10 }}
            tickFormatter={(v) => v.toLocaleString()}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(0, 0%, 10%)",
              border: "1px solid hsl(0, 0%, 18%)",
              borderRadius: "4px",
              fontSize: "12px",
            }}
            formatter={(value: number) => [value.toLocaleString(), "Count"]}
            labelFormatter={(label) => `Activation: ${Number(label).toFixed(2)}`}
          />
          <Bar dataKey="count" fill="hsl(28, 100%, 60%)" opacity={0.8} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
