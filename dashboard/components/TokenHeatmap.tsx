"use client";

import { useMemo } from "react";

interface TokenHeatmapProps {
  tokens: string[];
  activations: number[];
  selectedTokenIndex?: number | null;
  onTokenClick?: (index: number) => void;
  colorScheme?: { r: number; g: number; b: number };
}

export default function TokenHeatmap({
  tokens,
  activations,
  selectedTokenIndex,
  onTokenClick,
  colorScheme,
}: TokenHeatmapProps) {
  const { r, g, b } = colorScheme ?? { r: 234, g: 88, b: 12 };
  const maxAct = useMemo(
    () => Math.max(...activations, 0.001),
    [activations]
  );

  return (
    <div className="flex flex-wrap gap-0.5">
      {tokens.map((token, i) => {
        const val = activations[i] ?? 0;
        const intensity = val / maxAct;
        const bgOpacity = Math.max(0.02, intensity * 0.85);

        let bgColor: string;
        if (intensity <= 0.01) {
          bgColor = "rgba(255,255,255,0.03)";
        } else {
          bgColor = `rgba(${r}, ${g}, ${b}, ${bgOpacity})`;
        }

        const isSelected = selectedTokenIndex === i;

        return (
          <span
            key={i}
            onClick={() => onTokenClick?.(i)}
            className={cn(
              "px-1 py-0.5 text-xs rounded cursor-default transition-all",
              isSelected ? "ring-1 ring-primary" : "",
              onTokenClick ? "cursor-pointer hover:ring-1 hover:ring-primary/50" : ""
            )}
            style={{ backgroundColor: bgColor }}
            title={`${token}: ${val.toFixed(3)}`}
          >
            {token.replace(/\n/g, "\\n")}
          </span>
        );
      })}
    </div>
  );
}

function cn(...classes: (string | boolean | undefined | null)[]) {
  return classes.filter(Boolean).join(" ");
}
