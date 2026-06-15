"use client";

import { cn } from "@/lib/utils";

interface ActivationBadgeProps {
  token: string;
  activation: number;
  className?: string;
}

export default function ActivationBadge({
  token,
  activation,
  className,
}: ActivationBadgeProps) {
  const intensity = Math.min(1, activation / 10);
  const bgOpacity = Math.max(0.05, intensity * 0.7);

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono",
        className
      )}
      style={{
        backgroundColor: `rgba(234, 88, 12, ${bgOpacity})`,
        color: intensity > 0.5 ? "#fff" : "hsl(0, 0%, 70%)",
      }}
      title={`${token}: ${activation.toFixed(3)}`}
    >
      <span>{token.replace(/\n/g, "\\n")}</span>
      <span className="opacity-60">{activation.toFixed(1)}</span>
    </span>
  );
}
