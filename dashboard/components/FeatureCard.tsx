"use client";

import type { FeatureSummary } from "@/lib/types";

interface FeatureCardProps {
  feature: FeatureSummary;
  href?: string;
}

export default function FeatureCard({
  feature,
  href,
}: FeatureCardProps) {
  const content = (
    <div className="p-4 bg-card rounded border border-border hover:border-primary/50 transition-colors cursor-pointer">
      <div className="flex items-center justify-between mb-2">
        <span className="text-primary font-mono font-medium">
          Feature {feature.feature_id}
        </span>
        <span className="text-muted-foreground text-xs">
          {(feature.frequency * 100).toFixed(1)}% freq
        </span>
      </div>
      <p className="text-sm text-white mb-2 truncate">{feature.label}</p>
      <div className="flex gap-1 flex-wrap">
        {(feature.top_tokens ?? []).slice(0, 5).map((t, i) => (
          <span
            key={i}
            className="px-1.5 py-0.5 bg-muted rounded text-xs text-muted-foreground"
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );

  if (href) {
    return <a href={href}>{content}</a>;
  }
  return content;
}
