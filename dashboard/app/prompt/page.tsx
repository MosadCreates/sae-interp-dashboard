"use client";

import { useState, useCallback } from "react";
import { analyzePrompt } from "@/lib/api";
import type { PromptResponse } from "@/lib/types";
import TokenHeatmap from "@/components/TokenHeatmap";
import LoadingSkeleton from "@/components/LoadingSkeleton";

export default function PromptPage() {
  const [prompt, setPrompt] = useState(
    "The transformer architecture revolutionised natural language processing by introducing self-attention mechanisms."
  );
  const [result, setResult] = useState<PromptResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);

  const analyze = useCallback(async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setSelectedFeature(null);
    setSelectedToken(null);
    try {
      const data = await analyzePrompt(prompt, 10);
      setResult(data);
      if (data.top_features.length > 0) {
        setSelectedFeature(data.top_features[0].feature_id);
      }
    } catch {
      setResult(null);
    }
    setLoading(false);
  }, [prompt]);

  const getTokenActivations = useCallback(
    (featureId: number | null): number[] => {
      if (!result || featureId === null) return [];
      return result.tokens.map((t) => {
        const act = t.activations.find((a) => a.feature_id === featureId);
        return act?.value ?? 0;
      });
    },
    [result]
  );

  const selectedTokenFeatures = useCallback((): {
    token: string;
    features: { feature_id: number; value: number }[];
  } | null => {
    if (!result || selectedToken === null) return null;
    const tok = result.tokens[selectedToken];
    if (!tok) return null;
    return {
      token: tok.token,
      features: tok.activations.slice(0, 10),
    };
  }, [result, selectedToken]);

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white">Prompt Analyser</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          See which SAE features activate on each token of your prompt.
        </p>
      </header>

      {/* Input */}
      <div className="flex gap-3 mb-8">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          className="flex-1 px-4 py-3 bg-card border border-border rounded text-white placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary resize-none text-sm"
        />
        <button
          onClick={analyze}
          disabled={loading || !prompt.trim()}
          className="px-6 py-2 bg-primary text-primary-foreground rounded font-medium text-sm self-end disabled:opacity-40 hover:opacity-90 transition-opacity"
        >
          {loading ? "Analysing..." : "Analyse"}
        </button>
      </div>

      {/* Results */}
      {loading && (
        <div className="space-y-4">
          <div className="h-8 w-48 bg-muted rounded animate-pulse" />
          <div className="h-16 bg-muted rounded animate-pulse" />
        </div>
      )}

      {result && !loading && (
        <>
          {/* Feature selector */}
          <section className="mb-8">
            <h2 className="text-sm font-medium text-white mb-3">
              Active Features
            </h2>
            <div className="flex flex-wrap gap-2">
              {result.top_features.map((f) => (
                <button
                  key={f.feature_id}
                  onClick={() => {
                    setSelectedFeature(f.feature_id);
                    setSelectedToken(null);
                  }}
                  className={`px-3 py-1.5 rounded text-xs font-medium border transition-colors ${
                    selectedFeature === f.feature_id
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-card text-muted-foreground border-border hover:border-primary hover:text-foreground"
                  }`}
                >
                  Feature {f.feature_id}
                  <span className="ml-1.5 opacity-60">
                    ({f.max_activation.toFixed(1)})
                  </span>
                </button>
              ))}
            </div>
          </section>

          {/* Token Heatmap */}
          <section className="mb-8">
            <h2 className="text-sm font-medium text-white mb-3">
              Token Heatmap
              {selectedFeature !== null && (
                <span className="text-muted-foreground ml-2">
                  (Feature {selectedFeature})
                </span>
              )}
            </h2>
            <div className="bg-card rounded border border-border p-4">
              <TokenHeatmap
                tokens={result.tokens.map((t) => t.token)}
                activations={getTokenActivations(selectedFeature)}
                selectedTokenIndex={selectedToken}
                onTokenClick={(idx) =>
                  setSelectedToken(selectedToken === idx ? null : idx)
                }
              />
            </div>
          </section>

          {/* Selected token breakdown */}
          {selectedTokenFeatures() && (
            <section className="bg-card rounded border border-border p-4">
              <h3 className="text-sm font-medium text-white mb-3">
                Token: &ldquo;{selectedTokenFeatures()?.token}&rdquo;
              </h3>
              <div className="flex flex-wrap gap-2">
                {selectedTokenFeatures()?.features.map((f) => (
                  <button
                    key={f.feature_id}
                    onClick={() => {
                      setSelectedFeature(f.feature_id);
                      setSelectedToken(null);
                    }}
                    className="px-3 py-1.5 bg-background rounded border border-border text-xs hover:border-primary transition-colors"
                  >
                    Feature {f.feature_id}
                    <span className="text-primary ml-1">{f.value.toFixed(2)}</span>
                  </button>
                ))}
              </div>
            </section>
          )}
        </>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div className="text-center py-16 text-muted-foreground text-sm">
          Enter a prompt and click Analyse to see feature activations.
        </div>
      )}
    </main>
  );
}
