"use client";

import { useParams } from "next/navigation";
import { useEffect, useState, useCallback } from "react";
import { getFeature, analyzePrompt } from "@/lib/api";
import type { FeatureDetail } from "@/lib/types";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import ActivationHistogram from "@/components/ActivationHistogram";
import ActivationBadge from "@/components/ActivationBadge";
import TokenHeatmap from "@/components/TokenHeatmap";

export default function FeatureDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<FeatureDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [testPrompt, setTestPrompt] = useState("");
  const [testResults, setTestResults] = useState<{ tokens: string[]; activations: number[] } | null>(null);
  const [testLoading, setTestLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    getFeature(Number(id))
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [id]);

  const handleTestFeature = useCallback(async () => {
    if (!testPrompt.trim() || testLoading) return;
    setTestLoading(true);
    setTestResults(null);
    try {
      const res = await analyzePrompt(testPrompt, 50);
      const tokens = res.tokens.map((t) => t.token);
      const activations = res.tokens.map((t) => {
        const act = t.activations.find((a) => a.feature_id === Number(id));
        return act?.value ?? 0;
      });
      setTestResults({ tokens, activations });
    } catch {
      setTestResults(null);
    } finally {
      setTestLoading(false);
    }
  }, [testPrompt, id, testLoading]);

  if (loading) {
    return (
      <main className="min-h-screen p-8 max-w-6xl mx-auto">
        <LoadingSkeleton variant="detail" />
      </main>
    );
  }

  if (!data) {
    return (
      <main className="min-h-screen p-8 max-w-6xl mx-auto">
        <p className="text-muted-foreground">Feature not found.</p>
        <a href="/" className="text-primary text-sm mt-4 inline-block">
          &larr; Back to features
        </a>
      </main>
    );
  }

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto">
      {/* Header */}
      <header className="mb-10">
        <a
          href="/"
          className="text-sm text-muted-foreground hover:text-primary transition-colors mb-4 inline-block"
        >
          &larr; Back to features
        </a>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">
              Feature {data.feature_id}
            </h1>
            <p className="text-lg text-primary mt-1">{data.label}</p>
          </div>
          <div className="text-right text-sm text-muted-foreground">
            <div>Frequency: {(data.frequency * 100).toFixed(2)}%</div>
            <div>Max Activation: {data.max_activation.toFixed(3)}</div>
            <div>Decoder Norm: {data.decoder_norm.toFixed(3)}</div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main column — Top Contexts */}
        <div className="lg:col-span-2 space-y-8">
          <section>
            <h2 className="text-xl font-semibold text-white mb-4">
              Top Activating Contexts
            </h2>
            <div className="space-y-2">
              {(data.top_contexts ?? []).slice(0, 30).map((ctx, i) => (
                <div
                  key={i}
                  className="flex items-start gap-3 p-3 bg-card rounded border border-border"
                >
                  <span className="text-muted-foreground text-xs w-5 mt-1 shrink-0">
                    {i + 1}
                  </span>
                  <div className="flex-1 flex flex-wrap gap-0.5 text-sm leading-relaxed">
                    {ctx.tokens.map((tok, j) => {
                      const tokenAct = j === Math.floor(ctx.tokens.length / 2) ? ctx.activation : 0;
                      return (
                        <span
                          key={j}
                          className={`px-0.5 rounded ${
                            tokenAct > 0
                              ? "bg-primary/20 text-primary font-medium"
                              : "text-muted-foreground"
                          }`}
                        >
                          {tok.replace(/\n/g, "\\n")}
                        </span>
                      );
                    })}
                  </div>
                  <span className="font-mono text-xs text-primary w-16 text-right shrink-0 mt-1">
                    {ctx.activation.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* Test This Feature */}
          <section>
            <h2 className="text-xl font-semibold text-white mb-4">
              Test This Feature
            </h2>
            <p className="text-sm text-muted-foreground mb-3">
              Enter a prompt to see which tokens activate Feature {data.feature_id}:
            </p>
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={testPrompt}
                onChange={(e) => setTestPrompt(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleTestFeature()}
                placeholder="Type a prompt here..."
                className="flex-1 px-3 py-2 bg-background border border-border rounded text-sm focus:outline-none focus:border-primary"
              />
              <button
                onClick={handleTestFeature}
                disabled={testLoading || !testPrompt.trim()}
                className="px-4 py-2 bg-primary text-primary-foreground rounded text-sm font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
              >
                {testLoading ? "Analysing..." : "Analyse"}
              </button>
            </div>
            {testResults && (
              <div className="bg-card rounded border border-border p-4">
                <TokenHeatmap
                  tokens={testResults.tokens}
                  activations={testResults.activations}
                />
              </div>
            )}
          </section>
        </div>

        {/* Sidebar */}
        <div className="space-y-8">
          {/* Histogram */}
          <section className="bg-card rounded border border-border p-4">
            <ActivationHistogram
              data={data.histogram ?? []}
              title="Activation Distribution"
            />
          </section>

          {/* Similar Features */}
          <section className="bg-card rounded border border-border p-4">
            <h3 className="text-sm font-medium text-white mb-3">
              Similar Features
            </h3>
            <div className="space-y-2">
              {(data.similar_features ?? []).length === 0 ? (
                <p className="text-xs text-muted-foreground">No similar features found.</p>
              ) : (
                (data.similar_features ?? []).slice(0, 10).map((s) => (
                  <a
                    key={s.feature_id}
                    href={`/features/${s.feature_id}`}
                    className="flex items-center justify-between px-3 py-2 bg-background rounded border border-border hover:border-primary/50 transition-colors text-sm"
                  >
                    <span className="text-primary font-mono text-xs">
                      Feature {s.feature_id}
                    </span>
                    <span className="text-muted-foreground text-xs">
                      {s.similarity.toFixed(3)}
                    </span>
                  </a>
                ))
              )}
            </div>
          </section>

          {/* Top tokens */}
          <section className="bg-card rounded border border-border p-4">
            <h3 className="text-sm font-medium text-white mb-3">
              Top Activating Tokens
            </h3>
            <div className="flex flex-wrap gap-1">
              {(data.top_contexts ?? []).slice(0, 20).map((ctx, i) => {
                const mid = Math.floor(ctx.tokens.length / 2);
                const midToken = ctx.tokens[mid] ?? "?";
                return (
                  <ActivationBadge
                    key={i}
                    token={midToken}
                    activation={ctx.activation}
                  />
                );
              })}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
