"use client";

import { useEffect, useState, useCallback } from "react";
import { useAppStore } from "@/lib/store";
import { getFeatures } from "@/lib/api";
import type { FeatureSummary } from "@/lib/types";
import LoadingSkeleton from "@/components/LoadingSkeleton";

type SortKey = "frequency" | "max_activation" | "feature_id";

export default function Home() {
  const { sort, setSort, searchQuery, setSearchQuery } = useAppStore();
  const [features, setFeatures] = useState<FeatureSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  const totalPages = Math.max(1, Math.ceil(total / sort.pageSize));

  const loadFeatures = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getFeatures(sort.page, sort.pageSize, sort.sortBy);
      setFeatures(data.features ?? []);
      setTotal(data.total ?? 0);
    } catch {
      setFeatures([]);
    }
    setLoading(false);
  }, [sort.page, sort.pageSize, sort.sortBy]);

  useEffect(() => {
    loadFeatures();
  }, [loadFeatures]);

  const handleSort = (key: SortKey) => {
    setSort({ sortBy: key, page: 1 });
  };

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white">
          Feature Explorer
        </h1>
        <p className="text-muted-foreground mt-1 text-sm">
          {total.toLocaleString()} features &middot; Sparse Autoencoder on GPT-2 Small
        </p>
      </header>

      {/* Search + Sort controls */}
      <div className="flex items-center gap-4 mb-6 flex-wrap">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && searchQuery.trim()) {
              window.location.href = `/search?q=${encodeURIComponent(searchQuery)}`;
            }
          }}
          placeholder="Search features..."
          className="flex-1 min-w-[200px] px-3 py-2 bg-card border border-border rounded text-white placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary text-sm"
        />
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Sort:</span>
          {(["frequency", "max_activation", "feature_id"] as SortKey[]).map(
            (key) => (
              <button
                key={key}
                onClick={() => handleSort(key)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  sort.sortBy === key
                    ? "bg-primary text-primary-foreground"
                    : "bg-card text-muted-foreground hover:text-foreground border border-border"
                }`}
              >
                {key === "feature_id" ? "ID" : key === "frequency" ? "Frequency" : "Max Act."}
              </button>
            )
          )}
        </div>
      </div>

      {/* Table */}
      {loading ? (
        <LoadingSkeleton variant="table" />
      ) : features.length === 0 ? (
        <div className="text-center py-16 text-muted-foreground">
          No features found. Train an SAE first.
        </div>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-muted-foreground text-xs uppercase tracking-wider">
                  <th className="text-left py-3 px-2 font-medium">ID</th>
                  <th className="text-left py-3 px-2 font-medium">Label</th>
                  <th className="text-left py-3 px-2 font-medium">Top Tokens</th>
                  <th className="text-right py-3 px-2 font-medium">Frequency</th>
                  <th className="text-right py-3 px-2 font-medium">Max Activation</th>
                </tr>
              </thead>
              <tbody>
                {features.map((f) => (
                  <tr
                    key={f.feature_id}
                    className="border-b border-border hover:bg-muted/30 cursor-pointer transition-colors"
                    onClick={() =>
                      (window.location.href = `/features/${f.feature_id}`)
                    }
                  >
                    <td className="py-3 px-2 font-mono text-primary text-xs">
                      {f.feature_id}
                    </td>
                    <td className="py-3 px-2 text-white max-w-[200px] truncate">
                      {f.label}
                    </td>
                    <td className="py-3 px-2">
                      <div className="flex gap-1 flex-wrap">
                        {(f.top_tokens ?? []).slice(0, 4).map((t, i) => (
                          <span
                            key={i}
                            className="px-1.5 py-0.5 bg-muted rounded text-xs text-muted-foreground"
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="py-3 px-2 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary rounded-full"
                            style={{
                              width: `${Math.min(f.frequency * 1000, 100)}%`,
                            }}
                          />
                        </div>
                        <span className="text-muted-foreground text-xs w-10 text-right">
                          {(f.frequency * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-2 text-right font-mono text-muted-foreground text-xs">
                      {f.max_activation.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-6 text-sm">
            <span className="text-muted-foreground">
              Page {sort.page} of {totalPages} ({total.toLocaleString()} features)
            </span>
            <div className="flex gap-2">
              <button
                disabled={sort.page <= 1}
                onClick={() => setSort({ page: sort.page - 1 })}
                className="px-3 py-1.5 bg-card border border-border rounded text-xs disabled:opacity-30 hover:border-primary transition-colors"
              >
                Previous
              </button>
              <button
                disabled={sort.page >= totalPages}
                onClick={() => setSort({ page: sort.page + 1 })}
                className="px-3 py-1.5 bg-card border border-border rounded text-xs disabled:opacity-30 hover:border-primary transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </main>
  );
}
