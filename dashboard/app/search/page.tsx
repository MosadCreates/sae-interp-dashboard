"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { searchFeatures } from "@/lib/api";
import type { SearchResult } from "@/lib/types";
import LoadingSkeleton from "@/components/LoadingSkeleton";

const SUGGESTIONS = ["months", "code", "names", "punctuation", "numbers", "verbs"];

function SearchContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [query, setQuery] = useState(searchParams.get("q") ?? "");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const doSearch = async (q: string) => {
    if (!q.trim()) {
      setResults([]);
      setSearched(false);
      return;
    }
    setLoading(true);
    setSearched(true);
    try {
      const data = await searchFeatures(q);
      setResults(data.results ?? []);
    } catch {
      setResults([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    const q = searchParams.get("q");
    if (q) {
      setQuery(q);
      doSearch(q);
    }
  }, [searchParams]);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (query.trim()) {
        router.replace(`/search?q=${encodeURIComponent(query)}`, { scroll: false });
        doSearch(query);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  return (
    <>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="e.g. months, code, names, punctuation..."
        className="w-full px-4 py-3 bg-card border border-border rounded text-white placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary text-base"
      />

      {loading && <LoadingSkeleton />}

      {!loading && results.length > 0 && (
        <div className="space-y-3 mt-6">
          <p className="text-sm text-muted-foreground mb-2">
            {results.length} feature{results.length !== 1 ? "s" : ""} found
          </p>
          {results.map((r) => (
            <a
              key={r.feature_id}
              href={`/features/${r.feature_id}`}
              className="block p-4 bg-card rounded border border-border hover:border-primary/50 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-primary font-mono text-sm font-medium">
                  Feature {r.feature_id}
                </span>
                <span className="text-muted-foreground text-xs">
                  {(r.frequency * 100).toFixed(1)}% frequency
                </span>
              </div>
              <p className="text-sm text-white mb-2">{r.label}</p>
              <div className="flex gap-1 flex-wrap">
                {(r.top_tokens ?? []).slice(0, 8).map((t, i) => (
                  <span
                    key={i}
                    className="px-1.5 py-0.5 bg-muted rounded text-xs text-muted-foreground"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </a>
          ))}
        </div>
      )}

      {!loading && searched && results.length === 0 && (
        <div className="text-center py-16">
          <p className="text-muted-foreground">
            No features found for &ldquo;{query}&rdquo;.
          </p>
          <p className="text-muted-foreground text-sm mt-2">
            Try: {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => setQuery(s)}
                className="text-primary hover:underline mx-1"
              >
                {s}
              </button>
            ))}
          </p>
        </div>
      )}

      {!loading && !searched && (
        <div className="text-center py-16">
          <p className="text-muted-foreground mb-4">
            Try searching for:
          </p>
          <div className="flex gap-3 justify-center flex-wrap">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => setQuery(s)}
                className="px-4 py-2 bg-card border border-border rounded text-sm hover:border-primary transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

export default function SearchPage() {
  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white">Search Features</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          Search for features by their top activating tokens.
        </p>
      </header>

      <div className="mb-8">
        <Suspense fallback={<LoadingSkeleton />}>
          <SearchContent />
        </Suspense>
      </div>
    </main>
  );
}
