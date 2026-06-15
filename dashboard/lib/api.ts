import type {
  FeatureSummary,
  FeatureDetail,
  FeatureListResponse,
  PromptResponse,
  SearchResult,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return res.json();
}

export async function getFeatures(
  page = 1,
  pageSize = 20,
  sortBy = "frequency"
): Promise<FeatureListResponse> {
  return fetchJson<FeatureListResponse>(
    `${BASE_URL}/api/features?page=${page}&page_size=${pageSize}&sort_by=${sortBy}`
  );
}

export async function getFeature(
  featureId: number
): Promise<FeatureDetail> {
  return fetchJson<FeatureDetail>(`${BASE_URL}/api/features/${featureId}`);
}

export async function searchFeatures(
  query: string
): Promise<{ results: SearchResult[] }> {
  return fetchJson<{ results: SearchResult[] }>(
    `${BASE_URL}/api/search?q=${encodeURIComponent(query)}`
  );
}

export async function analyzePrompt(
  prompt: string,
  topKFeatures = 10
): Promise<PromptResponse> {
  const res = await fetch(`${BASE_URL}/api/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, top_k_features: topKFeatures }),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return res.json();
}

export async function getSimilarFeatures(
  featureId: number
): Promise<{ feature_id: number; similar_features: { feature_id: number; cosine_similarity: number }[] }> {
  return fetchJson(`${BASE_URL}/api/features/${featureId}/similar`);
}

export async function getStats(): Promise<Record<string, unknown>> {
  return fetchJson(`${BASE_URL}/api/stats`);
}
