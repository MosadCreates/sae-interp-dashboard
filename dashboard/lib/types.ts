export interface FeatureSummary {
  feature_id: number;
  label: string;
  frequency: number;
  max_activation: number;
  top_tokens: string[];
}

export interface FeatureDetail {
  feature_id: number;
  label: string;
  frequency: number;
  max_activation: number;
  decoder_norm: number;
  top_contexts: TokenContext[];
  histogram: HistogramBin[];
  similar_features: SimilarFeature[];
}

export interface TokenContext {
  tokens: string[];
  activation: number;
}

export interface HistogramBin {
  bin_center: number;
  count: number;
}

export interface SimilarFeature {
  feature_id: number;
  similarity: number;
}

export interface PromptResponse {
  tokens: TokenActivation[];
  top_features: TopFeature[];
  full_matrix: number[][];
  active_feature_ids: number[];
}

export interface TokenActivation {
  token: string;
  activations: FeatureActivation[];
}

export interface FeatureActivation {
  feature_id: number;
  value: number;
}

export interface TopFeature {
  feature_id: number;
  max_activation: number;
}

export interface SearchResult {
  feature_id: number;
  label: string;
  top_tokens: string[];
  frequency: number;
  score: number;
}

export interface FeatureListResponse {
  features: FeatureSummary[];
  total: number;
  page: number;
  page_size: number;
}
