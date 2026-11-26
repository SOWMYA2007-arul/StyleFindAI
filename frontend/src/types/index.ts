export interface Product {
  id: number;
  productCode: string;
  name: string;
  category: string;
  categoryName: string;
  subcategory: string | null;
  gender: string | null;
  price: number | null;
  description: string | null;
  colour: string;
  pattern: string;
  sleeveType: string;
  neckline: string;
  imageUrl: string | null;
  sourceDataset: string;
  licence: string;
  isActive: boolean;
  hasEmbedding?: boolean;
  images?: ProductImage[];
}

export interface ProductImage {
  url: string;
  isPrimary: boolean;
  width: number | null;
  height: number | null;
}

export interface SearchFilters {
  category?: string[];
  colour?: string[];
  pattern?: string[];
  sleeve_type?: string[];
  neckline?: string[];
  min_price?: number;
  max_price?: number;
}

export interface IndexingFailure {
  id: number;
  productCode: string;
  name: string;
  status: "pending" | "failed";
  error: string | null;
  updatedAt: string;
}

export interface Paged<T> {
  items: T[];
  page: number;
  limit: number;
  total: number;
  pages: number;
}

export interface Category {
  id: number;
  slug: string;
  name: string;
  productCount: number;
  coverImageUrl: string | null;
}

export type MatchType = "exact" | "similar" | "related";

export interface Scores {
  final: number | null;
  visual: number | null;
  visual_cosine: number | null;
  colour: number | null;
  pattern: number | null;
  attribute: number | null;
}

export interface SearchResult {
  id: number;
  productCode: string;
  name: string;
  category: string;
  categoryName: string;
  subcategory: string | null;
  colour: string;
  pattern: string;
  sleeveType: string;
  neckline: string;
  price: number | null;
  imageUrl: string;
  matchType: MatchType;
  scores: Scores;
  reasons: string[];
}

export interface PredictedAttribute {
  value: string;
  confidence: number | null;
  source: string;
}

export interface SearchResponse {
  searchId: string | null;
  model: { key: string; version: string; name: string; dims: number };
  query: {
    width: number;
    height: number;
    quality_warnings: string[];
    garment_isolation: { method: string; reason?: string; label?: string };
    predicted_attributes: Record<string, PredictedAttribute>;
    palette: { lab: number[]; weight: number; name: string }[];
    text: string | null;
    requested_attributes: Record<string, string>;
    composition: { method: string; note?: string; source_text?: string; target_text?: string } | null;
  };
  candidatesConsidered: number;
  /** set when the shopper typed a colour: only products in that colour family (and garment type) are shown */
  colourRequirement?: { colour: string; category: string | null; checked: number; kept: number; removed: number } | null;
  retrieval?: { mode: string; returned: number; requested: number; filtered: boolean } | null;
  timingsMs: Record<string, number>;
  count: number;
  results: SearchResult[];
}

export interface Facets {
  attributes: Record<string, { value: string; count: number }[]>;
  price: { min: number; max: number } | null;
}

export interface AdminStats {
  products: number;
  activeProducts: number;
  images: number;
  searches: number;
  searchesLast7Days: number;
  avgSearchLatencyMs: number | null;
  primaryModel: string;
  missingPrimaryEmbeddings: number;
  failedIndexing?: number;
  byCategory: { slug: string; name: string; count: number }[];
  embeddings: { model: string; version: string; dims: number; count: number }[];
  sources: { source: string; count: number }[];
}

export interface ModelInfo {
  key: string;
  name: string;
  dims: number;
  version: string;
  supports_text: boolean;
  catalogue_embeddings: number;
}
