import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, errorMessage } from "../services/api";
import type { SearchFilters, SearchResponse } from "../types";

interface SearchState {
  file: File | null;
  previewUrl: string | null;
  text: string;
  crop: boolean;
  result: SearchResponse | null;
  filters: SearchFilters;
  loading: boolean;
  error: string | null;
  setFile: (f: File | null) => void;
  setText: (t: string) => void;
  setCrop: (c: boolean) => void;
  runSearch: (filters?: SearchFilters) => Promise<boolean>;
}

const Ctx = createContext<SearchState | null>(null);

/** Keeps the uploaded photo and the latest AI results while the user moves between pages. */
export function SearchProvider({ children }: { children: ReactNode }) {
  const [file, setFileState] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [crop, setCrop] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [filters, setFilters] = useState<SearchFilters>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setFile = useCallback((f: File | null) => {
    setFileState(f);
    setError(null);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return f ? URL.createObjectURL(f) : null;
    });
    if (!f) setResult(null);
    setFilters({});
  }, []);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /** Runs the real AI search. `nextFilters` (category, colour, price ...) are applied by the server
   *  inside the vector search, so results come from the whole catalogue, not just the last page. */
  const runSearch = useCallback(async (nextFilters?: SearchFilters) => {
    if (!file) {
      setError("Choose a photo first.");
      return false;
    }
    const f = nextFilters ?? filters;
    setLoading(true);
    setError(null);
    try {
      setResult(await api.searchImage(file, { text: text.trim() || undefined, crop, topK: 40, filters: f }));
      setFilters(f);
      return true;
    } catch (e) {
      setError(errorMessage(e));
      return false;
    } finally {
      setLoading(false);
    }
  }, [file, text, crop, filters]);

  return (
    <Ctx.Provider value={{ file, previewUrl, text, crop, result, filters, loading, error, setFile, setText, setCrop, runSearch }}>
      {children}
    </Ctx.Provider>
  );
}

export function useSearch() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useSearch must be used inside SearchProvider");
  return c;
}
