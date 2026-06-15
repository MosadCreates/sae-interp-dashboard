import { create } from "zustand";

interface SortConfig {
  sortBy: "frequency" | "max_activation" | "feature_id";
  page: number;
  pageSize: number;
}

interface AppStore {
  sort: SortConfig;
  setSort: (sort: Partial<SortConfig>) => void;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  selectedFeatureId: number | null;
  setSelectedFeatureId: (id: number | null) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  sort: { sortBy: "frequency", page: 1, pageSize: 20 },
  setSort: (partial) =>
    set((state) => ({ sort: { ...state.sort, ...partial } })),
  searchQuery: "",
  setSearchQuery: (q) => set({ searchQuery: q }),
  selectedFeatureId: null,
  setSelectedFeatureId: (id) => set({ selectedFeatureId: id }),
}));
