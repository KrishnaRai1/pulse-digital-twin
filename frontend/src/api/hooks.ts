import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type {
  Advisory,
  AlertsResponse,
  AuditRow,
  CycleRow,
  FieldTrend,
  Job,
  LatestCard,
  Overview,
  Recommendation,
  SampleFile,
  StoredCard,
  SystemInfo,
  Telemetry,
  ViscosityCurves,
  WellDetail,
  WellProfile,
} from "./types";

export const useOverview = () => useQuery({ queryKey: ["overview"], queryFn: () => api.get<Overview>("/field/overview"), refetchInterval: 6000, placeholderData: keepPreviousData });
export const useFieldTrend = (hours = 24) => useQuery({ queryKey: ["trend", hours], queryFn: () => api.get<FieldTrend>(`/field/trend?hours=${hours}`), refetchInterval: 20000 });
export const useAlerts = () => useQuery({ queryKey: ["alerts"], queryFn: () => api.get<AlertsResponse>("/alerts?limit=30"), refetchInterval: 15000 });
export const useSystemInfo = () => useQuery({ queryKey: ["system-info"], queryFn: () => api.get<SystemInfo>("/system/info"), refetchInterval: 30000 });
export const useSamples = () => useQuery({ queryKey: ["samples"], queryFn: () => api.get<SampleFile[]>("/samples"), staleTime: Infinity });

export const useWellDetail = (id: string) =>
  useQuery({ queryKey: ["well-detail", id], queryFn: () => api.get<WellDetail>(`/wells/${id}`), refetchInterval: 8000, placeholderData: keepPreviousData });
export const useTelemetry = (id: string, hours = 24) =>
  useQuery({ queryKey: ["telemetry", id, hours], queryFn: () => api.get<Telemetry>(`/wells/${id}/telemetry?hours=${hours}&max_points=360`), refetchInterval: 8000, placeholderData: keepPreviousData });
export const useProfile = (id: string) =>
  useQuery({ queryKey: ["profile", id], queryFn: () => api.get<WellProfile>(`/wells/${id}/profile`), refetchInterval: 15000, placeholderData: keepPreviousData });
export const useViscosity = (id: string) =>
  useQuery({ queryKey: ["viscosity", id], queryFn: () => api.get<ViscosityCurves>(`/wells/${id}/viscosity`), refetchInterval: 20000, placeholderData: keepPreviousData });
export const useLatestCard = (id: string) =>
  useQuery({ queryKey: ["latest-card", id], queryFn: () => api.get<LatestCard>(`/wells/${id}/card`), refetchInterval: 7000, placeholderData: keepPreviousData });
export const useCardHistory = (id: string, source?: "live" | "upload") =>
  useQuery({
    queryKey: ["card-history", id, source],
    queryFn: () => api.get<StoredCard[]>(`/wells/${id}/cards?limit=12${source ? `&source=${source}` : ""}`),
    refetchInterval: 10000,
  });

export const useRecommendation = (id: string) =>
  useQuery({ queryKey: ["recommendation", id], queryFn: () => api.get<Recommendation>(`/wells/${id}/recommendation`), refetchInterval: 10000, placeholderData: keepPreviousData });
export const useAdvisories = (status?: string) =>
  useQuery({ queryKey: ["advisories", status ?? "all"], queryFn: () => api.get<Advisory[]>(`/advisories?limit=30${status ? `&status=${status}` : ""}`), refetchInterval: 10000 });
export const useAudit = () => useQuery({ queryKey: ["audit"], queryFn: () => api.get<AuditRow[]>("/audit?limit=25"), refetchInterval: 15000 });

export function useDecision() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, approve, actor, note }: { id: number; approve: boolean; actor: string; note?: string }) =>
      api.post<Advisory>(`/advisories/${id}/${approve ? "approve" : "reject"}`, { actor, note: note || undefined }),
    onSettled: () => {
      for (const k of ["advisories", "recommendation", "audit", "overview", "well-detail", "latest-card"]) qc.invalidateQueries({ queryKey: [k] });
    },
  });
}

export function useJob<T>(id: string | null) {
  return useQuery({
    queryKey: ["job", id],
    enabled: !!id,
    queryFn: () => api.get<Job<T>>(`/jobs/${id}`),
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "done" || s === "failed" ? false : 1200;
    },
  });
}

export function useCycleMutations(wellId: string) {
  const qc = useQueryClient();
  const refresh = () => {
    for (const k of ["well-detail", "profile", "viscosity", "overview", "recommendation", "telemetry"]) qc.invalidateQueries({ queryKey: [k] });
  };
  return {
    update: useMutation({
      mutationFn: ({ cycle, body }: { cycle: number; body: Partial<CycleRow> }) => api.put<CycleRow>(`/wells/${wellId}/cycles/${cycle}`, body),
      onSuccess: refresh,
    }),
    add: useMutation({ mutationFn: (body: Partial<CycleRow>) => api.post<CycleRow>(`/wells/${wellId}/cycles`, body), onSuccess: refresh }),
    remove: useMutation({ mutationFn: (cycle: number) => api.del(`/wells/${wellId}/cycles/${cycle}`), onSuccess: refresh }),
  };
}
