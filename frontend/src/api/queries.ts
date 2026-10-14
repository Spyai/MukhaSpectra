import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { BUILTIN_SITE_CATALOGUE } from '../lib/sites';
import { useApi } from './ApiContext';
import type { DemoKind, ScreeningDetail, UploadRequest } from './types';

export const queryKeys = {
  sites: ['sites'] as const,
  version: ['version'] as const,
  screenings: ['screenings'] as const,
  screening: (id: string) => ['screening', id] as const,
  result: (id: string, version: number) => ['screening', id, 'result', version] as const,
  artifactJson: (id: string, key: string, version?: number) =>
    ['artifact-json', id, key, version ?? 'latest'] as const,
};

/** Site catalogue: the live one when reachable, the built-in copy otherwise. */
export function useSiteCatalogue() {
  const api = useApi();
  const q = useQuery({
    queryKey: queryKeys.sites,
    queryFn: () => api.getSites(),
    staleTime: Infinity,
    retry: 1,
  });
  return q.data ?? BUILTIN_SITE_CATALOGUE;
}

export function useVersionInfo() {
  const api = useApi();
  return useQuery({ queryKey: queryKeys.version, queryFn: () => api.getVersion(), staleTime: 300_000, retry: 0 });
}

export function useScreenings() {
  const api = useApi();
  return useQuery({ queryKey: queryKeys.screenings, queryFn: () => api.listScreenings(), retry: 1 });
}

export function useScreening(id: string | undefined) {
  const api = useApi();
  return useQuery({
    queryKey: queryKeys.screening(id ?? ''),
    queryFn: () => api.getScreening(id!),
    enabled: Boolean(id),
    retry: 1,
  });
}

export function useResultVersion(id: string | undefined, version: number | undefined, enabled: boolean) {
  const api = useApi();
  return useQuery({
    queryKey: queryKeys.result(id ?? '', version ?? 0),
    queryFn: () => api.getResultVersion(id!, version!),
    enabled: enabled && Boolean(id) && version !== undefined,
    staleTime: Infinity,
  });
}

export function useArtifactJson<T>(id: string, key: string | null | undefined, version?: number) {
  const api = useApi();
  return useQuery({
    queryKey: queryKeys.artifactJson(id, key ?? '', version),
    queryFn: () => api.getArtifactJson<T>(id, key!, version),
    enabled: Boolean(key),
    staleTime: Infinity,
  });
}

function useStoreDetail() {
  const qc = useQueryClient();
  return (detail: ScreeningDetail) => {
    qc.setQueryData(queryKeys.screening(detail.screening_id), detail);
    qc.setQueryData(queryKeys.result(detail.screening_id, detail.latest.result_version), detail.latest);
    void qc.invalidateQueries({ queryKey: queryKeys.screenings });
  };
}

export function useRunDemo() {
  const api = useApi();
  const store = useStoreDetail();
  return useMutation({ mutationFn: (kind: DemoKind) => api.runDemo(kind), onSuccess: store });
}

export function useCreateScreening() {
  const api = useApi();
  const store = useStoreDetail();
  return useMutation({ mutationFn: (req: UploadRequest) => api.createScreening(req), onSuccess: store });
}

export function useReanalyze(id: string) {
  const api = useApi();
  const store = useStoreDetail();
  return useMutation({ mutationFn: () => api.reanalyze(id), onSuccess: store });
}
