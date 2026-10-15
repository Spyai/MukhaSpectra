import type { TFunction } from 'i18next';
import { ApiError } from '../api/client';

/** Plain-language message for any error thrown by the API client. */
export function errorMessage(err: unknown, t: TFunction): string {
  if (err instanceof ApiError) {
    if (err.kind === 'network') return t('errors.network');
    if (err.kind === 'bad-response') return t('errors.badResponse');
    if (err.status === 404) return t('errors.notFound');
    return t('errors.server', { status: err.status ?? '?', detail: err.detail ?? '' }).trim();
  }
  if (err instanceof Error && err.message === 'bad-response') return t('errors.badResponse');
  return t('common.error');
}
