import { render } from '@testing-library/react';
import type { ReactElement } from 'react';
import { createMemoryRouter, MemoryRouter, RouterProvider } from 'react-router';
import { Providers, routes } from '../App';
import type { ApiClient } from '../api/types';
import { makeQueryClient } from '../lib/queryClient';
import { createMockClient } from '../mocks/mockClient';

export function fastMockClient(): ApiClient {
  return createMockClient({ analysisLatencyMs: 0, readLatencyMs: 0 });
}

function testQueryClient() {
  const qc = makeQueryClient();
  qc.setDefaultOptions({
    queries: { retry: false, staleTime: Infinity, refetchOnWindowFocus: false },
  });
  return qc;
}

/** Render a component inside all app providers (API, react-query, motion) and a router. */
export function renderWithProviders(ui: ReactElement, client: ApiClient = fastMockClient()) {
  return render(
    <Providers client={client} queryClient={testQueryClient()}>
      <MemoryRouter>{ui}</MemoryRouter>
    </Providers>,
  );
}

/** Render the full app at a route (lazy pages included). */
export function renderRoute(path: string, client: ApiClient = fastMockClient()) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  const utils = render(
    <Providers client={client} queryClient={testQueryClient()}>
      <RouterProvider router={router} />
    </Providers>,
  );
  return { ...utils, router };
}
