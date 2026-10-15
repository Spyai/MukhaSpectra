import { QueryClientProvider, type QueryClient } from '@tanstack/react-query';
import { makeQueryClient } from './lib/queryClient';
import { LazyMotion, MotionConfig, domAnimation } from 'motion/react';
import { lazy, Suspense, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { createBrowserRouter, RouterProvider, type RouteObject } from 'react-router';
import { ApiProvider } from './api/ApiContext';
import { createDefaultClient } from './api/client';
import type { ApiClient } from './api/types';
import { AppShell } from './components/layout/AppShell';
import HomePage from './pages/HomePage';

const UploadPage = lazy(() => import('./pages/UploadPage'));
const ResultsPage = lazy(() => import('./pages/ResultsPage'));
const ScreeningsPage = lazy(() => import('./pages/ScreeningsPage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));

function PageFallback() {
  const { t } = useTranslation();
  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6" aria-busy="true">
      <span className="sr-only">{t('common.loading')}</span>
      <div className="h-9 w-64 animate-pulse rounded-xl bg-surface-2" />
      <div className="mt-6 h-72 animate-pulse rounded-[1.25rem] bg-surface-2" />
    </div>
  );
}

const page = (node: ReactNode) => <Suspense fallback={<PageFallback />}>{node}</Suspense>;

// eslint-disable-next-line react-refresh/only-export-components
export const routes: RouteObject[] = [
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <HomePage /> },
      { path: 'upload', element: page(<UploadPage />) },
      { path: 'screenings', element: page(<ScreeningsPage />) },
      { path: 'screenings/:id', element: page(<ResultsPage />) },
      { path: '*', element: page(<NotFoundPage />) },
    ],
  },
];

/** Providers shared by the app and tests. */
export function Providers({
  client,
  queryClient,
  children,
}: {
  client: ApiClient;
  queryClient: QueryClient;
  children: ReactNode;
}) {
  return (
    <ApiProvider client={client}>
      <QueryClientProvider client={queryClient}>
        <LazyMotion features={domAnimation} strict>
          <MotionConfig reducedMotion="user">{children}</MotionConfig>
        </LazyMotion>
      </QueryClientProvider>
    </ApiProvider>
  );
}

export default function App() {
  const [client] = useState(createDefaultClient);
  const [queryClient] = useState(makeQueryClient);
  const [router] = useState(() => createBrowserRouter(routes));
  return (
    <Providers client={client} queryClient={queryClient}>
      <RouterProvider router={router} />
    </Providers>
  );
}
