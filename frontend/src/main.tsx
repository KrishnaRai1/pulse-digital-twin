import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import type { ApiError } from "./api/client";
import "./index.css";
import "./lib/echarts";
import { LiveProvider } from "./lib/stream";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // network errors (API asleep on free hosting, restarting) are retried with back-off; HTTP errors once
      retry: (n, err) => ((err as ApiError).status === 0 ? n < 6 : n < 1),
      retryDelay: (n) => Math.min(1000 * 2 ** n, 15000),
      refetchOnWindowFocus: false,
      staleTime: 2000,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <LiveProvider>
          <App />
        </LiveProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
