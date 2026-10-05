import { Navigate, Outlet, Route, Routes } from "react-router-dom";
import { AsOfDayProvider, DatasetGate } from "./components/dataset";
import { Layout } from "./components/Layout";
import { DataModels } from "./pages/DataModels";
import { Diagnostics } from "./pages/Diagnostics";
import { CycleOptimizer } from "./pages/ds/CycleOptimizer";
import { DatasetPage } from "./pages/ds/DatasetPage";
import { EarlyWarning } from "./pages/ds/EarlyWarning";
import { FieldHistory } from "./pages/ds/FieldHistory";
import { SpmAdvisor } from "./pages/ds/SpmAdvisor";
import { WellHistory } from "./pages/ds/WellHistory";
import { FieldOverview } from "./pages/FieldOverview";
import { Optimizer } from "./pages/Optimizer";
import { Planner } from "./pages/Planner";
import { WellTwin } from "./pages/WellTwin";

/** Field-dataset pages share the as-of day and wait for the dataset store to be ready. */
function DatasetShell() {
  return (
    <AsOfDayProvider>
      <DatasetGate>
        <Outlet />
      </DatasetGate>
    </AsOfDayProvider>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        {/* Field history: the 300-well Baghewala dataset */}
        <Route element={<DatasetShell />}>
          <Route index element={<FieldHistory />} />
          <Route path="history" element={<Navigate to="/history/WELL-001" replace />} />
          <Route path="history/:wellId" element={<WellHistory />} />
          <Route path="css" element={<Navigate to="/css/WELL-001" replace />} />
          <Route path="css/:wellId" element={<CycleOptimizer />} />
          <Route path="spm" element={<Navigate to="/spm/WELL-010" replace />} />
          <Route path="spm/:wellId" element={<SpmAdvisor />} />
          <Route path="warning" element={<EarlyWarning />} />
          <Route path="dataset" element={<DatasetPage />} />
        </Route>
        {/* Live twin: real-time simulation of 12 wells */}
        <Route path="live" element={<FieldOverview />} />
        <Route path="wells/:wellId" element={<WellTwin />} />
        <Route path="planner" element={<Planner />} />
        <Route path="planner/:wellId" element={<Planner />} />
        <Route path="diagnostics" element={<Diagnostics />} />
        <Route path="diagnostics/:wellId" element={<Diagnostics />} />
        <Route path="optimizer" element={<Optimizer />} />
        <Route path="optimizer/:wellId" element={<Optimizer />} />
        <Route path="data" element={<DataModels />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
