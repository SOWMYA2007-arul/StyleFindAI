import { BrowserRouter, Route, Routes } from "react-router";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { AuthProvider } from "./context/AuthContext";
import { SearchProvider } from "./context/SearchContext";
import { AdminLayout } from "./layouts/AdminLayout";
import { MainLayout } from "./layouts/MainLayout";
import { AboutPage } from "./pages/AboutPage";
import { AdminDashboardPage } from "./pages/AdminDashboardPage";
import { AdminLoginPage } from "./pages/AdminLoginPage";
import { CataloguePage } from "./pages/CataloguePage";
import { HomePage } from "./pages/HomePage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { ProductDetailsPage } from "./pages/ProductDetailsPage";
import { ProductManagementPage } from "./pages/ProductManagementPage";
import { SearchResultsPage } from "./pages/SearchResultsPage";
import { VisualSearchPage } from "./pages/VisualSearchPage";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <SearchProvider>
          <Routes>
            <Route element={<MainLayout />}>
              <Route index element={<HomePage />} />
              <Route path="search" element={<VisualSearchPage />} />
              <Route path="search/results" element={<SearchResultsPage />} />
              <Route path="catalogue" element={<CataloguePage />} />
              <Route path="products/:id" element={<ProductDetailsPage />} />
              <Route path="about" element={<AboutPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Route>
            <Route path="admin/login" element={<AdminLoginPage />} />
            <Route path="admin" element={<ProtectedRoute><AdminLayout /></ProtectedRoute>}>
              <Route index element={<AdminDashboardPage />} />
              <Route path="products" element={<ProductManagementPage />} />
            </Route>
          </Routes>
        </SearchProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
