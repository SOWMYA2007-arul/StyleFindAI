import { Link } from "react-router";
import { EmptyState } from "../components/Feedback";

export function NotFoundPage() {
  return (
    <section className="container-page py-16">
      <EmptyState title="This page doesn't exist">
        <Link to="/" className="font-semibold text-indigo underline">Go to the home page</Link>
      </EmptyState>
    </section>
  );
}
