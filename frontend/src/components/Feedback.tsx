import { AlertCircle, Loader2 } from "lucide-react";
import type { ReactNode } from "react";

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-10 text-sm text-muted">
      <Loader2 className="size-4 animate-spin" aria-hidden /> {label}
    </div>
  );
}

export function ErrorNotice({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-danger/30 bg-[#fbeeed] px-4 py-3 text-sm text-danger">
      <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />
      <p className="flex-1">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="font-semibold underline underline-offset-2">
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl bg-stone px-6 py-14 text-center">
      <h3 className="text-2xl">{title}</h3>
      {children && <div className="mx-auto mt-3 max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}
