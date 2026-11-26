import { useEffect, useRef } from "react";

interface Props {
  open: boolean;
  title: string;
  message: string;
  confirmLabel: string;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Accessible confirmation dialog for destructive actions (native <dialog>: focus trap + Esc). */
export function ConfirmDialog({ open, title, message, confirmLabel, busy, onConfirm, onCancel }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onCancel={(e) => { e.preventDefault(); onCancel(); }}
      aria-labelledby="confirm-title"
      className="m-auto w-[min(92vw,420px)] rounded-2xl p-6 shadow-xl backdrop:bg-ink/40"
    >
      <h2 id="confirm-title" className="font-sans text-lg font-semibold">{title}</h2>
      <p className="mt-2 text-sm text-muted">{message}</p>
      <div className="mt-6 flex justify-end gap-3">
        <button type="button" className="btn-ghost" onClick={onCancel} disabled={busy} autoFocus>Cancel</button>
        <button type="button" className="btn-danger" onClick={onConfirm} disabled={busy}>{confirmLabel}</button>
      </div>
    </dialog>
  );
}
