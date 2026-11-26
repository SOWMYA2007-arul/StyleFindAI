import { ImagePlus, X } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";

const ACCEPT = ["image/jpeg", "image/png", "image/webp"];
const MAX_MB = 8;

/** Checks type and size in the browser for quick feedback; the server re-validates everything. */
export function validateImageFile(f: File): string | null {
  if (!ACCEPT.includes(f.type)) return "Use a JPG, PNG or WEBP photo.";
  if (f.size > MAX_MB * 1024 * 1024) return `The photo is larger than ${MAX_MB} MB.`;
  return null;
}

interface Props {
  previewUrl: string | null;
  onFile: (f: File | null) => void;
  onError?: (msg: string | null) => void;
  tall?: boolean;
}

export function ImageDropzone({ previewUrl, onFile, onError, tall }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (f: File | undefined) => {
    if (!f) return;
    const err = validateImageFile(f);
    onError?.(err);
    if (!err) onFile(f);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    take(e.dataTransfer.files?.[0]);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={`relative flex ${tall ? "min-h-[440px]" : "min-h-[320px]"} items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed transition-colors ${
        over ? "border-indigo bg-indigo-soft" : "border-stone-deep bg-stone"
      }`}
    >
      {previewUrl ? (
        <>
          <img src={previewUrl} alt="Your uploaded photo" className="max-h-[480px] w-full object-contain" />
          <button
            type="button"
            onClick={() => {
              onFile(null);
              if (input.current) input.current.value = "";
            }}
            className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-full bg-white/95 px-3 py-1.5 text-xs font-semibold shadow-sm hover:text-danger"
          >
            <X className="size-3.5" aria-hidden /> Remove photo
          </button>
        </>
      ) : (
        <div className="flex flex-col items-center px-6 text-center">
          <ImagePlus className="size-10 text-indigo" strokeWidth={1.4} aria-hidden />
          <p className="mt-4 font-display text-2xl">Drop a photo of a garment</p>
          <p className="mt-2 max-w-xs text-sm text-muted">A catalogue shot, a screenshot or your own photo. JPG, PNG or WEBP, up to {MAX_MB} MB.</p>
          <button type="button" className="btn-ghost mt-6" onClick={() => input.current?.click()}>
            Browse files
          </button>
        </div>
      )}
      <input
        ref={input}
        type="file"
        accept={ACCEPT.join(",")}
        className="sr-only"
        aria-label="Choose a photo"
        onChange={(e) => take(e.target.files?.[0])}
      />
    </div>
  );
}
