import { titleCase } from "./ProductCard";

interface Props {
  legend: string;
  options: { value: string; label?: string; count?: number }[];
  selected: string[];
  onChange: (next: string[]) => void;
  max?: number;
}

export function CheckboxGroup({ legend, options, selected, onChange, max = 12 }: Props) {
  if (!options.length) return null;
  return (
    <fieldset className="border-t border-stone-deep pt-4">
      <legend className="mb-2 text-sm font-semibold">{legend}</legend>
      <div className="flex flex-col gap-1.5">
        {options.slice(0, max).map((o) => {
          const on = selected.includes(o.value);
          return (
            <label key={o.value} className="flex cursor-pointer items-center gap-2 text-sm text-muted hover:text-ink">
              <input
                type="checkbox"
                checked={on}
                onChange={() => onChange(on ? selected.filter((s) => s !== o.value) : [...selected, o.value])}
                className="size-4 accent-indigo"
              />
              <span className="flex-1">{o.label ?? titleCase(o.value)}</span>
              {o.count !== undefined && <span className="text-xs">{o.count}</span>}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
