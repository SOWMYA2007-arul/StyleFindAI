const PIPELINE = [
  ["Validation", "The file's real signature, size and dimensions are checked twice: in the Node.js API and in the Python AI service. Phone photos are rotated using their EXIF data and converted to RGB."],
  ["Optional garment isolation", "When enabled, a YOLO detector crops the person or garment and SAM 2 can mask the background. If either step fails, the full photo is used so the search never breaks."],
  ["Embedding", "FashionCLIP, a CLIP ViT-B/32 model fine-tuned on about 800,000 fashion products, turns the photo into a 512-number vector. Similar garments get similar vectors."],
  ["Vector search", "Catalogue vectors are computed once, offline, and stored in PostgreSQL with the pgvector extension. Your vector is compared with them using cosine distance through an HNSW index, returning the 50 closest products."],
  ["Hybrid re-ranking", "The 50 candidates are re-scored: 65% visual similarity, 15% colour (compared in the CIELAB colour space), 10% pattern texture and 10% attributes such as category and sleeves. Missing information is skipped, not penalised."],
  ["Exact match check", "A result is labelled an exact match only when the photo is nearly identical to exactly one catalogue image (perceptual hash plus a colour check). Otherwise results are called similar or related."],
] as const;

export function AboutPage() {
  return (
    <section className="container-page max-w-3xl py-14">
      <h1 className="text-4xl sm:text-5xl">How StyleFind AI works</h1>
      <p className="mt-5 text-lg leading-relaxed text-muted">
        StyleFind is an image-to-image search engine for clothing. It does not classify your photo into a fixed list; it measures
        how close your photo is to every product photo in the catalogue and ranks them.
      </p>
      <ol className="mt-12 space-y-10">
        {PIPELINE.map(([t, b], i) => (
          <li key={t} className="grid grid-cols-[3rem_1fr] gap-4">
            <span className="font-display text-4xl leading-none text-indigo/40">{i + 1}</span>
            <div>
              <h2 className="font-sans text-lg font-semibold">{t}</h2>
              <p className="mt-2 leading-relaxed text-muted">{b}</p>
            </div>
          </li>
        ))}
      </ol>
      <h2 className="mt-16 text-3xl">Searching with words and a photo</h2>
      <p className="mt-4 leading-relaxed text-muted">
        When you add text such as "in blue with full sleeves", the system describes what it sees in your photo, writes the same
        description with your change applied, and moves your photo's vector in the direction between the two text vectors. It also
        boosts products whose catalogue colour and sleeves match your request. This is a training-free method, so it works best for
        simple changes like colour and sleeve length.
      </p>
      <h2 className="mt-16 text-3xl">Limits</h2>
      <ul className="mt-4 list-disc space-y-2 pl-5 leading-relaxed text-muted">
        <li>The catalogue mostly contains studio photos on plain backgrounds, so busy street photos match less reliably.</li>
        <li>Attributes like neckline are only known when the product name mentions them; otherwise they show as "Not listed".</li>
        <li>The match score is a ranking signal, not a probability that two photos show the same item.</li>
      </ul>
    </section>
  );
}
