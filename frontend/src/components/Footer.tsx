import { Link } from "react-router";

export function Footer() {
  return (
    <footer className="mt-24 bg-indigo text-white/80">
      <div className="container-page grid gap-10 py-14 md:grid-cols-[2fr_1fr_1fr]">
        <div>
          <p className="font-display text-3xl text-white">StyleFind</p>
          <p className="mt-3 max-w-sm text-sm leading-relaxed">
            Photograph a kurta, saree or dress you like and find it, or the closest pieces to it, in our catalogue.
          </p>
        </div>
        <div className="flex flex-col gap-2 text-sm">
          <p className="mb-1 font-semibold text-white">Shop</p>
          <Link to="/catalogue" className="hover:text-white">Full catalogue</Link>
          <Link to="/search" className="hover:text-white">Search by photo</Link>
        </div>
        <div className="flex flex-col gap-2 text-sm">
          <p className="mb-1 font-semibold text-white">About</p>
          <Link to="/about" className="hover:text-white">How the AI works</Link>
          <Link to="/admin/login" className="hover:text-white">Admin</Link>
        </div>
      </div>
      <p className="container-page border-t border-white/15 py-5 text-xs text-white/60">
        Product images come from public research datasets and are shown with their recorded source and licence.
      </p>
    </footer>
  );
}
