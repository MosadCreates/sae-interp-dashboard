"use client";

type NavItem = {
  label: string;
  href: string;
};

const NAV_ITEMS: NavItem[] = [
  { label: "Features", href: "/" },
  { label: "Prompt Analyser", href: "/prompt" },
  { label: "Search", href: "/search" },
];

export default function NavBar() {
  return (
    <nav className="sticky top-0 z-50 bg-background/90 backdrop-blur border-b border-border">
      <div className="max-w-6xl mx-auto px-8 h-14 flex items-center justify-between">
        <a href="/" className="text-lg font-bold text-primary">
          Mini SAE
        </a>
        <div className="flex gap-6">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {item.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}
