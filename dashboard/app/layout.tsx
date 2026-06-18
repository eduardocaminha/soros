import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Soros Dashboard",
  description: "Trading bot monitor",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
