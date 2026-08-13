import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedBrain Reference Desk",
  description: "Clinical operations reference tool",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
