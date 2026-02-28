import type { Metadata } from "next";
import { JetBrains_Mono, Lato } from "next/font/google";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

const lato = Lato({
  weight: ["100", "300", "400", "700", "900"],
  variable: "--font-lato",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Voxtral Studio | Emotional Speech Recognition",
  description: "Advanced emotional speech recognition and transcription studio.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${lato.variable} ${jetbrainsMono.variable} antialiased selection:bg-blue-500/30 font-sans`}
      >
        <TooltipProvider>
          {children}
        </TooltipProvider>
      </body>
    </html>
  );
}
