import type { Metadata } from "next";
import { Lato } from "next/font/google";
import { Suspense } from "react";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/sidebar";

const lato = Lato({
  weight: ["100", "300", "400", "700", "900"],
  variable: "--font-lato",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Ethos Studio | Emotional Speech Recognition",
  description: "Advanced emotional speech recognition and transcription studio powered by Evoxtral.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${lato.variable} antialiased selection:bg-black/10 font-sans`}
      >
        <TooltipProvider>
          <SidebarProvider>
            <Suspense fallback={<div className="w-64 flex-shrink-0" />}>
              <AppSidebar />
            </Suspense>
            <main className="flex-1 flex flex-col min-w-0 min-h-screen">
              {children}
            </main>
          </SidebarProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
