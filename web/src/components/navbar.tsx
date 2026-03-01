"use client"

import React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  ArrowLeft,
  Plus,
  Bell,
  Microphone,
  VideoCamera,
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { SidebarTrigger } from "@/components/ui/sidebar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"

interface NavbarProps {
  children?: React.ReactNode
  breadcrumbs?: { label: string; href?: string }[]
  actions?: React.ReactNode
}

export function Navbar({ children, breadcrumbs, actions }: NavbarProps) {
  const pathname = usePathname()

  // Default breadcrumbs if none provided
  const defaultBreadcrumbs = React.useMemo(() => {
    if (breadcrumbs) return breadcrumbs

    const parts = pathname.split("/").filter(Boolean)
    if (parts.length === 0) return [{ label: "Create", href: "/" }]

    return parts.map((part, i) => {
      // Custom labels for known routes
      let label = part.charAt(0).toUpperCase() + part.slice(1)
      if (part === "studio") label = "Studio Session"
      if (part === "files") label = "Files"

      return {
        label,
        href: "/" + parts.slice(0, i + 1).join("/")
      }
    })
  }, [pathname, breadcrumbs])

  return (
    <header className="h-12 border-b border-border bg-background flex items-center px-4 gap-4 flex-shrink-0 sticky top-0 z-50">
      <div className="flex items-center gap-2">
        <SidebarTrigger className="-ml-2" />
        <Separator orientation="vertical" className="h-4 mx-1" />



        <div className="flex items-center gap-2 text-xs font-semibold">
          {defaultBreadcrumbs.map((crumb, i) => (
            <React.Fragment key={crumb.label}>
              {i > 0 && <span className="text-muted-foreground/30 font-normal">/</span>}
              {crumb.href ? (
                <Link
                  href={crumb.href}
                  className={cn(
                    "transition-colors",
                    i === defaultBreadcrumbs.length - 1 ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {crumb.label}
                </Link>
              ) : (
                <span className="text-foreground">{crumb.label}</span>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="flex-1" />

      <div className="flex items-center gap-2">
        {actions ? actions : (
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground">
              <Bell size={16} />
            </Button>
          </div>
        )}
      </div>
    </header>
  )
}
