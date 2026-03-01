"use client"

import React, { useEffect, useState } from "react"
import Link from "next/link"
import { usePathname, useSearchParams } from "next/navigation"
import {
  Waveform,
  Handshake,
  Heart,
  Plus,
} from "@phosphor-icons/react"
import Image from "next/image"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import {
  listRecentSessions,
  ensureStoreInitialized,
  type RecentSession,
} from "@/lib/session-store"

const PRODUCTS_ITEMS = [
  { title: "Sales", url: "/sales", icon: Handshake, disabled: true },
  { title: "Companionship", url: "/companionship", icon: Heart, disabled: true },
]

function formatDuration(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, "0")}`
}

function stripExtension(name: string) {
  return name.replace(/\.[^/.]+$/, "")
}

export function AppSidebar() {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const currentSessionId = searchParams.get("s")
  const [recent, setRecent] = useState<RecentSession[]>([])

  // Load and refresh recent sessions whenever pathname changes
  useEffect(() => {
    ensureStoreInitialized()
    setRecent(listRecentSessions())
  }, [pathname])

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="flex items-center gap-2 px-2 py-1">
          <Image src="/logo.svg" alt="Ethos Studio" width={32} height={32} className="rounded-lg" />
          <span className="text-sm flex items-center gap-1.5">
            <span className="font-semibold">Ethos</span>
            <span className="text-muted-foreground/40 font-light">|</span>
            <span className="font-medium text-muted-foreground">Studio</span>
          </span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        {/* Top-level nav */}
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={pathname === "/"}>
                  <Link href="/">
                    <Plus />
                    <span>Create</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Recent Sessions */}
        <SidebarGroup>
          <SidebarGroupLabel>
            Recent
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {recent.length === 0 ? (
                <li className="px-2 py-1.5 text-xs text-muted-foreground/60">
                  No sessions yet
                </li>
              ) : (
                recent.map((session) => {
                  const href = `/studio?s=${session.id}`
                  return (
                    <SidebarMenuItem key={session.id}>
                      <SidebarMenuButton
                        asChild
                        isActive={pathname === "/studio" && currentSessionId === session.id}
                        className="h-auto py-1.5 items-start"
                      >
                        <Link href={href}>
                          <Waveform className="mt-0.5 shrink-0" />
                          <div className="flex flex-col min-w-0">
                            <span className="truncate text-sm leading-snug">
                              {stripExtension(session.filename)}
                            </span>
                            <span className="text-xs text-muted-foreground font-normal">
                              {formatDuration(session.duration)} · {session.speakerCount} speaker{session.speakerCount !== 1 ? "s" : ""}
                            </span>
                          </div>
                        </Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )
                })
              )}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>


        {/* Products */}
        <SidebarGroup>
          <SidebarGroupLabel>Products</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {PRODUCTS_ITEMS.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild={!item.disabled}
                    isActive={pathname === item.url}
                    disabled={item.disabled}
                    className={item.disabled ? "opacity-50 cursor-not-allowed" : ""}
                  >
                    {item.disabled ? (
                      <div className="flex items-center gap-2">
                        <item.icon />
                        <span>{item.title}</span>
                      </div>
                    ) : (
                      <Link href={item.url}>
                        <item.icon />
                        <span>{item.title}</span>
                      </Link>
                    )}
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton disabled className="opacity-60 cursor-default">
              <span className="flex items-center gap-2 w-full">
                <span>Developer</span>
                <span className="ml-auto text-[10px] font-medium px-1.5 py-0.5 rounded border border-border text-muted-foreground bg-muted leading-none capitalize">
                  Coming soon
                </span>
              </span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}
