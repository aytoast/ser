"use client"

import React from "react"
import { Sparkle, Bell } from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"

export function Navbar() {
  return (
    <header className="h-12 border-b border-border bg-card flex items-center justify-between px-5 sticky top-0 z-50">
      <div className="flex items-center gap-2 text-foreground">
        <Skeleton className="h-5 w-5 rounded bg-foreground/20" />
        <span className="text-sm font-bold tracking-tight">Ethos</span>
      </div>
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="sm" className="text-muted-foreground font-medium h-8 px-3">Feedback</Button>
        <Button variant="ghost" size="sm" className="text-muted-foreground font-medium h-8 px-3">Docs</Button>
        <Button variant="ghost" size="sm" className="text-muted-foreground font-medium h-8 px-3 gap-1">
          <Sparkle size={14} />Ask
        </Button>
        <Separator orientation="vertical" className="h-4 mx-2" />
        <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground">
          <Bell size={16} />
        </Button>
        <Avatar className="h-7 w-7 cursor-pointer ml-1">
          <AvatarFallback className="bg-secondary text-foreground text-[11px] font-bold">U</AvatarFallback>
        </Avatar>
      </div>
    </header>
  )
}
