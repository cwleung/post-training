#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web/routers/rl_chapters.py — Research RL & Alignment Curriculum API
===================================================================
Exposes the 18 reinforcement learning curriculum chapters, serving
Markdown tutorial content and catalog metadata.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RL_DIR = _REPO_ROOT / "rl"
_TUTORIALS_DIR = _RL_DIR / "tutorials"

router = APIRouter()


@router.get("")
@router.get("/")
def list_chapters():
    """Lists all 18 RL chapters with titles, filenames, and byte sizes."""
    if not _TUTORIALS_DIR.exists():
        return {"total": 0, "chapters": []}

    chapters = []
    for f in sorted(_TUTORIALS_DIR.glob("*.md")):
        if f.name == "INDEX.md":
            continue
        title = f.stem
        try:
            with open(f, "r", encoding="utf-8") as tf:
                for line in tf:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
        except Exception:
            pass
        chapters.append({
            "id": f.stem,
            "filename": f.name,
            "title": title,
            "size_bytes": f.stat().st_size,
        })
    return {"total": len(chapters), "chapters": chapters}


@router.get("/{chapter_id}")
def get_chapter(chapter_id: str):
    """Returns markdown content for an RL chapter."""
    if not _TUTORIALS_DIR.exists():
        raise HTTPException(status_code=404, detail="RL tutorials directory not found")

    fpath = _TUTORIALS_DIR / f"{chapter_id}.md"
    if not fpath.exists():
        fpath = _TUTORIALS_DIR / chapter_id
    if not fpath.exists():
        # Support prefix matching (e.g., 'ch01' -> '01_*.md', or '01' -> '01_*.md')
        prefix = chapter_id.lower().replace("ch", "").strip()
        if prefix:
            matches = sorted(_TUTORIALS_DIR.glob(f"{prefix}_*.md"))
            if matches:
                fpath = matches[0]
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Chapter '{chapter_id}' not found")

    with open(fpath, "r", encoding="utf-8") as f:
        return {"id": chapter_id, "content": f.read()}
