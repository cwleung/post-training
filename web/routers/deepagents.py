#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web/routers/deepagents.py — DeepAgents Tutorials API
===================================================
Provides endpoints for querying DeepAgents 31 tutorials markdown contents.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TUTORIALS_DIR = _REPO_ROOT / "tutorials"

router = APIRouter()


@router.get("/tutorials")
def list_tutorials():
    """Lists all DeepAgents tutorial markdown files and summaries."""
    tutorials = []
    for f in sorted(_TUTORIALS_DIR.glob("*.md")):
        if f.name == "index.md":
            continue
        # Extract title from first H1
        title = f.stem
        try:
            with open(f, "r", encoding="utf-8") as tf:
                for line in tf:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
        except Exception:
            pass
        tutorials.append({
            "id": f.stem,
            "filename": f.name,
            "title": title,
            "size_bytes": f.stat().st_size,
        })
    return {"total": len(tutorials), "tutorials": tutorials}


@router.get("/tutorials/{tutorial_id}")
def get_tutorial(tutorial_id: str):
    """Returns the markdown content of a tutorial."""
    fpath = _TUTORIALS_DIR / f"{tutorial_id}.md"
    if not fpath.exists():
        fpath = _TUTORIALS_DIR / tutorial_id
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Tutorial not found")
    with open(fpath, "r", encoding="utf-8") as f:
        return {"id": tutorial_id, "content": f.read()}

