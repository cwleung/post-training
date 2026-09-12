#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web/routers/rlvr.py — Post-Training ML Engineer & RLVR Curriculum API
====================================================================
Exposes the 14 post-training chapters, production algorithm source code,
and Kaggle/Colab notebook showcases.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RLVR_DIR = _REPO_ROOT / "rlvr"
_TUTORIALS_DIR = _RLVR_DIR / "tutorials"
_SRC_DIR = _RLVR_DIR / "src"
_NOTEBOOKS_DIR = _RLVR_DIR / "notebooks"
_SHOWCASE_DIR = _RLVR_DIR / "kaggle_showcase"

router = APIRouter()


@router.get("/chapters")
def list_chapters():
    """Lists all 14 post-training chapters with titles and metadata."""
    chapters = []
    for f in sorted(_TUTORIALS_DIR.glob("*.md")):
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


@router.get("/chapters/{chapter_id}")
def get_chapter(chapter_id: str):
    """Returns markdown content for a post-training chapter."""
    fpath = _TUTORIALS_DIR / f"{chapter_id}.md"
    if not fpath.exists():
        fpath = _TUTORIALS_DIR / chapter_id
    if not fpath.exists():
        # Support prefix matching (e.g., 'rlvr01' -> '01_*.md')
        prefix = chapter_id.lower().replace("rlvr", "").strip()
        if prefix:
            matches = sorted(_TUTORIALS_DIR.glob(f"{prefix}_*.md"))
            if matches:
                fpath = matches[0]
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Chapter '{chapter_id}' not found")
    with open(fpath, "r", encoding="utf-8") as f:
        return {"id": chapter_id, "content": f.read()}


@router.get("/code")
def list_code_modules():
    """Lists production algorithm modules or runnable notebooks."""
    if _SRC_DIR.exists():
        modules = []
        for f in sorted(_SRC_DIR.glob("*.py")):
            doc = ""
            try:
                with open(f, "r", encoding="utf-8") as sf:
                    content = sf.read()
                    if '"""' in content:
                        doc = content.split('"""')[1].strip()
            except Exception:
                pass
            modules.append({
                "name": f.name,
                "module": f.stem,
                "doc": doc.split("\n")[0] if doc else "",
                "size_bytes": f.stat().st_size,
            })
        return {"total": len(modules), "modules": modules}

    # If src is removed, list runnable notebooks
    items = []
    for d, cat in [(_NOTEBOOKS_DIR, "Notebooks"), (_SHOWCASE_DIR, "Showcase")]:
        if d.exists():
            for f in sorted(d.glob("*.ipynb")):
                items.append({
                    "name": f.name,
                    "module": f.stem,
                    "doc": f"Interactive {cat} (Jupyter Notebook)",
                    "size_bytes": f.stat().st_size,
                })
    return {"total": len(items), "modules": items}


@router.get("/code/{module_name}")
def get_code_module(module_name: str):
    """Returns Python source code of a module."""
    if not _SRC_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail="Standalone Python source scripts have been removed. Use interactive notebooks instead.",
        )
    fpath = _SRC_DIR / module_name
    if not fpath.exists():
        fpath = _SRC_DIR / f"{module_name}.py"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Module not found")
    with open(fpath, "r", encoding="utf-8") as f:
        return {"name": fpath.name, "code": f.read()}


@router.get("/notebooks")
def list_notebooks():
    """Lists all Kaggle and Colab showcase notebooks."""
    notebooks = []
    for d, cat in [(_NOTEBOOKS_DIR, "Colab/Kaggle"), (_SHOWCASE_DIR, "Showcase")]:
        if d.exists():
            for f in sorted(d.glob("*.ipynb")):
                notebooks.append({
                    "name": f.name,
                    "category": cat,
                    "path": f"{d.name}/{f.name}",
                    "size_bytes": f.stat().st_size,
                })
    return {"total": len(notebooks), "notebooks": notebooks}
