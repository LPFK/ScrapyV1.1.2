#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate YouTube comments using the DeepL API and append a translated column to CSV/JSON.
"""
import argparse, glob, os, sys, time
from typing import List, Tuple, Optional

import pandas as pd
import requests
from tqdm import tqdm

DEEPL_FREE_HOST = "api-free.deepl.com"
DEEPL_PRO_HOST  = "api.deepl.com"

def pick_deepl_host(api_key: str, force_free: bool = False, force_pro: bool = False) -> str:
    if force_free and force_pro:
        raise ValueError("Choose either --free or --pro, not both.")
    if force_free:
        return DEEPL_FREE_HOST
    if force_pro:
        return DEEPL_PRO_HOST
    return DEEPL_FREE_HOST if (":fx" in api_key) else DEEPL_PRO_HOST

def translate_batch(api_key: str, host: str, texts: List[str], target_lang: str,
                    source_lang: Optional[str] = None, formality: Optional[str] = None,
                    timeout: int = 60) -> List[Tuple[str, Optional[str]]]:
    url = f"https://{host}/v2/translate"
    data = {
        "auth_key": api_key,
        "target_lang": target_lang,
        "preserve_formatting": 1,
    }
    if source_lang:
        data["source_lang"] = source_lang
    if formality:
        data["formality"] = formality
    for t in texts:
        data.setdefault("text", []).append("" if t is None else str(t))

    r = requests.post(url, data=data, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"DeepL API error {r.status_code}: {r.text[:300]}")
    payload = r.json()
    out = []
    for tr in payload.get("translations", []):
        out.append((tr.get("text"), tr.get("detected_source_language")))
    return out

def chunk_indices(n, batch_size):
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        yield i, j
        i = j

def _out_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    # Always write CSV for simplicity
    return f"{base}_translated.csv"

def process_file(path: str, api_key: str, target_lang: str, source_lang: Optional[str],
                 formality: Optional[str], batch_size: int, only_missing: bool,
                 inplace: bool, force_free: bool, force_pro: bool, sleep_s: float) -> str:
    host = pick_deepl_host(api_key, force_free, force_pro)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if "text" not in df.columns:
        raise ValueError("Input must contain a 'text' column.")

    # Ensure columns exist
    if "text_translated" not in df.columns:
        df["text_translated"] = ""
    if "deepl_detected_source_lang" not in df.columns:
        df["deepl_detected_source_lang"] = ""

    # indices to translate
    candidates = df.index.tolist()
    if only_missing:
        candidates = [i for i in candidates if pd.isna(df.at[i, "text_translated"]) or str(df.at[i, "text_translated"]).strip() == ""]

    if not candidates:
        out_path = path if inplace else _out_path(path)
        df.to_csv(out_path, index=False)
        return out_path

    for (i, j) in tqdm(list(chunk_indices(len(candidates), batch_size)), desc=f"Translating {os.path.basename(path)}"):
        batch_rows = candidates[i:j]
        batch_texts = [str(df.at[idx, "text"]) if not pd.isna(df.at[idx, "text"]) else "" for idx in batch_rows]

        # backoff on transient errors
        tries, backoff = 0, 2.0
        while True:
            try:
                translations = translate_batch(api_key, host, batch_texts, target_lang, source_lang, formality)
                break
            except Exception as e:
                tries += 1
                if tries > 5:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

        for offset, (txt, detected) in enumerate(translations):
            idx = batch_rows[offset]
            df.at[idx, "text_translated"] = txt
            if detected:
                df.at[idx, "deepl_detected_source_lang"] = detected

        if sleep_s > 0:
            time.sleep(sleep_s)

    out_path = path if inplace else _out_path(path)
    df.to_csv(out_path, index=False)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Translate comments using DeepL and append a translated column.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input glob(s), e.g., out\\*.csv")
    ap.add_argument("--target-lang", required=True, help="Target language (DeepL code, e.g., EN, EN-GB, FR, DE, ES, PT-BR, ZH, JA)")
    ap.add_argument("--source-lang", default=None, help="Source language code (optional, auto-detect if omitted)")
    ap.add_argument("--formality", default=None, choices=["default","more","less"], help="Formality level (supported languages only)")
    ap.add_argument("--batch-size", type=int, default=40, help="How many comments to send per request")
    ap.add_argument("--only-missing", action="store_true", help="Only translate rows missing 'text_translated'")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the original CSV")
    ap.add_argument("--deepl-key", default=None, help="DeepL API key (or set DEEPL_API_KEY env var)")
    ap.add_argument("--free", action="store_true", help="Force api-free.deepl.com")
    ap.add_argument("--pro", action="store_true", help="Force api.deepl.com")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests (optional throttle)")
    args = ap.parse_args()

    api_key = args.deepl_key or os.environ.get("DEEPL_API_KEY", "").strip()
    if not api_key:
        print("Error: Provide a DeepL API key via --deepl-key or DEEPL_API_KEY env var.", file=sys.stderr)
        sys.exit(2)

    files = []
    for pat in args.inputs:
        files.extend(glob.glob(pat))
    if not files:
        print("No input files matched.", file=sys.stderr)
        sys.exit(2)

    out_paths = []
    for p in files:
        out_paths.append(process_file(p, api_key, args.target_lang, args.source_lang, args.formality,
                                      args.batch_size, args.only_missing, args.inplace, args.free, args.pro, args.sleep))
    print("Done. Wrote:")
    for op in out_paths:
        print("  ", op)

if __name__ == "__main__":
    main()
