#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#made with <3 by 550LY
"""
YouTube scraping comments (v1.1)
- Windows-friendly CLI
- deux modes:
    1) noapi   : utilise youtube-comment-downloader + yt_dlp (pas de clef API requise)
    2) api     : utilise YouTube Data API v3 over HTTP (requiert une clef API)
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from typing import List, Dict, Optional

import pandas as pd
import pytz
import requests
from tqdm import tqdm

# -------------------- Exceptions Personnalisées --------------------
class SkipVideo(Exception):
    """Signal that a video should be skipped (comments disabled, private, etc.)."""
    pass

# -------------------- Imports Optionnels --------------------
try:
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
except Exception:
    YoutubeCommentDownloader = None
    SORT_BY_RECENT = None

try:
    import yt_dlp
except Exception:
    yt_dlp = None

UTC = pytz.UTC

# -------------------- UTILITAIRES --------------------
def ensure_out_dir():
    out_dir = os.path.join(os.getcwd(), "out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def parse_date(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)

def dt_from_iso_z(iso: str) -> dt.datetime:

    return dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)

def to_iso_z(d: dt.datetime) -> str:
    return d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def sanitize_filename(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]+', '_', s)[:120] or "output"

def extract_video_id(url_or_id: str) -> str:
    # Pour accpeter les ID en plus des URLs
    if re.fullmatch(r"^[A-Za-z0-9_-]{11}$", url_or_id):
        return url_or_id
    patterns = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract video ID from: {url_or_id}")

def extract_channel_id_from_url(channel_url: str) -> Optional[str]:
    """
    For API mode: only supports URLs that contain /channel/UC...
    For other styles, return None.
    """
    m = re.search(r"/channel/(UC[0-9A-Za-z_-]+)", channel_url)
    if m:
        return m.group(1)
    return None

def save_output(rows: List[Dict], out_dir: str, base_name: str, fmt: str):
    if not rows:
        print("No comments collected. Nothing to save.")
        return None
    os.makedirs(out_dir, exist_ok=True)
    base = sanitize_filename(base_name)
    if fmt.lower() == "csv":
        out_path = os.path.join(out_dir, f"{base}.csv")
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        out_path = os.path.join(out_dir, f"{base}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return out_path

# -------------------- MODE NO API --------------------
def list_channel_videos_noapi(channel_url: str, max_videos: Optional[int], sleep_s: float) -> List[Dict]:
    if yt_dlp is None:
        raise RuntimeError("yt_dlp is not installed. Re-run install.bat")
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True,
        'dump_single_json': True,
        'playlistend': max_videos if max_videos else None,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = info.get("entries", [])
    out = []
    for e in entries:
        vid = e.get("id")
        if vid and re.fullmatch(r"^[A-Za-z0-9_-]{11}$", vid):
            out.append({
                "videoId": vid,
                "title": e.get("title"),
                "channelId": e.get("channel_id"),
                "channelTitle": e.get("channel"),
                "url": f"https://www.youtube.com/watch?v={vid}"
            })
    # polite sleep
    if sleep_s > 0:
        time.sleep(sleep_s)
    return out

def fetch_comments_noapi(video_id: str, max_comments: Optional[int], include_replies: bool,
                         since: Optional[dt.datetime], until: Optional[dt.datetime],
                         sleep_s: float) -> List[Dict]:
    if YoutubeCommentDownloader is None:
        raise RuntimeError("youtube-comment-downloader is not installed. Re-run install.bat")
    downloader = YoutubeCommentDownloader()
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        comments_iter = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
    except Exception as e:
        print(f"SKIP {video_id} (noapi): {e}")
        return []
    rows = []
    count = 0
    for c in comments_iter:
        row = {
            "commentId": c.get("cid"),
            "videoId": video_id,
            "videoTitle": None,
            "channelId": None,
            "channelTitle": None,
            "author": c.get("author"),
            "authorChannelId": c.get("channel"),
            "text": c.get("text"),
            "publishedAt": None,
            "updatedAt": None,
            "likeCount": c.get("votes"),
            "replyCount": None,
            "parentId": c.get("reply_to"),
            "isReply": bool(c.get("reply_to")),
        }
        rows.append(row)
        count += 1
        if max_comments and count >= max_comments:
            break
    if sleep_s > 0:
        time.sleep(sleep_s)
    return rows

# -------------------- API Mode --------------------
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

def api_request(endpoint: str, params: Dict) -> Dict:
    url = f"{YOUTUBE_API_BASE}/{endpoint}"
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        try:
            payload = r.json()
            msg = (payload.get('error', {}).get('message') or '').lower()
            reasons = ' '.join([e.get('reason','') for e in payload.get('error', {}).get('errors', [])]).lower()
        except Exception:
            payload, msg, reasons = None, r.text.lower(), ''
        markers = [
            'disabled comments',
            'commentsdisabled',
            'comments are disabled',
            'comments have been disabled',
            'video is private',
            'private video',
        ]
        if r.status_code == 403 and (any(m in msg for m in markers) or any(m in reasons for m in ['commentsdisabled', 'private'])):
            raise SkipVideo(msg or 'Comments disabled/private')
        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
    return r.json()

def list_channel_videos_api(api_key: str, channel_id: str,
                            published_after: Optional[dt.datetime],
                            published_before: Optional[dt.datetime],
                            max_videos: Optional[int],
                            sleep_s: float) -> List[Dict]:
    params = {
        "key": api_key,
        "part": "snippet",
        "channelId": channel_id,
        "type": "video",
        "order": "date",
        "maxResults": 50,
    }
    if published_after:
        params["publishedAfter"] = to_iso_z(published_after)
    if published_before:
        params["publishedBefore"] = to_iso_z(published_before)
    videos = []
    total = 0
    page = 1
    while True:
        data = api_request("search", params)
        items = data.get("items", [])
        for it in items:
            vid = it["id"]["videoId"]
            sn = it["snippet"]
            videos.append({
                "videoId": vid,
                "title": sn.get("title"),
                "channelId": sn.get("channelId"),
                "channelTitle": sn.get("channelTitle"),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "publishedAt": sn.get("publishedAt"),
            })
            total += 1
            if max_videos and total >= max_videos:
                break
        if max_videos and total >= max_videos:
            break
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
        if sleep_s > 0:
            time.sleep(sleep_s)
        page += 1
    return videos

def fetch_comments_api_for_video(api_key: str, video_id: str,
                                 max_comments: Optional[int],
                                 include_replies: bool,
                                 since: Optional[dt.datetime],
                                 until: Optional[dt.datetime],
                                 sleep_s: float) -> List[Dict]:
    # Récupere les commentaires top-level en utilisant commentThreads.list    
    comments = []
    params = {
        "key": api_key,
        "part": "snippet,replies",
        "videoId": video_id,
        "maxResults": 100,
        "order": "time",  # newest first pour filtrer par date simplement
        # "textFormat": "plainText",
    }
    total = 0
    while True:
        try:
            data = api_request("commentThreads", params)
        except SkipVideo as sk:
            print(f"SKIP {video_id}: {sk}")
            return []
        items = data.get("items", [])
        for it in items:
            top = it["snippet"]["topLevelComment"]["snippet"]
            top_id = it["snippet"]["topLevelComment"]["id"]
            published = dt_from_iso_z(top["publishedAt"])
            if since and published < since:
                pass
            if until and published > until:
                continue

            row = {
                "commentId": top_id,
                "videoId": video_id,
                "videoTitle": None,
                "channelId": None,
                "channelTitle": None,
                "author": top.get("authorDisplayName"),
                "authorChannelId": (top.get("authorChannelId") or {}).get("value"),
                "text": top.get("textDisplay"),
                "publishedAt": top.get("publishedAt"),
                "updatedAt": top.get("updatedAt"),
                "likeCount": top.get("likeCount"),
                "replyCount": it["snippet"].get("totalReplyCount"),
                "parentId": None,
                "isReply": False,
            }
            comments.append(row)
            total += 1
            if max_comments and total >= max_comments:
                break

            if include_replies and it["snippet"].get("totalReplyCount", 0) > 0:
                parent_id = top_id
                # Recup de toutes les réponses
                rep_params = {
                    "key": api_key,
                    "part": "snippet",
                    "parentId": parent_id,
                    "maxResults": 100,
                }
                while True:
                    try:
                        rep_data = api_request("comments", rep_params)
                    except SkipVideo as sk:
                        print(f"SKIP replies for {video_id}: {sk}")
                        break
                    rep_items = rep_data.get("items", [])
                    for rep in rep_items:
                        sn = rep["snippet"]
                        rep_published = dt_from_iso_z(sn["publishedAt"])
                        if since and rep_published < since:
                            pass
                        elif until and rep_published > until:
                            pass
                        else:
                            comments.append({
                                "commentId": rep["id"],
                                "videoId": video_id,
                                "videoTitle": None,
                                "channelId": None,
                                "channelTitle": None,
                                "author": sn.get("authorDisplayName"),
                                "authorChannelId": (sn.get("authorChannelId") or {}).get("value"),
                                "text": sn.get("textDisplay"),
                                "publishedAt": sn.get("publishedAt"),
                                "updatedAt": sn.get("updatedAt"),
                                "likeCount": sn.get("likeCount"),
                                "replyCount": None,
                                "parentId": parent_id,
                                "isReply": True,
                            })
                            total += 1
                        if max_comments and total >= max_comments:
                            break
                    if max_comments and total >= max_comments:
                        break
                    token = rep_data.get("nextPageToken")
                    if not token:
                        break
                    rep_params["pageToken"] = token
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            if max_comments and total >= max_comments:
                break

        if max_comments and total >= max_comments:
            break
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
        if sleep_s > 0:
            time.sleep(sleep_s)
    return comments

# -------------------- DEFINITION DU CLI --------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Scrape YouTube comments (video or channel).")
    p.add_argument("--mode", choices=["noapi", "api"], required=True, help="Scrape mode.")
    p.add_argument("--video", help="Video URL or ID.")
    p.add_argument("--channel", help="Channel URL (supports @handle, /channel/UC..., /c/username) in noapi mode; in api mode, prefer /channel/UC...")
    p.add_argument("--channel-id", help="Channel ID (UC...) for API mode.")
    p.add_argument("--max-comments", type=int, default=None, help="Max total comments to fetch.")
    p.add_argument("--max-videos", type=int, default=None, help="Max number of recent videos from a channel.")
    p.add_argument("--since", help="ISO date (YYYY-MM-DD) lower bound for comment publish time (API mode only exact).")
    p.add_argument("--until", help="ISO date (YYYY-MM-DD) upper bound for comment publish time (API mode only exact).")
    p.add_argument("--include-replies", action="store_true", help="Also fetch replies to top-level comments.")
    p.add_argument("--output-format", choices=["csv", "json"], default="csv")
    p.add_argument("--api-key", help="YouTube Data API v3 key (required in api mode).")
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between paged requests.")
    return p

def main():
    args = build_arg_parser().parse_args()

    out_dir = ensure_out_dir()

    since = parse_date(args.since) if args.since else None
    until = parse_date(args.until) if args.until else None

    if args.mode == "api":
        if not args.api_key:
            print("--api-key is required in api mode.", file=sys.stderr)
            sys.exit(2)

        if not args.video and not (args.channel or args.channel_id):
            print("Provide --video or --channel (or --channel-id) in api mode.", file=sys.stderr)
            sys.exit(2)

        if args.video:
            vid = extract_video_id(args.video)
            rows = fetch_comments_api_for_video(
                api_key=args.api_key,
                video_id=vid,
                max_comments=args.max_comments,
                include_replies=args.include_replies,
                since=since,
                until=until,
                sleep_s=args.sleep,
            )
            if not rows:
                print(f"No comments saved for {vid} (possibly disabled/private).")
                return
            out_path = save_output(rows, out_dir, f"comments_{vid}", args.output_format)
            if out_path:
                print(f"Saved: {out_path}")
            return


        if args.channel_id:
            ch_id = args.channel_id
        else:
            ch_id = None
            if args.channel:
                ch_id = extract_channel_id_from_url(args.channel)
                if not ch_id:
                    print("API mode requires a /channel/UC... URL or --channel-id. Use noapi mode for handles.", file=sys.stderr)
                    sys.exit(2)

        videos = list_channel_videos_api(
            api_key=args.api_key,
            channel_id=ch_id,
            published_after=since,
            published_before=until,
            max_videos=args.max_videos,
            sleep_s=args.sleep,
        )
        print(f"Found {len(videos)} videos.")
        total_rows = []
        for v in tqdm(videos, desc="Scraping videos (API)"):
            vid = v["videoId"]
            rows = fetch_comments_api_for_video(
                api_key=args.api_key,
                video_id=vid,
                max_comments=args.max_comments,
                include_replies=args.include_replies,
                since=since,
                until=until,
                sleep_s=args.sleep,
            )
            for r in rows:
                r["videoTitle"] = v.get("title")
                r["channelId"] = v.get("channelId")
                r["channelTitle"] = v.get("channelTitle")
            if rows:
                total_rows.extend(rows)
        out_path = save_output(total_rows, out_dir, f"comments_channel_{ch_id}", args.output_format)
        if out_path:
            print(f"Saved: {out_path}")
        return

    # noapi mode
    if args.mode == "noapi":
        if not args.video and not args.channel:
            print("Provide --video or --channel in noapi mode.", file=sys.stderr)
            sys.exit(2)

        if args.video:
            vid = extract_video_id(args.video)
            rows = fetch_comments_noapi(
                video_id=vid,
                max_comments=args.max_comments,
                include_replies=args.include_replies,
                since=since,
                until=until,
                sleep_s=args.sleep,
            )
            if not rows:
                print(f"No comments saved for {vid}.")
                return
            out_path = save_output(rows, out_dir, f"comments_{vid}", args.output_format)
            if out_path:
                print(f"Saved: {out_path}")
            return

        # (noapi-rep)
        videos = list_channel_videos_noapi(args.channel, args.max_videos, args.sleep)
        print(f"Found {len(videos)} videos.")
        total_rows = []
        for v in tqdm(videos, desc="Scraping videos (noapi)"):
            vid = v["videoId"]
            rows = fetch_comments_noapi(
                video_id=vid,
                max_comments=args.max_comments,
                include_replies=args.include_replies,
                since=parse_date(args.since) if args.since else None,
                until=parse_date(args.until) if args.until else None,
                sleep_s=args.sleep,
            )
            for r in rows:
                r["videoTitle"] = v.get("title")
                r["channelId"] = v.get("channelId")
                r["channelTitle"] = v.get("channelTitle")
            if rows:
                total_rows.extend(rows)
        base_name = "comments_channel"
        if args.channel:
            base_name += "_" + sanitize_filename(args.channel.split("/")[-1])
        out_path = save_output(total_rows, out_dir, base_name, args.output_format)
        if out_path:
            print(f"Saved: {out_path}")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(1)
    except SkipVideo as e:
        print(f"SKIP: {e}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
# EOF