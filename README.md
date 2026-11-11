# Kagu

Flat graphics to SVG, then edit the result. Everything is computed on the server; the browser only shows it.

- **Queue**: tracing jobs wait in a persistent queue (one at a time). Close the page and come back later: the page
  remembers your projects ("My projects" on the first step), shows the queue position and progress, and the link
  `/#<project id>` reopens a project from any device. Edits are saved on the server after every step.
- **Trace**: upload a PNG/JPEG/WebP, the Line Art engine (`lineart` + `flat2svg` from the scripts repo) turns it into SVG
  in a background job with progress. SVG files are imported directly.
- **Edit** (server side, with undo/redo/reset): delete shapes (remove, or fill with the neighbour colour), remove the
  background, select specks, scan problem spots, merge/recolour colours, auto-merge by Lab distance (ΔE), limit the
  number of colours, shift hue/saturation/lightness. Sliders show a server-rendered preview before you apply.
- **Export** (server side): SVG (group by colour, combine shapes, Illustrator-friendly, stock-ready), PNG, JPEG, WebP at
  any size up to 12000 px, background, quality, 300 DPI metadata.

## Run locally

```
pip install -r requirements.txt          # also install librsvg (rsvg-convert) or cairosvg
export KAGU_ENGINE_DIR=/path/to/folder/with/lineart-and-flat2svg   # e.g. a checkout of the scripts repo
uvicorn server.app:app --port 8080
```

Open http://localhost:8080. Without the engine, SVG upload, editing and export still work.

## Deploy to Google Cloud Run

Copy `lineart/` and `flat2svg/` from the scripts repo into `engine/`, then:

```
gcloud run deploy kagu --source . --region us-central1 --memory 8Gi --cpu 4 --timeout 3600 \
  --concurrency 8 --min-instances 1 --max-instances 1 --allow-unauthenticated
```

Projects are stored in `KAGU_DATA_DIR` (default `/tmp/kagu`). On Cloud Run that disk is temporary: to keep projects
across restarts mount a Cloud Storage bucket (or a Filestore volume) and point `KAGU_DATA_DIR` at it. Keep
`--max-instances 1` for now: the queue lives in one process.

## Sign-in (Supabase, optional)

Without these variables the server is open (fine for local use). With them, every project belongs to a signed-in
user and nobody else can open it.

1. Create a Supabase project. Authentication -> Providers: keep Email (magic link) on; add Google if you want it.
   Authentication -> URL Configuration: add your Kagu address to the redirect URLs.
2. Set on the server: `SUPABASE_URL`, `SUPABASE_ANON_KEY` (Project Settings -> API), and `SUPABASE_JWT_SECRET`
   (Project Settings -> API -> JWT Secret, the legacy HS256 secret).
3. Each user may have `KAGU_MAX_QUEUED_PER_USER` (3) jobs waiting at once.

Not done yet: emails when a project is ready, payments, per-user quotas.

## Settings (environment variables)

`KAGU_ENGINE_DIR`, `KAGU_DATA_DIR` (default temp dir), `KAGU_MAX_UPLOAD_MB` (40), `KAGU_TRACE_WORKERS` (1),
`KAGU_KEEP_DAYS` (14: finished projects older than this are deleted).

## MVP limits

- No accounts: anyone with the link can use the server. Add auth (Supabase) before sharing publicly.
- Without the Supabase variables anyone who has a project link can open and edit it.
- One server process owns the queue; undo history is not kept across restarts (the edited scene is).
- Tracing takes minutes for large line art (the engine upscales small images); it runs one job at a time.
- Not yet ported from the first prototype: node simplification.

## Free hosting

See `deploy/HUGGINGFACE.md` for the free setup (Hugging Face Space + Supabase Storage for persistence).
