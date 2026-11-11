# Deploy Kagu on a Hugging Face Space

A Space is its own git repository, so it gets a copy of Kagu plus the engine.

1. Create the bucket-backed storage (free): in Supabase open Project settings > API and copy the
   `service_role` key. It is secret: only ever put it in the Space secrets, never in the browser or in git.
   The server creates a private bucket called `kagu` on first start.
2. On huggingface.co create a new Space: SDK = Docker, blank template, hardware = CPU basic (free).
   Keep it Private if the engine code must stay private.
3. In the Space settings add these secrets:
   - `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET` (sign-in, see the main README)
   - `SUPABASE_SERVICE_KEY` (the service_role key from step 1)
   - optional: `KAGU_KEEP_DAYS=7` to stay well inside the 1 GB free storage
4. Build the Space folder and push it:

       git clone https://huggingface.co/spaces/<user>/<space> space
       cp -r server web requirements.txt Dockerfile space/
       cp deploy/space_readme.md space/README.md
       mkdir space/engine && cp -r <scripts>/lineart <scripts>/flat2svg space/engine/
       cd space && git add -A && git commit -m "Kagu" && git push

5. For more speed switch the Space hardware to CPU Upgrade (8 vCPU, 32 GB, billed per hour) in its settings.

Notes: the free Space sleeps after 48 hours without visitors and wakes on the next visit. Projects
survive because they are mirrored to Supabase Storage and restored at start-up; a job that was running
when the Space slept is put back in the queue.
