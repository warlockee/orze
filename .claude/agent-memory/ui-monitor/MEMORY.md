# UI Monitor Memory Index

## Orze Admin UI Architecture
- SPA (React + Vite) served by FastAPI via `StaticFiles(directory=..., html=True)` mount
- Tab-based navigation (useState), NOT client-side routing -- no /queue, /nodes sub-routes exist
- Lazy-loaded chunks: QueueTabView, NodesTab, RunsTab, AlertsTab, etc. fetched from /assets/ on demand
- Server code: `/home/ec2-user/fsx/vlm/orze/src/orze/admin/server.py`
- UI source: `/home/ec2-user/fsx/vlm/orze/src/orze/admin/ui/`
- Built dist: `/home/ec2-user/fsx/vlm/orze/src/orze/admin/ui/dist/`

## Known Failure Patterns
- [debugging.md](debugging.md) -- stale install path issue, restart procedures

## API Endpoints (8 verified)
- /api/status, /api/nodes, /api/runs, /api/alerts, /api/queue, /api/leaderboard, /api/ideas, /api/config
- POST endpoints: /api/ideas, /api/actions/stop, /api/actions/kill

## Asset Chunks (12 files as of v2.4.0)
- index-BE8hoqa3.js, index-DY-UcEz9.css, vendor-react, vendor-icons, vendor-motion
- Tab views: QueueTabView, AlertsTab, LeaderboardTab, NodesTab, OverviewTab, RunsTab, SettingsTab

## Process Management
- Orze runs from `/home/ec2-user/fsx/vlm/` with `orze -c orze.yaml`
- Admin panel on port 8787
- Venv: `/home/ec2-user/fsx/vlm/venv_38b/`
- Restart: `cd /home/ec2-user/fsx/vlm && source venv_38b/bin/activate && nohup orze -c orze.yaml > /tmp/orze.log 2>&1 &`
