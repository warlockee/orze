# Debugging Notes

## 2026-02-28: Stale Install Path Causing 404s on All Assets

### Symptom
`QueueTabView-D3DJBqxT.js` returned 404, along with ALL other asset chunks.
Error in browser: `TypeError: Failed to fetch dynamically imported module`

### Root Cause
The running orze process (started before the editable install was configured) resolved
`Path(__file__).parent / "ui" / "dist"` to the stale `lib64` path:
`/home/ec2-user/fsx/vlm/venv_38b/lib64/python3.9/site-packages/orze/admin/ui/dist`

That directory did NOT exist because the package was installed as editable (only dist-info
and a .pth file were in site-packages, no actual source files).

The editable install's .pth file correctly pointed to `/home/ec2-user/fsx/vlm/orze/src`,
but the already-running process had imported the module from the old non-editable install path.

### Fix
1. Clean reinstall: `rm -rf dist build *.egg-info && pip install -e ".[admin]"`
2. Kill old orze process: `kill -9 <PID>`
3. Restart orze: `cd /home/ec2-user/fsx/vlm && nohup orze -c orze.yaml > /tmp/orze.log 2>&1 &`

### Key Diagnostic Commands
- Check startup log: `head -20 /tmp/orze.log | grep "UI mounted"`
  - Should show source path, not lib64/site-packages path
- Verify module resolution: `python3 -c "import orze.admin.server as s; print(s._ui_dist)"`
- Check editable path: `cat venv_38b/lib/python3.9/site-packages/__editable__.orze-*.pth`

### Lesson
After any `pip install -e`, ALWAYS restart the running orze process. The old process
retains stale `__file__` paths from the previous import. This is not detectable by
checking the install -- only by checking the running process's startup log.
