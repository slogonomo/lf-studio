# Run it from NVIDIA Sync

NVIDIA Sync can start Sol-H3 Studio on your DGX Spark and forward its port to your laptop. Which command the
custom app runs depends on whether [spark-switchboard](https://github.com/slogonomo/spark-switchboard) manages
the studio on that Spark.

## Without spark-switchboard

Add a custom app in Sync with (field names can differ between Sync versions):

| Field | Value |
|---|---|
| Name | Sol-H3 Studio |
| Port | 8190 |
| Launch script / command | `bash ~/src/sol-h3-studio/bin/sol-h3-studio` |

Adjust the path if the repo is elsewhere. Start the app in Sync, wait for "Sol-H3 Studio ready", then open
<http://localhost:8190> on the laptop. The launcher's `>>> STATUS:` lines show its progress: checks, start-up,
refusals (memory, port), and the memory guard.

How the launcher behaves under Sync:

- It runs in the foreground over Sync's SSH session and listens on 127.0.0.1:8190 on the Spark; Sync forwards the
  port, so nothing is exposed on the LAN.
- Closing the app in Sync stops it: the launcher handles SIGTERM and SIGHUP and stops the server, which closes its
  pipeline; then the launcher stops leftover workers and the Qwen container. The server runs in its own session,
  so a signal to the launcher's process group does not cut it short. If Sync only kills its wrapper script, the
  launcher notices that its parent went away and shuts down the same way.
- Sync's SSH login can be older than your docker group membership; the launcher then uses `sg docker`
  (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#the-server-wont-start)).
- It needs ~110 GiB free. If another Sync app holds the memory, the start is refused with a list of the biggest
  users; close that app first.

## With spark-switchboard

When switchboard manages the studio, switchboard owns port 8190 and starts the studio on demand, so
`bin/sol-h3-studio` started from Sync is refused ("port 8190 is already in use"). Point the custom app at
switchboard's `delegate.sh` instead:

| Field | Value |
|---|---|
| Name | Sol-H3 Studio |
| Port | 8190 |
| Launch script / command | `bash ~/src/spark-switchboard/examples/launchers/delegate.sh sol-h3-studio` |

`sol-h3-studio` is the service name (the file name of `integrations/spark-switchboard/sol-h3-studio.toml` without
`.toml`). `delegate.sh` asks switchboard to start the studio, shows switchboard's progress as `>>> STATUS:` lines
and stays in the foreground, so Sync keeps the port forwarded. When you close the app in Sync, it asks switchboard
to stop the studio unless someone used it recently (`stop?if_idle=1`). It never starts the studio itself; if
switchboard is not running, it says so and exits, and you use the standalone command above.
