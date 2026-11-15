#!/usr/bin/env bash
# Consistent backup of the compose stack's persistent data, with retention.
#
# Usage: scripts/backup-volumes.sh [options]
#   -d, --dest DIR        where backups go (default: $BACKUP_DIR or ./backups)
#   -k, --keep N          keep the newest N complete backups (default: 7)
#   -f, --compose-file F  compose file; repeat for overlays (default: $COMPOSE_FILE
#                         split on ':' if set, else docker-compose.yml). On the VPS:
#                         -f docker-compose.yml -f docker-compose.vps.yml
#   -p, --project NAME    compose project name (default: compose's own default)
#       --db NAME         Neo4j database to dump (default: $NEO4J_DATABASE or memory)
#       --skip LIST       comma list of parts to skip: neo4j,qdrant,redis,files
#       --quiesce         stop the API and all three workers for the whole run and
#                         start them again on exit (also on failure). Without it
#                         the parts are taken one after another, each consistent
#                         on its own, but NOT as of one instant across stores.
#   -h, --help            this text
#
# What each persistent volume gets (compose volume -> method):
#   neo4j_data     neo4j-admin database dump. Neo4j Community has no online backup,
#                  so the neo4j container is STOPPED for the dump (about a minute for
#                  a small graph) and started again, even if the dump fails. Graph
#                  reads and writes fail while it is down; run this off-peak. Only
#                  the $NEO4J_DATABASE database is dumped (not `system`: users/roles
#                  are recreated from NEO4J_AUTH on a fresh volume).
#   qdrant_data    Qdrant snapshot API, one snapshot per collection, downloaded and
#                  then deleted server side. Online, consistent per collection.
#   redis_data     BGSAVE, wait for it to finish, copy dump.rdb. Online. Redis holds
#                  the ARQ job queues and task status only, so this is optional for
#                  recovery of memories (they live in Qdrant and Neo4j).
#   ingest_data,   tar of /data/ingest and /data/exemplars taken through the API
#   exemplar_data  service's mounts. Online; a file written during the tar may be
#                  missing from it (the originals are also kept in the vector store).
# Not covered: the Obsidian vault bind mount (a host directory, OBSIDIAN_VAULT_PATH):
# back it up with your normal host tooling.
#
# Consistency: each store is internally consistent (offline dump, Qdrant snapshot,
# Redis BGSAVE, tar). They are taken minutes apart, so with the app running a write
# can land in one store's copy and not another's. For a cross-store point in time use
# --quiesce (the app is unavailable while it runs); without it, pick a quiet hour.
# Only one run per DEST at a time (a lock directory in DEST refuses a second).
#
# Output: DEST/ns-backup-<UTC stamp>/{neo4j,qdrant,redis,files}/ + MANIFEST.txt +
# SHA256SUMS. It is built in ns-backup-<stamp>.partial and renamed only when every
# selected part succeeded, so a failed run never looks like a good backup and is
# never counted by retention. Everything is created 0700 for the invoking user.
#
# Needs: docker with the compose plugin, run from anywhere (it cd's to the repo
# root so .env is found), the stack up (redis, qdrant, neo4j, and the API image
# present for the qdrant/files parts). It touches only the compose project it is
# pointed at.
#
# RESTORE (stop the app first; replace $B with the backup directory):
#   B=/path/to/backups/ns-backup-<stamp>;  (cd "$B" && sha256sum -c SHA256SUMS)
#   dc="docker compose -f docker-compose.yml"      # same -f / -p flags as the backup
#   $dc stop neuralscape neuralscape-worker neuralscape-graph-worker neuralscape-ingest-worker
#   Neo4j:   $dc stop neo4j
#            $dc run --rm --no-deps -T -v "$B/neo4j:/backups" neo4j \
#                neo4j-admin database load memory --from-path=/backups --overwrite-destination=true
#            $dc start neo4j
#   Qdrant:  for each collection file listed in $B/qdrant/index.json:
#            curl -X POST "http://127.0.0.1:6333/collections/<name>/snapshots/upload?priority=snapshot" \
#                 -H 'Content-Type:multipart/form-data' -F "snapshot=@$B/qdrant/<file>"
#            (where 6333 is not published, e.g. the VPS overlay, run the same upload with
#             `$dc run --rm --no-deps -T -v "$B/qdrant:/backups" neuralscape python ...`
#             against http://qdrant:6333, or temporarily publish it on 127.0.0.1.)
#   Redis:   $dc stop redis; $dc cp "$B/redis/dump.rdb" redis:/data/dump.rdb; $dc start redis
#   Files:   $dc run --rm --no-deps -T -i --entrypoint tar neuralscape -C /data -xzf - \
#                < "$B/files/ingest-exemplars.tar.gz"
#   Then: $dc up -d
# Test a restore on a scratch compose project (-p scratch) before you need it.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${BACKUP_DIR:-$ROOT/backups}"
KEEP=7
PROJECT=""
DB="${NEO4J_DATABASE:-memory}"
SKIP=""
QUIESCE=0
FILES=()

usage() { sed -n '2,/^# Test a restore/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "backup-volumes: $*" >&2; exit 1; }
log() { echo "backup-volumes: $*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    -d|--dest) DEST="${2:?--dest needs a directory}"; shift 2 ;;
    -k|--keep) KEEP="${2:?--keep needs a number}"; shift 2 ;;
    -f|--compose-file) FILES+=("${2:?--compose-file needs a file}"); shift 2 ;;
    -p|--project) PROJECT="${2:?--project needs a name}"; shift 2 ;;
    --db) DB="${2:?--db needs a name}"; shift 2 ;;
    --skip) SKIP="${2:?--skip needs a list}"; shift 2 ;;
    --quiesce) QUIESCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

case "$KEEP" in ''|*[!0-9]*) die "--keep must be a whole number" ;; esac
[ "$KEEP" -ge 1 ] || die "--keep must be at least 1"
if [ ${#FILES[@]} -eq 0 ]; then
  if [ -n "${COMPOSE_FILE:-}" ]; then IFS=':' read -r -a FILES <<<"$COMPOSE_FILE"; else FILES=(docker-compose.yml); fi
fi
for part in ${SKIP//,/ }; do
  case "$part" in neo4j|qdrant|redis|files) ;; *) die "unknown --skip part: $part" ;; esac
done
skipped() { case ",$SKIP," in *",$1,"*) return 0 ;; esac; return 1; }

command -v docker >/dev/null 2>&1 || die "docker not found"
cd "$ROOT"
DC=(docker compose)
[ -z "$PROJECT" ] || DC+=(-p "$PROJECT")
for f in "${FILES[@]}"; do DC+=(-f "$f"); done
dc() { "${DC[@]}" "$@"; }

umask 077
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
FINAL="$DEST/ns-backup-$STAMP"
WORK="$FINAL.partial"


LOCK="$DEST/.backup.lock"
LOCK_HELD=0
NEO4J_DOWN=0
APP_STOPPED=()
APP_SERVICES=(neuralscape neuralscape-worker neuralscape-graph-worker neuralscape-ingest-worker)
cleanup() {
  status=$?
  if [ "$NEO4J_DOWN" = 1 ]; then
    log "restarting neo4j"
    dc up -d --no-deps --wait --wait-timeout 300 neo4j >&2 || log "WARNING: neo4j did not come back; start it by hand: docker compose start neo4j"
  fi
  if [ "${#APP_STOPPED[@]}" -gt 0 ]; then
    log "starting ${APP_STOPPED[*]}"
    dc start "${APP_STOPPED[@]}" >&2 || log "WARNING: app services did not all start; run: docker compose start ${APP_STOPPED[*]}"
  fi
  if [ "$LOCK_HELD" = 1 ]; then rm -rf -- "$LOCK"; fi
  if [ "$status" -ne 0 ]; then
    if [ -d "$WORK" ]; then log "FAILED (exit $status); partial output left in $WORK and NOT counted as a backup"; else log "FAILED (exit $status)"; fi
  fi
}
trap cleanup EXIT

# One run at a time per DEST: two runs would both stop neo4j and race on the dump.
acquire_lock() {
  if mkdir "$LOCK" 2>/dev/null; then
    echo $$ > "$LOCK/pid"; LOCK_HELD=1; return 0
  fi
  other="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
    die "another backup (pid $other) is running against $DEST; lock: $LOCK"
  fi
  log "removing stale lock (pid ${other:-unknown} is gone)"
  rm -rf -- "$LOCK"
  mkdir "$LOCK" 2>/dev/null || die "could not take the lock $LOCK"
  echo $$ > "$LOCK/pid"; LOCK_HELD=1
}
acquire_lock

[ ! -e "$FINAL" ] && [ ! -e "$WORK" ] || die "$FINAL already exists; wait a second and retry"
mkdir -p "$WORK"

# Containers that the parts below talk to must be running (neo4j may be stopped:
# it is stopped for the dump anyway, but then the dump has nothing recent to read).
running() { [ -n "$(dc ps --status running -q "$1" 2>/dev/null)" ]; }
for svc in redis qdrant neo4j; do
  skipped "$svc" && continue
  running "$svc" || die "service '$svc' is not running in this compose project (use --skip $svc to leave it out)"
done

if [ "$QUIESCE" = 1 ]; then
  log "quiesce: stopping ${APP_SERVICES[*]}"
  for svc in "${APP_SERVICES[@]}"; do
    if running "$svc"; then APP_STOPPED+=("$svc"); fi
  done
  [ "${#APP_STOPPED[@]}" -eq 0 ] || dc stop "${APP_STOPPED[@]}" >&2
fi

PARTS=()

# ── Redis: BGSAVE, wait, copy ───────────────────────────────────────
if ! skipped redis; then
  log "redis: BGSAVE"
  mkdir -p "$WORK/redis"
  rcli() { dc exec -T redis redis-cli "$@" | tr -d '\r'; }
  # LASTSAVE has 1 s resolution (a save right after the previous one looks like no
  # save), so watch INFO persistence instead: BGSAVE sets rdb_bgsave_in_progress:1
  # before it replies, and 0 with status ok means that save is finished.
  started=0
  for _ in $(seq 1 60); do
    out="$(rcli BGSAVE || true)"
    case "$out" in
      *"Background saving started"*) started=1; break ;;
      *"already in progress"*|*"scheduled"*)
        for _ in $(seq 1 120); do
          rcli INFO persistence | grep -q '^rdb_bgsave_in_progress:1' || break
          sleep 1
        done ;;
      *) die "redis BGSAVE refused: $out" ;;
    esac
  done
  [ "$started" = 1 ] || die "redis BGSAVE could not be started"
  done_ok=0
  for _ in $(seq 1 600); do
    info="$(rcli INFO persistence)"
    if ! grep -q '^rdb_bgsave_in_progress:1' <<<"$info"; then done_ok=1; break; fi
    sleep 1
  done
  [ "$done_ok" = 1 ] || die "redis BGSAVE did not finish within 600s"
  grep -q '^rdb_last_bgsave_status:ok' <<<"$info" || die "redis BGSAVE finished with an error"
  dc cp redis:/data/dump.rdb "$WORK/redis/dump.rdb"
  [ -s "$WORK/redis/dump.rdb" ] || die "redis dump.rdb is empty"
  PARTS+=(redis)
fi

# ── Qdrant: one snapshot per collection ─────────────────────────────
if ! skipped qdrant; then
  log "qdrant: snapshot per collection"
  # 0777 so the container user (not the host user) can write; the parent is 0700.
  mkdir -m 0777 "$WORK/qdrant"
  dc run --rm --no-deps -T -v "$WORK/qdrant:/backups" neuralscape python - <<'PY'
import json, os, shutil, sys, urllib.parse, urllib.request

base = os.environ["QDRANT_URL"].rstrip("/")
out = "/backups"


def req(method, path):
    return urllib.request.Request(base + path, method=method, data=b"" if method == "POST" else None)


with urllib.request.urlopen(req("GET", "/collections"), timeout=60) as r:
    names = [c["name"] for c in json.load(r)["result"]["collections"]]
index = {}
for n in names:
    q = urllib.parse.quote(n, safe="")
    with urllib.request.urlopen(req("POST", f"/collections/{q}/snapshots"), timeout=600) as r:
        snap = json.load(r)["result"]["name"]
    fname = f"{q}.{snap}"
    with urllib.request.urlopen(req("GET", f"/collections/{q}/snapshots/{urllib.parse.quote(snap, safe='')}"), timeout=600) as r, open(os.path.join(out, fname), "wb") as f:
        shutil.copyfileobj(r, f)
    # Free the server-side copy: snapshots live on the qdrant volume and add up.
    urllib.request.urlopen(req("DELETE", f"/collections/{q}/snapshots/{urllib.parse.quote(snap, safe='')}"), timeout=60).close()
    if os.path.getsize(os.path.join(out, fname)) == 0:
        sys.exit(f"empty snapshot for collection {n}")
    index[n] = fname
with open(os.path.join(out, "index.json"), "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)
print(f"qdrant: {len(index)} collection snapshot(s)", file=sys.stderr)
PY
  [ -s "$WORK/qdrant/index.json" ] || die "qdrant snapshot step produced no index"
  PARTS+=(qdrant)
fi

# ── Ingest + exemplar volumes ───────────────────────────────────────
if ! skipped files; then
  log "files: ingest_data + exemplar_data"
  mkdir -p "$WORK/files"
  dc run --rm --no-deps -T --entrypoint tar neuralscape -C /data -czf - ingest exemplars > "$WORK/files/ingest-exemplars.tar.gz"
  tar -tzf "$WORK/files/ingest-exemplars.tar.gz" >/dev/null || die "files archive is not a readable tar.gz"
  PARTS+=(files)
fi

# ── Neo4j: offline dump (last: it is the only part with downtime) ───
if ! skipped neo4j; then
  log "neo4j: stopping for an offline dump of '$DB' (graph unavailable until it restarts)"
  mkdir -m 0777 "$WORK/neo4j"
  dc stop neo4j >&2
  NEO4J_DOWN=1
  dc run --rm --no-deps -T -v "$WORK/neo4j:/backups" neo4j \
    neo4j-admin database dump "$DB" --to-path=/backups >&2
  [ -s "$WORK/neo4j/$DB.dump" ] || die "neo4j dump file $DB.dump missing or empty"
  dc up -d --no-deps --wait --wait-timeout 300 neo4j >&2
  NEO4J_DOWN=0
  PARTS+=(neo4j)
fi

[ ${#PARTS[@]} -gt 0 ] || die "nothing to back up (everything skipped)"

# ── Manifest + checksums, then publish ──────────────────────────────
{
  echo "created_utc=$STAMP"
  echo "parts=${PARTS[*]}"
  echo "neo4j_database=$DB"
  echo "compose_files=${FILES[*]}"
  echo "compose_project=${PROJECT:-default}"
  echo "git_head=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$WORK/MANIFEST.txt"
if command -v sha256sum >/dev/null 2>&1; then SUM=(sha256sum); else SUM=(shasum -a 256); fi
(cd "$WORK" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | while IFS= read -r f; do "${SUM[@]}" "$f"; done > SHA256SUMS)
[ -s "$WORK/SHA256SUMS" ] || die "checksum file is empty"
mv "$WORK" "$FINAL"
log "backup complete: $FINAL (${PARTS[*]})"

# ── Retention: newest KEEP complete backups ─────────────────────────
n=0
find "$DEST" -maxdepth 1 -type d -name 'ns-backup-[0-9]*T[0-9]*Z' | LC_ALL=C sort -r | while IFS= read -r d; do
  n=$((n + 1))
  if [ "$n" -gt "$KEEP" ]; then
    log "retention: removing $d"
    rm -rf -- "$d"
  fi
done
echo "$FINAL"
