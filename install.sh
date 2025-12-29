#!/usr/bin/env bash
set -euo pipefail

# One-shot, idempotent installer for Ubuntu 22.04/24.04
# - Supports local venv (default) or docker mode
# - Prompts for secrets + strategy/risk params (or non-interactive via env/defaults)
# - Writes .env (secrets) and patches config/config.yaml safely via Python (PyYAML) when available

############################
# Logging helpers
############################
log()  { echo -e "[install] $*"; }
warn() { echo -e "[install][WARN] $*" >&2; }
die()  { echo -e "[install][ERROR] $*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

############################
# Arg parsing
############################
NON_INTERACTIVE=0
ASSUME_YES=0
FORCE_TESTNET=""  # "true"|"false"|empty
MODE=""           # "venv"|"docker"|empty

usage() {
  cat <<'USAGE'
Usage: bash install.sh [--non-interactive] [--yes] [--venv|--docker] [--testnet true|false]

Flags:
  --non-interactive   Use env vars/defaults; fail if required secrets are missing.
  --yes               Auto-accept apt installs and (if docker mode) docker installation.
  --venv              Force local venv mode.
  --docker            Force docker mode.
  --testnet X         Override testnet toggle (true/false).

Non-interactive env vars (recommended):
  BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET

Optional env overrides for config (non-interactive):
  XS_TOP_N_BY_VOLUME, XS_REBALANCE_TIME_UTC
  XS_MAX_SPREAD_BPS, XS_MIN_ORDERBOOK_DEPTH_USD
  XS_DAILY_LOSS_LIMIT_PCT, XS_MAX_DRAWDOWN_PCT, XS_KILL_SWITCH_ENABLED
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --non-interactive) NON_INTERACTIVE=1; shift;;
    --yes) ASSUME_YES=1; shift;;
    --venv) MODE="venv"; shift;;
    --docker) MODE="docker"; shift;;
    --testnet)
      [[ $# -ge 2 ]] || die "--testnet requires true|false"
      FORCE_TESTNET="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1 (use --help)";;
  esac
done

############################
# Repo / project detection
############################
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=""

if [[ -f "$ROOT_DIR/pyproject.toml" && -d "$ROOT_DIR/src" ]]; then
  PROJECT_DIR="$ROOT_DIR"
elif [[ -f "$ROOT_DIR/bybit_xsreversal/pyproject.toml" ]]; then
  PROJECT_DIR="$ROOT_DIR/bybit_xsreversal"
else
  die "Could not detect project root. Expected pyproject.toml at repo root or bybit_xsreversal/pyproject.toml"
fi

CONFIG_PATH="$PROJECT_DIR/config/config.yaml"
ENV_PATH="$PROJECT_DIR/.env"
ENV_EXAMPLE_PATH="$PROJECT_DIR/.env.example"

############################
# OS detection
############################
OS_ID=""
OS_VERSION_ID=""
if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  OS_ID="${ID:-}"
  OS_VERSION_ID="${VERSION_ID:-}"
fi

if [[ "${OS_ID}" != "ubuntu" ]]; then
  die "Unsupported OS: ${OS_ID:-unknown}. This installer supports Ubuntu 22.04/24.04."
fi

if [[ "${OS_VERSION_ID}" != "22.04" && "${OS_VERSION_ID}" != "24.04" ]]; then
  warn "Detected Ubuntu ${OS_VERSION_ID}. Proceeding anyway, but only 22.04/24.04 are tested."
fi

############################
# Apt helpers
############################
apt_install() {
  local pkgs=("$@")
  local missing=()
  for p in "${pkgs[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi

  if [[ $ASSUME_YES -eq 1 ]]; then
    log "Installing packages: ${missing[*]}"
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
  else
    log "Missing packages: ${missing[*]}"
    read -r -p "Install missing packages via apt? [Y/n] " ans
    ans="${ans:-Y}"
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      sudo apt-get update
      sudo apt-get install "${missing[@]}"
    else
      die "Cannot proceed without required packages."
    fi
  fi
}

ensure_base_packages() {
  apt_install bash curl git python3 python3-venv python3-pip build-essential jq
}

############################
# Prompt helpers
############################
prompt() {
  # prompt <var_name> <message> <default>
  local __var="$1"
  local __msg="$2"
  local __def="$3"

  if [[ $NON_INTERACTIVE -eq 1 ]]; then
    printf -v "$__var" "%s" "${__def}"
    return 0
  fi

  local val
  read -r -p "$__msg [$__def]: " val
  val="${val:-$__def}"
  printf -v "$__var" "%s" "$val"
}

prompt_bool() {
  # prompt_bool <var_name> <message> <default true|false>
  local __var="$1"
  local __msg="$2"
  local __def="$3"

  if [[ $NON_INTERACTIVE -eq 1 ]]; then
    printf -v "$__var" "%s" "${__def}"
    return 0
  fi

  local suffix="y/N"
  [[ "$__def" == "true" ]] && suffix="Y/n"
  local ans
  read -r -p "$__msg [$suffix]: " ans
  ans="${ans:-}"
  if [[ -z "$ans" ]]; then
    printf -v "$__var" "%s" "$__def"
    return 0
  fi
  if [[ "$ans" =~ ^[Yy]$ ]]; then
    printf -v "$__var" "%s" "true"
  elif [[ "$ans" =~ ^[Nn]$ ]]; then
    printf -v "$__var" "%s" "false"
  else
    warn "Invalid input, using default: $__def"
    printf -v "$__var" "%s" "$__def"
  fi
}

prompt_secret() {
  # prompt_secret <var_name> <message>
  local __var="$1"
  local __msg="$2"

  if [[ $NON_INTERACTIVE -eq 1 ]]; then
    local v="${!__var:-}"
    [[ -n "$v" ]] || die "Missing required secret env var: $__var"
    return 0
  fi

  local val=""
  while [[ -z "$val" ]]; do
    read -r -s -p "$__msg: " val
    echo ""
    if [[ -z "$val" ]]; then
      warn "Value cannot be empty."
    fi
  done
  printf -v "$__var" "%s" "$val"
}

############################
# File helpers (idempotent)
############################
ensure_line_in_file() {
  local line="$1"
  local file="$2"
  touch "$file"
  grep -Fqx "$line" "$file" || echo "$line" >>"$file"
}

write_env_kv() {
  # write_env_kv <file> <KEY> <VALUE>
  local file="$1"
  local key="$2"
  local value="$3"
  touch "$file"
  if grep -E "^[[:space:]]*${key}=" "$file" >/dev/null 2>&1; then
    # shellcheck disable=SC2001
    sed -i -E "s|^[[:space:]]*${key}=.*|${key}=\"${value//\"/\\\"}\"|g" "$file"
  else
    echo "${key}=\"${value//\"/\\\"}\"" >>"$file"
  fi
}

############################
# Docker install (optional)
############################
ensure_docker() {
  if command -v docker >/dev/null 2>&1 && docker --version >/dev/null 2>&1; then
    return 0
  fi

  if [[ $ASSUME_YES -eq 1 ]]; then
    log "Installing Docker..."
  else
    read -r -p "Docker not found. Install Docker Engine + compose plugin? [Y/n] " ans
    ans="${ans:-Y}"
    [[ "$ans" =~ ^[Yy]$ ]] || die "Docker mode selected but docker is not installed."
  fi

  sudo apt-get update -y
  sudo apt-get install -y ca-certificates curl gnupg

  sudo install -m 0755 -d /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
  fi

  local arch
  arch="$(dpkg --print-architecture)"
  echo \
    "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  if ! groups | grep -q docker; then
    warn "Your user is not in the docker group. You may need: sudo usermod -aG docker $USER && re-login"
  fi
}

############################
# Mode selection + prompts
############################
ensure_base_packages
need_cmd python3
need_cmd git
need_cmd curl
need_cmd jq

if [[ -z "$MODE" ]]; then
  if [[ $NON_INTERACTIVE -eq 1 ]]; then
    MODE="venv"
  else
    log "Install mode:"
    echo "  [1] local venv (default)"
    echo "  [2] docker"
    read -r -p "Choose install mode [1/2]: " m
    m="${m:-1}"
    [[ "$m" == "2" ]] && MODE="docker" || MODE="venv"
  fi
fi

BYBIT_TESTNET="${BYBIT_TESTNET:-true}"
if [[ -n "$FORCE_TESTNET" ]]; then
  BYBIT_TESTNET="$FORCE_TESTNET"
fi
BYBIT_TESTNET="${BYBIT_TESTNET,,}"
if [[ "$BYBIT_TESTNET" != "true" && "$BYBIT_TESTNET" != "false" ]]; then
  warn "BYBIT_TESTNET must be true/false; defaulting to true"
  BYBIT_TESTNET="true"
fi
if [[ "$BYBIT_TESTNET" == "true" ]]; then
  BYBIT_TESTNET_PY="True"
else
  BYBIT_TESTNET_PY="False"
fi

if [[ $NON_INTERACTIVE -eq 0 ]]; then
  prompt_bool BYBIT_TESTNET "Use Bybit testnet?" "$BYBIT_TESTNET"
  BYBIT_TESTNET="${BYBIT_TESTNET,,}"
  if [[ "$BYBIT_TESTNET" != "true" && "$BYBIT_TESTNET" != "false" ]]; then
    warn "BYBIT_TESTNET must be true/false; defaulting to true"
    BYBIT_TESTNET="true"
  fi
  if [[ "$BYBIT_TESTNET" == "true" ]]; then
    BYBIT_TESTNET_PY="True"
  else
    BYBIT_TESTNET_PY="False"
  fi
fi

BYBIT_API_KEY="${BYBIT_API_KEY:-}"
BYBIT_API_SECRET="${BYBIT_API_SECRET:-}"
BYBIT_SUBACCOUNT="${BYBIT_SUBACCOUNT:-}"
POSITION_MODE="${POSITION_MODE:-one-way}"

if [[ $NON_INTERACTIVE -eq 1 ]]; then
  [[ -n "$BYBIT_API_KEY" ]] || die "BYBIT_API_KEY is required in --non-interactive mode"
  [[ -n "$BYBIT_API_SECRET" ]] || die "BYBIT_API_SECRET is required in --non-interactive mode"
else
  prompt BYBIT_API_KEY "Bybit API key" "${BYBIT_API_KEY:-}"
  [[ -n "$BYBIT_API_KEY" ]] || die "API key cannot be empty"
  prompt_secret BYBIT_API_SECRET "Bybit API secret"
  prompt BYBIT_SUBACCOUNT "Optional subaccount name (blank ok)" "${BYBIT_SUBACCOUNT:-}"
  prompt POSITION_MODE "Position mode preference (one-way recommended)" "$POSITION_MODE"
fi

# Baseline config used for optimization seed (most strategy knobs will be optimized).
TOP_N_BY_VOLUME="${XS_TOP_N_BY_VOLUME:-80}"
REBALANCE_TIME_UTC="${XS_REBALANCE_TIME_UTC:-00:05}"
MAX_SPREAD_BPS="${XS_MAX_SPREAD_BPS:-15}"
MIN_OB_DEPTH_USD="${XS_MIN_ORDERBOOK_DEPTH_USD:-50000}"

# Seeds used only if we must generate a brand new config (e.g., PyYAML missing in docker-mode host python).
SEED_LOOKBACK_DAYS="${XS_SEED_LOOKBACK_DAYS:-1}"
SEED_LONG_Q="${XS_SEED_LONG_Q:-0.1}"
SEED_SHORT_Q="${XS_SEED_SHORT_Q:-0.1}"
SEED_TARGET_GROSS_LEVERAGE="${XS_SEED_TARGET_GROSS_LEVERAGE:-1.0}"

DAILY_LOSS_LIMIT_PCT="${XS_DAILY_LOSS_LIMIT_PCT:-2.0}"
MAX_DRAWDOWN_PCT="${XS_MAX_DRAWDOWN_PCT:-20.0}"
KILL_SWITCH_ENABLED="${XS_KILL_SWITCH_ENABLED:-true}"
KILL_SWITCH_ENABLED="${KILL_SWITCH_ENABLED,,}"
if [[ "$KILL_SWITCH_ENABLED" != "true" && "$KILL_SWITCH_ENABLED" != "false" ]]; then
  warn "XS_KILL_SWITCH_ENABLED must be true/false; defaulting to true"
  KILL_SWITCH_ENABLED="true"
fi
if [[ "$KILL_SWITCH_ENABLED" == "true" ]]; then
  KILL_SWITCH_ENABLED_PY="True"
else
  KILL_SWITCH_ENABLED_PY="False"
fi

OPT_LEVEL="${XS_OPT_LEVEL:-quick}"
OPT_LEVEL="${OPT_LEVEL,,}"
if [[ "$OPT_LEVEL" != "quick" && "$OPT_LEVEL" != "standard" && "$OPT_LEVEL" != "deep" ]]; then
  OPT_LEVEL="quick"
fi
if [[ $NON_INTERACTIVE -eq 0 ]]; then
  echo "Optimization depth (how hard to search for parameters):"
  echo "  - quick    (recommended on install, ~1-5 min)"
  echo "  - standard (~5-20 min)"
  echo "  - deep     (~20-60+ min)"
  prompt OPT_LEVEL "Choose optimization depth (quick/standard/deep)" "$OPT_LEVEL"
  OPT_LEVEL="${OPT_LEVEL,,}"
  if [[ "$OPT_LEVEL" != "quick" && "$OPT_LEVEL" != "standard" && "$OPT_LEVEL" != "deep" ]]; then
    OPT_LEVEL="quick"
  fi
fi

############################
# Gitignore (idempotent)
############################
ensure_line_in_file ".env" "$ROOT_DIR/.gitignore"
ensure_line_in_file ".venv/" "$ROOT_DIR/.gitignore"
ensure_line_in_file "__pycache__/" "$ROOT_DIR/.gitignore"
ensure_line_in_file ".pytest_cache/" "$ROOT_DIR/.gitignore"
ensure_line_in_file "outputs/" "$ROOT_DIR/.gitignore"
ensure_line_in_file "data_cache/" "$ROOT_DIR/.gitignore"

############################
# .env.example + .env
############################
mkdir -p "$PROJECT_DIR"
if [[ ! -f "$ENV_EXAMPLE_PATH" ]]; then
  # Cannot include secrets here.
  cat >"$ENV_EXAMPLE_PATH" <<'ENVEX'
# Bybit API credentials (use a sub-account key with minimal permissions)
BYBIT_API_KEY="YOUR_KEY"
BYBIT_API_SECRET="YOUR_SECRET"
BYBIT_TESTNET="true"

# Optional metadata
BYBIT_SUBACCOUNT=""
POSITION_MODE="one-way"
ENVEX
  log "Created $ENV_EXAMPLE_PATH"
fi

if [[ ! -f "$ENV_PATH" ]]; then
  touch "$ENV_PATH"
  chmod 600 "$ENV_PATH" || true
  log "Created $ENV_PATH"
fi

write_env_kv "$ENV_PATH" "BYBIT_API_KEY" "$BYBIT_API_KEY"
write_env_kv "$ENV_PATH" "BYBIT_API_SECRET" "$BYBIT_API_SECRET"
write_env_kv "$ENV_PATH" "BYBIT_TESTNET" "$BYBIT_TESTNET"
write_env_kv "$ENV_PATH" "BYBIT_SUBACCOUNT" "$BYBIT_SUBACCOUNT"
write_env_kv "$ENV_PATH" "POSITION_MODE" "$POSITION_MODE"

############################
# Install dependencies (venv) OR ensure docker
############################
if [[ "$MODE" == "venv" ]]; then
  log "Setting up Python venv in $PROJECT_DIR/.venv"
  if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    python3 -m venv "$PROJECT_DIR/.venv"
  fi
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.venv/bin/activate"
  python -m pip install -U pip
  python -m pip install -e "$PROJECT_DIR"
elif [[ "$MODE" == "docker" ]]; then
  ensure_docker
  if [[ ! -f "$ROOT_DIR/docker-compose.yml" ]]; then
    warn "docker-compose.yml not found; expected at repo root. (We ship a minimal one in this repo.)"
  fi
else
  die "Unknown mode: $MODE"
fi

############################
# Patch config/config.yaml
############################
mkdir -p "$PROJECT_DIR/config"

patch_config_python() {
  local pybin="${1}"
  "$pybin" - <<PY
from __future__ import annotations

from pathlib import Path

cfg_path = Path(r"${CONFIG_PATH}")
cfg_path.parent.mkdir(parents=True, exist_ok=True)

payload = {
  "exchange": {
    "api_key_env": "BYBIT_API_KEY",
    "api_secret_env": "BYBIT_API_SECRET",
    "testnet": ${BYBIT_TESTNET_PY},
    "category": "linear",
  },
  "universe": {
    "top_n_by_volume": int(${TOP_N_BY_VOLUME}),
  },
  # signal/sizing fields will be optimized; seed sane defaults so the bot is runnable even if optimization is rejected
  "signal": {
    "lookback_days": int(${SEED_LOOKBACK_DAYS}),
    "long_quantile": float(${SEED_LONG_Q}),
    "short_quantile": float(${SEED_SHORT_Q}),
  },
  "rebalance": {
    "time_utc": "${REBALANCE_TIME_UTC}",
  },
  "sizing": {
    "target_gross_leverage": float(${SEED_TARGET_GROSS_LEVERAGE}),
  },
  "filters": {
    "max_spread_bps": float(${MAX_SPREAD_BPS}),
    "min_orderbook_depth_usd": float(${MIN_OB_DEPTH_USD}),
  },
  "risk": {
    "daily_loss_limit_pct": float(${DAILY_LOSS_LIMIT_PCT}),
    "max_drawdown_pct": float(${MAX_DRAWDOWN_PCT}),
    "kill_switch_enabled": ${KILL_SWITCH_ENABLED_PY},
  }
}

try:
  import yaml
except Exception as e:
  raise SystemExit(f"PyYAML missing: {e}")

existing = {}
if cfg_path.exists():
  existing = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
  if not isinstance(existing, dict):
    existing = {}

def deep_merge(d: dict, u: dict) -> dict:
  for k, v in u.items():
    if isinstance(v, dict) and isinstance(d.get(k), dict):
      deep_merge(d[k], v)
    else:
      d[k] = v
  return d

merged = deep_merge(existing, payload)
cfg_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
print(f"Patched {cfg_path}")
PY
}

if [[ -f "$CONFIG_PATH" ]]; then
  log "Updating existing config at $CONFIG_PATH"
else
  log "Creating config at $CONFIG_PATH"
fi

if [[ "$MODE" == "venv" ]]; then
  patch_config_python "$PROJECT_DIR/.venv/bin/python"
else
  # Docker mode: try system python (PyYAML might be missing); fallback to template if needed.
  if python3 -c "import yaml" >/dev/null 2>&1; then
    patch_config_python "python3"
  else
    warn "PyYAML not available on system python; generating fresh config.yaml (backup if exists)."
    if [[ -f "$CONFIG_PATH" ]]; then
      cp "$CONFIG_PATH" "$CONFIG_PATH.bak.$(date +%s)"
    fi
    cat >"$CONFIG_PATH" <<YAML
exchange:
  api_key_env: "BYBIT_API_KEY"
  api_secret_env: "BYBIT_API_SECRET"
  testnet: ${BYBIT_TESTNET,,}
  category: "linear"
universe:
  top_n_by_volume: ${TOP_N_BY_VOLUME}
signal:
  lookback_days: ${SEED_LOOKBACK_DAYS}
  long_quantile: ${SEED_LONG_Q}
  short_quantile: ${SEED_SHORT_Q}
rebalance:
  frequency: "daily"
  time_utc: "${REBALANCE_TIME_UTC}"
sizing:
  target_gross_leverage: ${SEED_TARGET_GROSS_LEVERAGE}
filters:
  max_spread_bps: ${MAX_SPREAD_BPS}
  min_orderbook_depth_usd: ${MIN_OB_DEPTH_USD}
risk:
  daily_loss_limit_pct: ${DAILY_LOSS_LIMIT_PCT}
  max_drawdown_pct: ${MAX_DRAWDOWN_PCT}
  kill_switch_enabled: ${KILL_SWITCH_ENABLED,,}
backtest:
  start_date: "2023-01-01"
  end_date: "2025-01-01"
  initial_equity: 10000.0
  taker_fee_bps: 6.0
  maker_fee_bps: 1.0
  slippage_bps: 3.0
  borrow_cost_bps: 0.0
  allow_partial_fills: true
  cache_dir: "data_cache"
YAML
  fi
fi

############################
# Ensure wrapper scripts are executable
############################
chmod +x "$ROOT_DIR/scripts/run_backtest.sh" "$ROOT_DIR/scripts/run_live.sh" || true
chmod +x "$ROOT_DIR/install.sh" || true

############################
# Optimization (instead of manual parameter input)
############################
log "Running parameter optimization (this will download/cached daily candles; first run can take a few minutes)..."
if [[ "$MODE" == "venv" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.venv/bin/activate"
  # Use a shorter rolling window on install to ensure enough symbols have history.
  export BYBIT_OPT_WINDOW_DAYS="${BYBIT_OPT_WINDOW_DAYS:-180}"
  # Reject negative Sharpe on install so we don't overwrite defaults with worse params.
  export BYBIT_OPT_MIN_SHARPE="${BYBIT_OPT_MIN_SHARPE:-0.0}"
  bybit-xsreversal --config "$CONFIG_PATH" optimize --level "$OPT_LEVEL"
else
  # Docker mode: run optimizer inside container using compose if available
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose run --rm bot bash -lc "export BYBIT_OPT_WINDOW_DAYS=${BYBIT_OPT_WINDOW_DAYS:-180} BYBIT_OPT_MIN_SHARPE=${BYBIT_OPT_MIN_SHARPE:-0.0} && pip install -e . && bybit-xsreversal --config config/config.yaml optimize --level ${OPT_LEVEL}"
  else
    warn "Docker compose not available; skipping optimization in docker mode."
  fi
fi

############################
# Optional systemd install
############################
SYSTEMD_INSTALL="false"
if [[ $NON_INTERACTIVE -eq 0 ]]; then
  prompt_bool SYSTEMD_INSTALL "Install systemd service (template provided under systemd/)?" "false"
fi

if [[ "$SYSTEMD_INSTALL" == "true" ]]; then
  local_unit="/etc/systemd/system/bybit_xsreversal.service"
  log "Installing systemd unit to $local_unit"
  sudo mkdir -p /etc/systemd/system
  sudo cp "$ROOT_DIR/systemd/bybit_xsreversal.service" "$local_unit"
  sudo systemctl daemon-reload
  if [[ $ASSUME_YES -eq 1 ]]; then
    sudo systemctl enable --now bybit_xsreversal.service || true
  else
    read -r -p "Enable and start service now? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      sudo systemctl enable --now bybit_xsreversal.service
    fi
  fi
fi

############################
# Summary
############################
cat <<SUMMARY

========================
INSTALL SUMMARY
========================
OS:                Ubuntu ${OS_VERSION_ID}
Project dir:        ${PROJECT_DIR}
Install mode:       ${MODE}
Config YAML:        ${CONFIG_PATH}
Env file:           ${ENV_PATH}  (chmod 600)
Testnet:            ${BYBIT_TESTNET}

Next commands:
  # Backtest
  ./scripts/run_backtest.sh

  # Live (dry-run first)
  ./scripts/run_live.sh --dry-run

Notes:
  - Live runs a daily scheduler; it will sleep until the next rebalance time (${REBALANCE_TIME_UTC} UTC).
  - To change parameters later, re-run: bash install.sh  (idempotent) or edit config/config.yaml.
SUMMARY


