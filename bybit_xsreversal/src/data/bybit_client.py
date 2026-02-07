from __future__ import annotations

import hmac
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter


class BybitAPIError(RuntimeError):
    def __init__(self, message: str, *, ret_code: int | None = None, ret_msg: str | None = None) -> None:
        super().__init__(message)
        self.ret_code = ret_code
        self.ret_msg = ret_msg


@dataclass(frozen=True)
class BybitAuth:
    api_key: str
    api_secret: str


def _ts_ms() -> str:
    return str(int(time.time() * 1000))


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


class BybitClient:
    """
    Minimal, production-oriented Bybit v5 REST client with:
    - auth signing (SIGN-TYPE=2)
    - retries (tenacity) + basic rate limit backoff
    - consistent error handling
    """

    def __init__(
        self,
        *,
        auth: BybitAuth | None,
        testnet: bool,
        recv_window_ms: int = 5000,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._auth = auth
        self._recv_window = str(recv_window_ms)
        self._base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout_seconds)

    def close(self) -> None:
        self._http.close()

    def _headers(self, *, payload: str) -> dict[str, str]:
        if self._auth is None:
            return {"Content-Type": "application/json"}
        ts = _ts_ms()
        to_sign = ts + self._auth.api_key + self._recv_window + payload
        sig = _sign(self._auth.api_secret, to_sign)
        return {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self._auth.api_key,
            "X-BAPI-SIGN": sig,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": self._recv_window,
        }

    def _raise_if_error(self, data: dict[str, Any]) -> None:
        ret_code = int(data.get("retCode", -1))
        if ret_code == 0:
            return
        ret_msg = str(data.get("retMsg", ""))
        raise BybitAPIError(f"Bybit error retCode={ret_code} retMsg={ret_msg}", ret_code=ret_code, ret_msg=ret_msg)

    def _sleep_rate_limit(self, resp: httpx.Response) -> None:
        """
        Bybit often returns retCode=10006 for rate limiting and may include reset timestamp headers.
        Prefer sleeping until reset rather than a fixed delay.
        """
        reset_hdr = None
        for k in ("X-Bapi-Limit-Reset-Timestamp", "X-BAPI-LIMIT-RESET-TIMESTAMP", "x-bapi-limit-reset-timestamp"):
            if k in resp.headers:
                reset_hdr = resp.headers.get(k)
                break
        now_ms = int(time.time() * 1000)
        sleep_s = 1.5
        if reset_hdr:
            try:
                reset_val = int(float(str(reset_hdr)))
                # Heuristic: if seconds epoch, convert to ms
                if reset_val < 10_000_000_000:
                    reset_val *= 1000
                delta_ms = max(0, reset_val - now_ms)
                # Add a small safety buffer
                sleep_s = max(1.0, delta_ms / 1000.0 + 0.25)
            except Exception:
                sleep_s = 1.5
        logger.warning("Bybit rate limit backoff: sleeping {:.2f}s", sleep_s)
        time.sleep(sleep_s)

    @retry(
        retry=retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPError)
            or (isinstance(e, BybitAPIError) and int(e.ret_code or -1) in (10006,))
        ),
        wait=wait_exponential_jitter(initial=0.5, max=10.0),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _request(self, method: Literal["GET", "POST"], path: str, params: dict[str, Any] | None, body: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        body = body or {}

        if method == "GET":
            # IMPORTANT: Bybit signature validation is sensitive to the exact query-string.
            # Sign the *same ordered param sequence* that we send over the wire.
            # (Sorting can cause "Error sign" if httpx encodes params in insertion order.)
            pairs = [(k, str(v)) for k, v in params.items() if v is not None]
            query = urlencode(pairs)
            headers = self._headers(payload=query)
            resp = self._http.get(path, params=pairs, headers=headers)
        else:
            # Bybit signs the raw JSON string (no whitespace). httpx will json-dump similarly, but we control it.
            import json

            payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
            headers = self._headers(payload=payload)
            resp = self._http.post(path, content=payload.encode("utf-8"), params=params, headers=headers)

        if resp.status_code == 429:
            # HTTP-level rate limit
            self._sleep_rate_limit(resp)
            raise httpx.HTTPStatusError(f"Bybit 429 rate limit", request=resp.request, response=resp)

        if resp.status_code >= 500:
            raise httpx.HTTPStatusError(f"Bybit 5xx: {resp.status_code}", request=resp.request, response=resp)

        data = resp.json()
        try:
            self._raise_if_error(data)
        except BybitAPIError as e:
            # 10006: rate limit
            if e.ret_code == 10006:
                self._sleep_rate_limit(resp)
            raise
        return data

    # -------- Market data --------
    def get_tickers(self, *, category: str = "linear") -> list[dict[str, Any]]:
        data = self._request("GET", "/v5/market/tickers", {"category": category}, None)
        return list((data.get("result") or {}).get("list") or [])

    def get_kline(
        self,
        *,
        category: str,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[str]]:
        params: dict[str, Any] = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)
        data = self._request("GET", "/v5/market/kline", params, None)
        return list((data.get("result") or {}).get("list") or [])

    def get_orderbook(self, *, category: str, symbol: str, limit: int = 50) -> dict[str, Any]:
        data = self._request("GET", "/v5/market/orderbook", {"category": category, "symbol": symbol, "limit": limit}, None)
        return dict(data.get("result") or {})

    def get_instruments_info(self, *, category: str, symbol: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"category": category, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        data = self._request("GET", "/v5/market/instruments-info", params, None)
        return list((data.get("result") or {}).get("list") or [])

    def get_funding_history(
        self,
        *,
        category: str,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Funding history endpoint (USDT linear perps).
        Bybit v5: /v5/market/funding/history
        """
        params: dict[str, Any] = {"category": category, "symbol": symbol, "limit": limit}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        data = self._request("GET", "/v5/market/funding/history", params, None)
        return list((data.get("result") or {}).get("list") or [])

    # -------- Account/positions/orders (auth required) --------
    def get_wallet_balance(self, *, account_type: str = "UNIFIED") -> dict[str, Any]:
        if self._auth is None:
            raise ValueError("Auth required for wallet balance.")
        data = self._request("GET", "/v5/account/wallet-balance", {"accountType": account_type}, None)
        return dict(data.get("result") or {})

    def get_positions(self, *, category: str = "linear", settle_coin: str = "USDT", limit: int | None = None) -> list[dict[str, Any]]:
        if self._auth is None:
            raise ValueError("Auth required for positions.")
        # Bybit v5 position/list supports cursor-based pagination
        # Fetch all positions using pagination to avoid missing any
        all_positions: list[dict[str, Any]] = []
        cursor: str | None = None
        
        while True:
            params: dict[str, Any] = {"category": category, "settleCoin": settle_coin, "limit": 50}
            if cursor:
                params["cursor"] = cursor
            
            data = self._request("GET", "/v5/position/list", params, None)
            result_data = data.get("result") or {}
            chunk = list(result_data.get("list") or [])
            all_positions.extend(chunk)
            
            # Check for next page
            cursor = result_data.get("nextPageCursor") or None
            if not cursor or len(chunk) == 0:
                break
            
            # Safety limit to prevent infinite loops
            if len(all_positions) > 1000:
                logger.warning("Position list exceeded 1000 items, stopping pagination")
                break
        
        logger.debug("Fetched {} positions via pagination ({} pages)", len(all_positions), "multiple" if cursor else "single")
        return all_positions

    def set_leverage(self, *, category: str, symbol: str, leverage: str) -> None:
        if self._auth is None:
            raise ValueError("Auth required for set_leverage.")
        body = {"category": category, "symbol": symbol, "buyLeverage": leverage, "sellLeverage": leverage}
        try:
            self._request("POST", "/v5/position/set-leverage", None, body)
        except BybitAPIError as e:
            # 110043: leverage not modified (typically means it's already set to requested value)
            if int(e.ret_code or -1) == 110043:
                return
            raise

    def create_order(self, *, category: str, order: dict[str, Any]) -> dict[str, Any]:
        if self._auth is None:
            raise ValueError("Auth required for create_order.")
        body = {"category": category, **order}
        data = self._request("POST", "/v5/order/create", None, body)
        return dict(data.get("result") or {})

    def cancel_order(self, *, category: str, symbol: str, order_id: str) -> None:
        if self._auth is None:
            raise ValueError("Auth required for cancel_order.")
        body = {"category": category, "symbol": symbol, "orderId": order_id}
        try:
            self._request("POST", "/v5/order/cancel", None, body)
        except BybitAPIError as e:
            # Non-fatal: order is already filled/canceled, or too late to cancel.
            # Don't retry / don't fail the whole rebalance.
            if int(e.ret_code or -1) == 110001:
                return
            raise

    def cancel_all_orders(
        self,
        *,
        category: str,
        symbol: str | None = None,
        settle_coin: str | None = "USDT",
        base_coin: str | None = None,
        order_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Cancel orders for the account. Bybit v5 supports scoping by symbol / settleCoin / baseCoin and
        (optionally) orderFilter to cancel normal orders vs conditional/stop orders.

        This is the most reliable way to ensure a clean slate before rebalancing on a dedicated sub-account.
        """
        if self._auth is None:
            raise ValueError("Auth required for cancel_all_orders.")
        body: dict[str, Any] = {"category": category}
        if symbol:
            body["symbol"] = symbol
        if settle_coin:
            body["settleCoin"] = settle_coin
        if base_coin:
            body["baseCoin"] = base_coin
        if order_filter:
            body["orderFilter"] = order_filter
        data = self._request("POST", "/v5/order/cancel-all", None, body)
        return dict(data.get("result") or {})

    def get_open_orders(
        self,
        *,
        category: str,
        symbol: str | None = None,
        settle_coin: str | None = None,
        base_coin: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._auth is None:
            raise ValueError("Auth required for open orders.")
        # Bybit pagination uses nextPageCursor.
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(20):  # hard cap
            params: dict[str, Any] = {"category": category, "openOnly": 1, "limit": 50}
            if cursor:
                params["cursor"] = cursor
            if symbol:
                params["symbol"] = symbol
            else:
                # Bybit requires one of: symbol / settleCoin / baseCoin for order/realtime.
                # Default to USDT-settled perps when not specifying a symbol.
                if settle_coin is None and base_coin is None:
                    settle_coin = "USDT"
            if settle_coin:
                params["settleCoin"] = settle_coin
            if base_coin:
                params["baseCoin"] = base_coin

            data = self._request("GET", "/v5/order/realtime", params, None)
            res = data.get("result") or {}
            out.extend(list(res.get("list") or []))
            cursor = res.get("nextPageCursor") or None
            if not cursor:
                break
        return out


