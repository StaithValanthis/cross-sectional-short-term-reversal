from __future__ import annotations

import hmac
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


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

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, BybitAPIError)),
        wait=wait_exponential_jitter(initial=0.5, max=10.0),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _request(self, method: Literal["GET", "POST"], path: str, params: dict[str, Any] | None, body: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        body = body or {}

        if method == "GET":
            query = urlencode(sorted([(k, str(v)) for k, v in params.items() if v is not None]))
            payload = query
            headers = self._headers(payload=payload)
            resp = self._http.get(path, params=params, headers=headers)
        else:
            # Bybit signs the raw JSON string (no whitespace). httpx will json-dump similarly, but we control it.
            import json

            payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
            headers = self._headers(payload=payload)
            resp = self._http.post(path, content=payload.encode("utf-8"), params=params, headers=headers)

        if resp.status_code >= 500:
            raise httpx.HTTPStatusError(f"Bybit 5xx: {resp.status_code}", request=resp.request, response=resp)

        data = resp.json()
        try:
            self._raise_if_error(data)
        except BybitAPIError as e:
            # 10006: rate limit
            if e.ret_code == 10006:
                logger.warning("Bybit rate limit hit (10006). Backing off then retrying...")
                time.sleep(1.2)
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

    # -------- Account/positions/orders (auth required) --------
    def get_wallet_balance(self, *, account_type: str = "UNIFIED") -> dict[str, Any]:
        if self._auth is None:
            raise ValueError("Auth required for wallet balance.")
        data = self._request("GET", "/v5/account/wallet-balance", {"accountType": account_type}, None)
        return dict(data.get("result") or {})

    def get_positions(self, *, category: str = "linear", settle_coin: str = "USDT") -> list[dict[str, Any]]:
        if self._auth is None:
            raise ValueError("Auth required for positions.")
        data = self._request("GET", "/v5/position/list", {"category": category, "settleCoin": settle_coin}, None)
        return list((data.get("result") or {}).get("list") or [])

    def set_leverage(self, *, category: str, symbol: str, leverage: str) -> None:
        if self._auth is None:
            raise ValueError("Auth required for set_leverage.")
        body = {"category": category, "symbol": symbol, "buyLeverage": leverage, "sellLeverage": leverage}
        self._request("POST", "/v5/position/set-leverage", None, body)

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
        self._request("POST", "/v5/order/cancel", None, body)

    def get_open_orders(self, *, category: str, symbol: str | None = None) -> list[dict[str, Any]]:
        if self._auth is None:
            raise ValueError("Auth required for open orders.")
        params: dict[str, Any] = {"category": category, "openOnly": 1, "limit": 50}
        if symbol:
            params["symbol"] = symbol
        data = self._request("GET", "/v5/order/realtime", params, None)
        return list((data.get("result") or {}).get("list") or [])


