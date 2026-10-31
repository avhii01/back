"""HTTP client for the SIMULATED payment / inventory / shipping services.

Responsibilities: timeouts, correlation + idempotency headers, strict response validation
(service output is untrusted), and mapping transport/HTTP failures onto the error taxonomy:

    connect error / 502 / 503      -> Transient  (request did not take effect)
    read timeout / 504 (mutation)  -> Ambiguous  (may or may not have taken effect)
    4xx                            -> Permanent
"""
from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from relay.errors import AmbiguousToolError, PermanentToolError, TransientToolError
from relay.untrusted import sanitize_text


class _Resp(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PaymentRecord(_Resp):
    id: str = Field(max_length=60)
    amount_cents: int
    currency: str = "USD"
    status: str = Field(max_length=20)
    memo: str | None = None


class PaymentEventRecord(_Resp):
    payment_id: str | None = None
    type: str = Field(max_length=20)
    amount_cents: int
    source: str = "webhook"
    memo: str | None = None


class RefundRecord(_Resp):
    id: str
    payment_id: str
    amount_cents: int
    replayed: bool = False


class PaymentsResponse(_Resp):
    order_id: str
    payments: list[PaymentRecord] = []
    events: list[PaymentEventRecord] = []
    refunds: list[RefundRecord] = []


class StockResponse(_Resp):
    sku: str
    available: int
    reserved: int = 0


class ReservationRecord(_Resp):
    id: str
    order_id: str
    sku: str
    quantity: int
    status: str
    replayed: bool = False


class ReservationsResponse(_Resp):
    order_id: str
    reservations: list[ReservationRecord] = []


class ShipmentRecord(_Resp):
    id: str
    order_id: str
    status: str
    carrier: str = ""
    tracking: str = ""
    replayed: bool = False


class ShipmentsResponse(_Resp):
    order_id: str
    shipments: list[ShipmentRecord] = []


def _clean_memos(model: BaseModel) -> BaseModel:
    """Sanitise free-text fields coming from the (untrusted) service."""
    for name in ("memo",):
        if hasattr(model, name):
            setattr(model, name, sanitize_text(getattr(model, name), 200))
    return model


class SimClient:
    def __init__(self, base_url: str, timeout_s: float = 2.0, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = httpx.Client(base_url=self.base_url, transport=transport)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, *, mutating: bool, model: type[_Resp], correlation_id: str | None,
                 idem_key: str | None = None, params: dict | None = None, json: dict | None = None,
                 timeout: float | None = None) -> Any:
        headers = {"X-Correlation-ID": correlation_id or "-"}
        if idem_key:
            headers["Idempotency-Key"] = idem_key
        try:
            resp = self._client.request(method, path, params=params, json=json, headers=headers,
                                        timeout=timeout or self.timeout_s)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise TransientToolError(f"cannot reach simulated service: {type(e).__name__}", code="CONNECTION_FAILED") from e
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError) as e:
            if mutating:
                raise AmbiguousToolError(f"timeout while waiting for response ({type(e).__name__}); outcome unknown") from e
            raise TransientToolError(f"read timed out ({type(e).__name__})", code="TIMEOUT") from e
        except httpx.HTTPError as e:
            raise TransientToolError(f"transport error: {type(e).__name__}", code="TRANSPORT_ERROR") from e

        code = resp.status_code
        if code in (502, 503):
            raise TransientToolError(f"service returned {code}", code="SERVICE_UNAVAILABLE", details={"status": code})
        if code == 504:
            if mutating:
                raise AmbiguousToolError("gateway timeout (504); outcome unknown", details={"status": 504})
            raise TransientToolError("gateway timeout (504)", code="TIMEOUT", details={"status": 504})
        if code >= 500:
            raise TransientToolError(f"service error {code}", code="SERVICE_ERROR", details={"status": code})
        if code >= 400:
            detail = _safe_detail(resp)
            raise PermanentToolError(f"service rejected request ({code}): {detail.get('error', 'error')}",
                                     code=str(detail.get("error", "REJECTED")).upper(),
                                     details={"status": code, **detail})
        try:
            return _clean_memos_deep(model.model_validate(resp.json()))
        except (ValidationError, ValueError) as e:
            raise PermanentToolError("service returned a malformed response", code="MALFORMED_RESPONSE",
                                     details={"error": str(e)[:300]}) from e

    # ------------------------------------------------------------------ reads
    def get_payments(self, order_id: str, *, correlation_id: str | None = None, timeout: float | None = None) -> PaymentsResponse:
        return self._request("GET", f"/v1/payments/orders/{order_id}", mutating=False, model=PaymentsResponse,
                             correlation_id=correlation_id, timeout=timeout)

    def get_stock(self, sku: str, *, correlation_id: str | None = None, timeout: float | None = None) -> StockResponse:
        return self._request("GET", f"/v1/inventory/stock/{sku}", mutating=False, model=StockResponse,
                             correlation_id=correlation_id, timeout=timeout)

    def get_reservations(self, order_id: str, *, correlation_id: str | None = None, timeout: float | None = None) -> ReservationsResponse:
        return self._request("GET", "/v1/inventory/reservations", mutating=False, model=ReservationsResponse,
                             params={"order_id": order_id}, correlation_id=correlation_id, timeout=timeout)

    def get_shipments(self, order_id: str, *, correlation_id: str | None = None, timeout: float | None = None) -> ShipmentsResponse:
        return self._request("GET", "/v1/shipping/shipments", mutating=False, model=ShipmentsResponse,
                             params={"order_id": order_id}, correlation_id=correlation_id, timeout=timeout)

    # ------------------------------------------------------------------ mutations
    def create_reservation(self, *, order_id: str, sku: str, quantity: int, idem_key: str,
                           correlation_id: str | None = None, timeout: float | None = None) -> ReservationRecord:
        return self._request("POST", "/v1/inventory/reservations", mutating=True, model=ReservationRecord,
                             json={"order_id": order_id, "sku": sku, "quantity": quantity}, idem_key=idem_key,
                             correlation_id=correlation_id, timeout=timeout)

    def create_shipment(self, *, order_id: str, sku: str, quantity: int, idem_key: str,
                        correlation_id: str | None = None, timeout: float | None = None) -> ShipmentRecord:
        return self._request("POST", "/v1/shipping/shipments", mutating=True, model=ShipmentRecord,
                             json={"order_id": order_id, "sku": sku, "quantity": quantity}, idem_key=idem_key,
                             correlation_id=correlation_id, timeout=timeout)

    def create_refund(self, *, payment_id: str, amount_cents: int, reason: str, idem_key: str,
                      correlation_id: str | None = None, timeout: float | None = None) -> RefundRecord:
        return self._request("POST", f"/v1/payments/{payment_id}/refunds", mutating=True, model=RefundRecord,
                             json={"amount_cents": amount_cents, "reason": reason}, idem_key=idem_key,
                             correlation_id=correlation_id, timeout=timeout)

    # ------------------------------------------------------------------ admin (scenario setup / eval only)
    def admin(self, method: str, path: str, json: dict | None = None) -> dict:
        resp = self._client.request(method, path, json=json, timeout=10)
        resp.raise_for_status()
        return resp.json()


def _safe_detail(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
        d = body.get("detail", body) if isinstance(body, dict) else {}
        return d if isinstance(d, dict) else {"error": str(d)[:100]}
    except ValueError:
        return {"error": "non_json_error"}


def _clean_memos_deep(model: _Resp) -> _Resp:
    _clean_memos(model)
    for value in model.__dict__.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, BaseModel):
                    _clean_memos(item)
    return model
