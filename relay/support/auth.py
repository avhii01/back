"""Simulated customer sign-in with real server-side session records.

This is explicitly a demo identity provider: seeded customers and demo passwords are
shown in the support UI. It is still checked server-side: the browser never asserts
who it is, and every customer route resolves the token to a Customer row.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import timedelta

from sqlalchemy import select

from relay.context import AppContext
from relay.models import Customer, CustomerSession, utcnow

_ITERATIONS = 120_000


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii", "ignore")).hexdigest()


def create_session(app: AppContext, customer_id: str) -> tuple[str, CustomerSession]:
    token = secrets.token_urlsafe(32)
    session = CustomerSession(token_hash=_token_hash(token), customer_id=customer_id,
                              expires_at=utcnow() + timedelta(hours=app.settings.customer_session_hours))
    with app.session() as s:
        s.add(session)
        s.commit()
        s.refresh(session)
    return token, session


def customer_from_token(app: AppContext, token: str | None) -> Customer | None:
    if not token:
        return None
    with app.session() as s:
        session = s.get(CustomerSession, _token_hash(token))
        if session is None:
            return None
        if session.expires_at < utcnow():
            s.delete(session)
            s.commit()
            return None
        return s.get(Customer, session.customer_id)


def revoke_session(app: AppContext, token: str | None) -> None:
    if not token:
        return
    with app.session() as s:
        row = s.get(CustomerSession, _token_hash(token))
        if row is not None:
            s.delete(row)
            s.commit()


def authenticate(app: AppContext, email: str, password: str) -> Customer | None:
    with app.session() as s:
        customer = s.scalars(select(Customer).where(Customer.email == email.strip().lower())).first()
        if customer is None or not verify_password(password, customer.password_hash):
            return None
        return customer


def upsert_customer(app: AppContext, *, customer_id: str, email: str, name: str, password: str) -> Customer:
    with app.session() as s:
        row = s.scalars(select(Customer).where(Customer.email == email.lower())).first()
        if row is None:
            row = Customer(id=customer_id, email=email.lower(), name=name, password_hash=hash_password(password))
            s.add(row)
        else:
            row.name = name
        s.commit()
        s.refresh(row)
        return row


import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_user_to_excel(
    user_id: str,
    name: str,
    email: str,
    password: str,
    file_path: str = "users.xlsx",
) -> None:
    """Persist newly registered user details to an Excel workbook and companion CSV."""
    now_str = utcnow().strftime("%Y-%m-%d %H:%M:%S")
    target_path = Path(file_path)

    # 1. Update / write Excel (.xlsx) file
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill

        if target_path.exists():
            wb = openpyxl.load_workbook(target_path)
            ws = wb.active
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Registered Users"
            headers = ["Registered At (UTC)", "User ID", "Name", "Email", "Password"]
            ws.append(headers)
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")

        ws.append([now_str, user_id, name, email, password])
        wb.save(target_path)
        logger.info("Saved user %s to excel %s", email, target_path)
    except Exception as e:
        logger.warning("Could not write user to %s: %s", target_path, e)

    # 2. Update / write companion CSV (.csv) file
    try:
        csv_path = target_path.with_suffix(".csv")
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or csv_path.stat().st_size == 0:
                writer.writerow(["Registered At (UTC)", "User ID", "Name", "Email", "Password"])
            writer.writerow([now_str, user_id, name, email, password])
        logger.info("Saved user %s to csv %s", email, csv_path)
    except Exception as e:
        logger.warning("Could not write user to csv: %s", e)


def register_customer(
    app: AppContext,
    *,
    name: str,
    email: str,
    user_id: str | None,
    password: str,
) -> tuple[str, CustomerSession, Customer]:
    """Register a new customer, store details in Excel, and return session credentials."""
    clean_name = name.strip()
    clean_email = email.strip().lower()
    clean_uid = (user_id or "").strip()
    if not clean_uid:
        clean_uid = f"usr_{secrets.token_hex(4)}"

    hashed = hash_password(password)
    with app.session() as s:
        existing = s.scalars(select(Customer).where(Customer.email == clean_email)).first()
        if existing is not None:
            existing.name = clean_name
            existing.password_hash = hashed
            customer = existing
        else:
            if s.get(Customer, clean_uid) is not None:
                clean_uid = f"usr_{secrets.token_hex(4)}"
            customer = Customer(id=clean_uid, email=clean_email, name=clean_name, password_hash=hashed)
            s.add(customer)
        s.commit()
        s.refresh(customer)

    save_user_to_excel(customer.id, customer.name, customer.email, password)
    token, session = create_session(app, customer.id)
    return token, session, customer
