"""Automatic retention purge running inside the API process (CT-26, DA-14).

The loop is an ``asyncio`` task started by the API lifespan: it purges once at
startup and then every ``purge_interval_seconds``. The blocking SQLite work runs
in a worker thread so the event loop keeps serving requests.
"""

import asyncio
import logging

from ticket_classifier.settings import Settings
from ticket_classifier.storage.database import connect, utc_now
from ticket_classifier.storage.purge import PURGE_MESSAGE, purge_expired_predictions

logger = logging.getLogger(__name__)

PURGE_FAILED_MESSAGE = "Automatic purge failed ({error}); will retry at the next interval."


def _purge_blocking(settings: Settings) -> int:
    # The connection is opened, used and closed in the same worker thread.
    conn = connect(settings.db_path)
    try:
        return purge_expired_predictions(conn, utc_now(), settings.retention_days)
    finally:
        conn.close()


async def run_purge_once(settings: Settings) -> int:
    """Run one purge; return the deleted count, or -1 on failure (never raises)."""
    try:
        count = await asyncio.to_thread(_purge_blocking, settings)
    except Exception as exc:  # noqa: BLE001 - a failed purge must not stop the loop
        # Only the exception type is logged: messages may carry file system paths.
        logger.error(PURGE_FAILED_MESSAGE.format(error=type(exc).__name__))
        return -1
    logger.info(PURGE_MESSAGE.format(count=count, days=settings.retention_days))
    return count


async def _purge_loop(settings: Settings) -> None:
    while True:
        await run_purge_once(settings)
        await asyncio.sleep(settings.purge_interval_seconds)


def start_purge_loop(settings: Settings) -> asyncio.Task[None]:
    """Start the purge loop on the running event loop and return its task."""
    return asyncio.create_task(_purge_loop(settings), name="ticket-purge-loop")
