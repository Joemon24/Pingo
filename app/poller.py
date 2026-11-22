import logging
import time

from google.auth.exceptions import RefreshError

from app.config import settings
from app.gmail import Gmail
from app.pipeline import handle_message, require_dry_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("poller")


def poll_once(max_results: int = 10) -> dict[str, int]:
    require_dry_run()
    gmail = Gmail()
    message_ids = gmail.list_new_message_ids(max_results=max_results)
    failed = 0
    for message_id in message_ids:
        try:
            handle_message(gmail, message_id)
        except RefreshError:
            raise
        except Exception as exc:
            failed += 1
            # One bad email must not stop the loop.
            logger.error("Message handling failed (%s)", type(exc).__name__)
    return {"found": len(message_ids), "failed": failed}


def main() -> None:
    require_dry_run()
    logger.info("Polling every %s seconds", settings.poll_interval_seconds)
    while True:
        try:
            result = poll_once()
            if result["found"]:
                logger.info("Inbox check found %s messages with %s errors", result["found"], result["failed"])
        except Exception as exc:
            logger.error("Poll failed (%s)", type(exc).__name__)
        time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":
    main()