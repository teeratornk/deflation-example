"""Submission bookkeeping that preserves an unsubmitted restart request."""


def submit_pending(entry, submit):
    """Consume the pending checkpoint only after the scheduler returns a job ID.

    The caller persists the entry after success. A scheduler exception leaves
    every field unchanged, so a later poll submits the same restart request.
    """
    resume = entry.get("resume_from")
    result = submit(resume)
    if not result or not result.get("job"):
        raise ValueError("Submission returned no job identity")
    entry["live"] = result
    entry.pop("resume_from", None)
    return result
