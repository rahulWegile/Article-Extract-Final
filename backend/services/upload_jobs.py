import threading
import uuid

_jobs = {}
_lock = threading.Lock()


def create_job():

    job_id = uuid.uuid4().hex

    with _lock:

        _jobs[job_id] = {
            "status": "processing",
            "stage": "Queued",
            "progress": 0,
            "document_id": None,
            "error": None,
        }

    return job_id


def update_job(job_id, **fields):

    with _lock:

        if job_id in _jobs:

            _jobs[job_id].update(fields)


def get_job(job_id):

    with _lock:

        job = _jobs.get(job_id)

        return dict(job) if job is not None else None
