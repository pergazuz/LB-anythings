"""What Label Studio's model card sees about training, and how the Detector survives a run."""

from fastapi.testclient import TestClient

from lb_anythings.domain.checkpoint import Checkpoint
from tests.conftest import BEST, Fakes
from tests.label_configs import PIPE

ONE_TASK = [{"id": 1, "data": {"image": "a.jpg"}}]


def test_is_training_reflects_an_active_training_run(client: TestClient, fakes: Fakes) -> None:
    assert client.get("/is_training").json() == {"is_training": False}

    run = fakes.trainer.start()
    assert client.get("/is_training").json() == {"is_training": True}

    fakes.trainer.finish(run, succeeded=True)
    assert client.get("/is_training").json() == {"is_training": False}


def test_a_checkpoint_that_fails_to_load_keeps_the_serving_detector(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})
    assert client.post("/predict", json={"tasks": ONE_TASK}).json()["model_version"] == "best.pt#1"

    # a Training Run is mid-write: the file is newer but not yet loadable
    fakes.checkpoints.checkpoint = Checkpoint(BEST, modified_at=2.0)
    fakes.detectors.fail_next_load = True
    body = client.post("/predict", json={"tasks": ONE_TASK}).json()

    assert body["model_version"] == "best.pt#1"
    # ...and the next call tries again and gets the new one
    assert client.post("/predict", json={"tasks": ONE_TASK}).json()["model_version"] == "best.pt#2"
