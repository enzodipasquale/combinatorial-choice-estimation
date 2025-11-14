import logging
import pytest

from bundlechoice.comm_manager import CommManager


class FakeComm:
    def __init__(self, *, size: int = 4, rank: int = 1):
        self._size = size
        self._rank = rank
        self.aborted = False
        self.errhandler = None

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def Set_errhandler(self, handler):
        self.errhandler = handler

    def Abort(self, errorcode=1):
        self.aborted = True
        self.errorcode = errorcode

    # MPI API used by CommManager methods
    def bcast(self, *args, **kwargs):
        raise RuntimeError("simulated broadcast failure")


def test_comm_manager_abort_on_mpi_failure(caplog):
    fake_comm = FakeComm()
    manager = CommManager(fake_comm)
    caplog.set_level(logging.ERROR)

    with pytest.raises(RuntimeError, match="simulated broadcast failure"):
        manager.broadcast_from_root({"payload": 1})

    assert fake_comm.aborted, "comm.Abort should be invoked when a rank fails"
    assert any("encountered an error during 'broadcast_from_root'" in message for message in caplog.messages)


def test_fail_fast_context_manager_aborts(caplog):
    fake_comm = FakeComm()
    manager = CommManager(fake_comm)
    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError, match="context failure"):
        with manager.fail_fast("context operation", errorcode=5):
            raise ValueError("context failure")

    assert fake_comm.aborted
    assert fake_comm.errorcode == 5
    assert any("context operation" in message for message in caplog.messages)


def test_abort_all_helper_logs_and_aborts(caplog):
    fake_comm = FakeComm()
    manager = CommManager(fake_comm)
    caplog.set_level(logging.ERROR)

    manager.abort_all("manual abort", errorcode=7)

    assert fake_comm.aborted
    assert fake_comm.errorcode == 7
    assert any("manual abort" in message for message in caplog.messages)

