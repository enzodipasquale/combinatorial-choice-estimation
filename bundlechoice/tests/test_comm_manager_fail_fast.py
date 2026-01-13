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

