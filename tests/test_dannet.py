"""Test dasem.dannet."""

import pytest

from dasem.dannet import Dannet


@pytest.fixture
def dannet():
    return Dannet()


def test_download(dannet):
    dannet.download()


def test_glossary(dannet):
    assert len(dannet.glossary('virksomhed')) == 3
