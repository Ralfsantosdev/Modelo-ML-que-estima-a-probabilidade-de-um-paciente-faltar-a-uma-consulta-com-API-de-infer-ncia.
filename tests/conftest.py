import pytest
from noshow.data import gerar_dataset


@pytest.fixture(scope="session")
def df_small():
    return gerar_dataset(n_pacientes=800, n_consultas=6000, seed=1)
