from astromesh_codemode.search import MAX_LIMIT, Index
from tests.fixtures import synthetic_catalog


def test_ranks_by_name_and_description():
    idx = Index(synthetic_catalog(64))
    assert idx.search("invoice total")[0].name == "billing_get_invoice"


def test_param_names_count():
    idx = Index(synthetic_catalog(64))
    names = [s.name for s in idx.search("customer email")]
    assert "crm_find_contact" in names


def test_limits():
    specs = synthetic_catalog(64)
    idx = Index(specs)
    assert len(idx.search("x", limit=10000)) <= MAX_LIMIT
    assert len(Index(synthetic_catalog(512)).search("get", limit=10000)) == MAX_LIMIT
    assert idx.search("", 5) == specs[:5]


def test_deterministic_and_sizes():
    for n in (8, 64, 512):
        a, b = synthetic_catalog(n), synthetic_catalog(n)
        assert len(a) == n
        assert [t.name for t in a] == [t.name for t in b]
        assert len({t.name for t in a}) == n
    assert [t.name for t in synthetic_catalog(2)] == ["billing_get_invoice", "list_orders"]
