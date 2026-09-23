from src import data


def test_label_set_is_the_60_massive_intents():
    assert len(data.intents()) == 60


def test_multilingual_split_has_the_english_budget_spread_evenly():
    en = data.train_set("en")
    multi = data.train_set("multi", seed=0)
    per_locale = len(en) // len(data.LOCALES)
    assert len(multi) == per_locale * len(data.LOCALES)     # same budget, up to rounding
    assert data.train_set("multi", seed=0) == multi          # seeded, so runs are repeatable
    assert data.train_set("multi", seed=1) != multi
