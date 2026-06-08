import math

import pytest
import torch

from bytelatent.opencompass import ByteLatentOpenCompassModel


class _FakeGenerator:
    def __init__(self, loglikelihoods=None, generations=None):
        self.max_gen_len = 7
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.until = []
        self.max_until_size = 1
        self._loglikelihoods = loglikelihoods or []
        self._generations = generations or []
        self.seen_prompts = None

    def generate(self, prompts):
        self.seen_prompts = prompts
        generations = self._generations or ["" for _ in prompts]
        return generations, self._loglikelihoods, []


class _AdapterForTest(ByteLatentOpenCompassModel):
    def __init__(self, generator):
        super().__init__(ckpt_dir="/tmp/unused", consolidate_if_needed=False)
        self._generator = generator
        self._model = object()
        self._tokenwise_nll = [
            -item.float() for item in getattr(generator, "_loglikelihoods", [])
        ]

    def _load_model(self):
        return None

    def _score_tokenwise_nll(self, inputs):
        return self._tokenwise_nll, [0 for _ in inputs]


class _CharTokenizer:
    def encode(self, text, add_bos=True, add_eos=False):
        tokens = list(range(len(text)))
        if add_bos:
            tokens = [-1] + tokens
        if add_eos:
            tokens = tokens + [-2]
        return tokens

    def decode(self, tokens):
        return "".join(str(item) for item in tokens)


def test_get_ppl_returns_mean_nll_not_exp_perplexity():
    adapter = _AdapterForTest(
        _FakeGenerator(loglikelihoods=[torch.tensor([-1.0, -2.0, -3.0])])
    )

    assert adapter.get_ppl(["abc"]) == pytest.approx([2.0])
    assert adapter.get_ppl(["abc"])[0] != pytest.approx(math.exp(2.0))


def test_get_ppl_applies_opencompass_mask_length_convention():
    adapter = _AdapterForTest(
        _FakeGenerator(loglikelihoods=[torch.tensor([-1.0, -2.0, -3.0, -4.0])])
    )

    # OpenCompass masks a prefix of M tokens by starting CE at index M - 1.
    assert adapter.get_ppl(["abc"], mask_length=[3]) == pytest.approx([3.5])


def test_get_ppl_tokenwise_returns_nll_after_mask():
    adapter = _AdapterForTest(
        _FakeGenerator(loglikelihoods=[torch.tensor([-1.0, -2.0, -3.0])])
    )

    result = adapter.get_ppl_tokenwise(["abc"], mask_length=[2])
    assert len(result) == 1
    assert result[0] == pytest.approx([2.0, 3.0])


def test_get_ppl_tokenwise_accepts_positional_mask_for_older_callers():
    adapter = _AdapterForTest(
        _FakeGenerator(loglikelihoods=[torch.tensor([-1.0, -2.0, -3.0])])
    )

    result = adapter.get_ppl_tokenwise(["abc"], [2])
    assert result[0] == pytest.approx([2.0, 3.0])


def test_get_ppl_tokenwise_accepts_opencompass_label_signature():
    adapter = _AdapterForTest(
        _FakeGenerator(loglikelihoods=[torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])])
    )
    adapter._tokenizer = _CharTokenizer()

    loss_sums, token_counts = adapter.get_ppl_tokenwise(
        ["abcde"], [[[1, 4, 1]]], mask_length=None
    )
    assert loss_sums == pytest.approx([9.0])
    assert token_counts == [3]


def test_generate_forwards_generation_controls():
    generator = _FakeGenerator(generations=["A"])
    adapter = _AdapterForTest(generator)

    assert adapter.generate(
        ["question"],
        max_out_len=3,
        min_out_len=1,
        temperature=0.2,
        top_p=0.9,
        top_k=4,
        stop=["\n"],
        stopping_criteria=["END"],
    ) == ["A"]
    assert generator.seen_prompts == ["question"]
    assert generator.max_gen_len == 3
    assert generator.min_gen_len == 1
    assert generator.temperature == 0.2
    assert generator.top_p == 0.9
    assert generator.top_k == 4
    assert generator.until == ["\n", "END"]


def test_adapter_registers_with_opencompass_registry_when_available(tmp_path):
    import subprocess
    import sys
    import textwrap

    fake_pkg = tmp_path / "fake_opencompass_registry"
    base_dir = fake_pkg / "opencompass" / "models"
    base_dir.mkdir(parents=True)
    (fake_pkg / "opencompass" / "__init__.py").write_text("")
    (base_dir / "__init__.py").write_text("")
    (base_dir / "base.py").write_text(
        textwrap.dedent(
            """
            class BaseModel:
                def __init__(self, *args, **kwargs):
                    pass
            """
        )
    )
    (fake_pkg / "opencompass" / "registry.py").write_text(
        textwrap.dedent(
            """
            class Registry:
                def __init__(self):
                    self.registered = {}
                def register_module(self):
                    def decorate(cls):
                        self.registered[cls.__name__] = cls
                        return cls
                    return decorate
            MODELS = Registry()
            """
        )
    )
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(fake_pkg)!r})
        from bytelatent.opencompass import ByteLatentOpenCompassModel
        from opencompass.registry import MODELS
        assert MODELS.registered['ByteLatentOpenCompassModel'] is ByteLatentOpenCompassModel
        print('registered')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "registered"


def test_adapter_initializes_opencompass_base_model_when_available(tmp_path):
    import subprocess
    import sys
    import textwrap

    fake_pkg = tmp_path / "fake_opencompass"
    base_dir = fake_pkg / "opencompass" / "models"
    base_dir.mkdir(parents=True)
    (fake_pkg / "opencompass" / "__init__.py").write_text("")
    (base_dir / "__init__.py").write_text("")
    (base_dir / "base.py").write_text(
        textwrap.dedent(
            """
            class BaseModel:
                def __init__(self, path, max_seq_len=2048, tokenizer_only=False,
                             meta_template=None, generation_kwargs=None,
                             sync_rank=False):
                    self.base_init_args = dict(
                        path=path,
                        max_seq_len=max_seq_len,
                        tokenizer_only=tokenizer_only,
                        meta_template=meta_template,
                        generation_kwargs=generation_kwargs or {},
                        sync_rank=sync_rank,
                    )
            """
        )
    )
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(fake_pkg)!r})
        from bytelatent.opencompass import ByteLatentOpenCompassModel
        model = ByteLatentOpenCompassModel(
            ckpt_dir='/tmp/ckpt',
            max_seq_len=123,
            tokenizer_only=False,
            generation_kwargs={{'do_sample': False}},
            sync_rank=True,
        )
        assert model.base_init_args['path'] == '/tmp/ckpt'
        assert model.base_init_args['max_seq_len'] == 123
        assert model.base_init_args['tokenizer_only'] is False
        assert model.base_init_args['generation_kwargs'] == {{'do_sample': False}}
        assert model.base_init_args['sync_rank'] is True
        print('ok')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "ok"
