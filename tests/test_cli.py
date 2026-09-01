import pytest

from prompt_workbench import __version__
from prompt_workbench.cli import main


def test_cli_version_flag_prints_version(capsys):
    """`--version` must print the package version (from the single source of
    truth) and exit cleanly, as argparse's version action does."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_without_args_still_greets(capsys):
    """The default behaviour (no flags) must remain the greeting."""
    main([])

    assert "Hello from prompt_workbench" in capsys.readouterr().out
