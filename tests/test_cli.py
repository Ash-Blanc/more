from more.cli import main


def test_demo_and_route_commands(capsys) -> None:
    assert main(["demo", "--name", "Ada"]) == 0
    assert "Ada" in capsys.readouterr().out
    assert main(["route", "--scores", "0.2", "0.8"]) == 0
    output = capsys.readouterr().out
    assert "expert=" in output


def test_parser_has_training_commands() -> None:
    from more.cli import build_parser

    help_text = build_parser().format_help()
    assert "train" in help_text
    assert "generate" in help_text
