from typer.testing import CliRunner

from fifth_dimension_search import cli

runner = CliRunner()


def test_cli_info_runs():
    result = runner.invoke(cli.app, ["info"])
    assert result.exit_code == 0
    assert "Fifth Dimension Sandbox" in result.stdout


def test_cli_datasets_list():
    result = runner.invoke(cli.app, ["datasets", "list"])
    assert result.exit_code == 0
    assert "phase_amp_summary.csv" in result.stdout


def test_cli_plot(tmp_path):
    output = tmp_path / "plot.png"
    result = runner.invoke(cli.app, ["plot", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
