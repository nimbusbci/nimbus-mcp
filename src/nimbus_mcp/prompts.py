"""Example prompts (MCP ``prompts`` capability).

Registries such as Smithery list these on the server page, and MCP clients
show them as slash-commands / starters. Each one is a plain user message that
names real tools and template/dataset ids, so the assistant can follow it
without guessing. Keep them in sync with the tool names (a unit test checks).
"""

from __future__ import annotations

from fastmcp import FastMCP

# name -> (title, description, prompt text, tools the text relies on)
EXAMPLE_PROMPTS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "mi_csp_lda_bci_iv_2a": (
        "Train CSP+LDA on BCI IV 2a",
        "Run the built-in motor-imagery CSP -> LDA template and show the confusion matrix.",
        "Train a CSP+LDA motor-imagery pipeline on BCI Competition IV 2a and show the "
        "confusion matrix. Use catalog.template('mi_bciiv2a_csp_lda'), start it with "
        "execution.run, poll execution.get until it finishes, then summarize "
        "execution.results and render plots.confusion.",
        (
            "catalog.template",
            "execution.run",
            "execution.get",
            "execution.results",
            "plots.confusion",
        ),
    ),
    "compare_mi_pipelines": (
        "Compare MI pipelines",
        "One paced experiment: CSP+LDA vs FBCSP+LDA vs EEGNet on BCI IV 2a.",
        "Compare CSP+LDA, FBCSP+LDA and EEGNet on BCI Competition IV 2a. Build the graphs "
        "from catalog.template ('mi_bciiv2a_csp_lda', 'mi_bciiv2a_fbcsp_lda', "
        "'mi_eegnet_moabb'), run them together with experiment.run, poll experiment.get, "
        "and rank the pipelines by accuracy.",
        ("catalog.template", "experiment.run", "experiment.get"),
    ),
    "foundation_backbones": (
        "Pick a foundation backbone",
        "Benchmark every pretrained EEG backbone on a dataset and pick the best one.",
        "Which pretrained EEG foundation backbone works best on BCI Competition IV 2a "
        "(BNCI2014_001), subject 1? Run foundation.compare with source='dataset' and show "
        "the ranked chart with accuracy, confidence interval and wall time.",
        ("foundation.compare",),
    ),
    "inspect_my_eeg": (
        "Inspect my EEG file",
        "Upload a recording and get an exploratory summary before building a pipeline.",
        "Upload my EEG recording with data.upload (pass sampling_rate if it's a "
        "CSV/TSV/TXT file), then run data.inspect_file on it and summarize the channels, "
        "sampling rate, class balance and band powers. Suggest a matching starter "
        "template from catalog.templates.",
        ("data.upload", "data.inspect_file", "catalog.templates"),
    ),
    "leaderboard_standing": (
        "Check the leaderboard",
        "See the public leaderboard and where your submissions stand.",
        "Show the public Nimbus leaderboard with catalog.leaderboard and list my own "
        "submissions and their scores with leaderboard.mine.",
        ("catalog.leaderboard", "leaderboard.mine"),
    ),
}


def register(mcp: FastMCP) -> None:
    for name, (title, description, text, _tools) in EXAMPLE_PROMPTS.items():

        def _make(t: str = text):
            def _prompt() -> str:
                return t

            return _prompt

        mcp.prompt(name=name, title=title, description=description)(_make())
