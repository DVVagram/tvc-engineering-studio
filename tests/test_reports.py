import pytest

from tvc_studio.analysis import reports


def test_mass_report_contains_breakdown(sample_project):
    report = reports.mass_report(sample_project)
    text = report.to_text()
    assert "Total mass" in text
    assert "Breakdown" in text
    assert report.tables["Breakdown"]
    assert any("Fastener" in row["item"] or row["kind"] == "fastener"
               for row in report.tables["Breakdown"])


def test_fastener_report_lists_installed(sample_project):
    report = reports.fastener_report(sample_project)
    rows = report.tables["Installed fasteners"]
    assert len(rows) == len(sample_project.assembly.fasteners)
    assert all("mode" in row for row in rows)


def test_joint_report_reports_dof(sample_project):
    report = reports.joint_report(sample_project)
    assert "constraint" in report.to_text().lower()
    assert report.tables["Constraints"]


def test_clearance_report_runs(sample_project):
    report = reports.clearance_report_for_project(sample_project)
    assert "Pairs" in report.tables
    assert report.tables["Pairs"]


def test_report_json_round_trip(tmp_path, sample_project):
    report = reports.project_report(sample_project)
    path = str(tmp_path / "report.json")
    report.save(path)
    import json
    payload = json.load(open(path, encoding="utf-8"))
    assert payload["title"] == report.title
