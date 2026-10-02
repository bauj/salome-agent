import base64

import pytest

from salome_agent.cli import (check_image_uris, load_reference_image, load_reference_images,
                              parse_args, parse_image_paths)


def test_parses_batch_and_no_session():
    args = parse_args(["make a box", "--batch", "--no-session"])
    assert args.request == "make a box"
    assert args.batch is True
    assert args.no_session is True


def test_defaults_are_interactive_with_session():
    args = parse_args(["make a box"])
    assert args.batch is False
    assert args.no_session is False


def test_verbose_counts():
    assert parse_args(["x"]).verbose == 0
    assert parse_args(["x", "-v"]).verbose == 1
    assert parse_args(["x", "-vv"]).verbose == 2


def test_tools_flag():
    assert parse_args(["x"]).tools is None
    assert parse_args(["x", "--tools", "all"]).tools == "all"
    assert parse_args(["x", "--tools", "hints"]).tools == "hints"


def test_no_live_gui_or_port():
    args = parse_args(["make a box"])
    assert not hasattr(args, "live_gui")
    assert not hasattr(args, "port")


def test_images_flag_defaults_to_none():
    assert parse_args(["make a box"]).images is None


def test_images_flag_parses_comma_separated_paths():
    args = parse_args(["make a box", "--images", "front.png,side.jpg,top.bmp"])
    assert args.images == "front.png,side.jpg,top.bmp"


def test_parse_image_paths_splits_and_trims():
    assert parse_image_paths("front.png, side.jpg ,top.bmp") == \
        ["front.png", "side.jpg", "top.bmp"]


def test_parse_image_paths_drops_empty_entries():
    assert parse_image_paths("front.png,,side.jpg,") == ["front.png", "side.jpg"]


def test_parse_image_paths_single_path():
    assert parse_image_paths("ref.png") == ["ref.png"]


PNG = b"\x89PNG\r\n\x1a\n" + b"rest"
BMP = b"BM" + b"rest"


def test_load_reference_image_returns_data_uri(tmp_path):
    path = tmp_path / "ref.png"
    path.write_bytes(PNG)
    uri = load_reference_image(str(path))
    assert uri == "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


@pytest.mark.parametrize("data,mime", [
    (BMP, "image/bmp"), (b"\xff\xd8\xff\xe0rest", "image/jpeg"),
    (b"GIF89arest", "image/gif"), (b"RIFF\0\0\0\0WEBPrest", "image/webp")])
def test_load_reference_image_detects_the_type_from_content(tmp_path, data, mime):
    path = tmp_path / "ref.bin"
    path.write_bytes(data)
    assert load_reference_image(str(path)).startswith(f"data:{mime};base64,")


def test_load_reference_image_ignores_a_misleading_extension(tmp_path):
    path = tmp_path / "ref.png"
    path.write_text("hello")
    with pytest.raises(ValueError, match="is not a PNG, JPEG, WEBP, GIF or BMP image"):
        load_reference_image(str(path))


def test_load_reference_image_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        load_reference_image(str(tmp_path / "missing.png"))


def test_load_reference_images_returns_one_uri_per_path(tmp_path):
    p1 = tmp_path / "front.png"
    p2 = tmp_path / "side.bmp"
    p1.write_bytes(PNG)
    p2.write_bytes(BMP)
    uris = load_reference_images([str(p1), str(p2)])
    assert len(uris) == 2
    assert uris[0].startswith("data:image/png;base64,")
    assert uris[1].startswith("data:image/bmp;base64,")


def test_load_reference_images_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no image path given"):
        load_reference_images(parse_image_paths(" , "))


def test_load_reference_images_caps_the_count(tmp_path):
    from salome_agent.cli import MAX_IMAGES

    path = tmp_path / "ref.png"
    path.write_bytes(PNG)
    with pytest.raises(ValueError, match=f"at most {MAX_IMAGES}"):
        load_reference_images([str(path)] * (MAX_IMAGES + 1))


def test_load_reference_images_caps_the_total_size(tmp_path, monkeypatch):
    from salome_agent import cli

    monkeypatch.setattr(cli, "MAX_IMAGE_BYTES", 10)
    path = tmp_path / "ref.png"
    path.write_bytes(PNG)
    with pytest.raises(ValueError, match="MB of images"):
        load_reference_images([str(path), str(path)])


def uri(data, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def test_check_image_uris_passes_good_images():
    assert check_image_uris([uri(PNG), uri(BMP, "image/bmp")]) == [uri(PNG), uri(BMP, "image/bmp")]


def test_check_image_uris_relabels_by_content():
    # a browser types a file by its extension: a JPEG saved as .png says image/png
    jpeg = b"\xff\xd8\xff\xe0rest"
    assert check_image_uris([uri(jpeg, "image/png")]) == [uri(jpeg, "image/jpeg")]


@pytest.mark.parametrize("bad", [
    "x", uri(b"<svg/>", "image/svg+xml"), uri(b"hello", "text/plain"),
    'data:image/png;base64,AA"onerror=x', "data:image/png," + "abc", 1])
def test_check_image_uris_rejects_what_is_not_an_image(bad):
    with pytest.raises(ValueError):
        check_image_uris([bad])


def test_check_image_uris_rejects_a_non_list():
    with pytest.raises(ValueError, match="list"):
        check_image_uris(uri(PNG))


def test_check_image_uris_caps_the_count():
    from salome_agent.cli import MAX_IMAGES

    with pytest.raises(ValueError, match=f"at most {MAX_IMAGES}"):
        check_image_uris([uri(PNG)] * (MAX_IMAGES + 1))


def test_check_image_uris_caps_the_total_size(monkeypatch):
    from salome_agent import cli

    monkeypatch.setattr(cli, "MAX_IMAGE_BYTES", 15)
    check_image_uris([uri(PNG)])
    with pytest.raises(ValueError, match="MB of images"):
        check_image_uris([uri(PNG), uri(PNG)])


def test_main_rejects_an_empty_images_flag(tmp_path, monkeypatch):
    from salome_agent import cli, config

    monkeypatch.setattr(config, "load_config", lambda path: {"work_dir": str(tmp_path)})
    assert cli.main(["make a box", "--images", ","]) == 1


def test_main_no_session_delegates_to_run_request(tmp_path, monkeypatch, capsys):
    import json

    from salome_agent import cli
    from salome_agent.runner import RunResult

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"llm": {"base_url": "http://x", "model": "m"},
                               "salome": {"root": "/x"},
                               "policy": {"on_stage_failure": "abort"},
                               "work_dir": str(tmp_path)}))
    seen = {}

    def fake_run(request, config, **kw):
        seen.update(kw, request=request, config=config)
        return RunResult(run_id=kw["run_id"], run_dir=str(tmp_path / kw["run_id"]),
                         final_state={"report": "REPORT-TEXT"}, exit_code=0)

    monkeypatch.setattr("salome_agent.runner.run_request", fake_run)
    rc = cli.main(["make a box", "--no-session", "--batch", "--config", str(cfg),
                   "--on-failure", "continue"])
    assert rc == 0
    assert seen["request"] == "make a box"
    assert seen["client"] is None and seen["batch"] is True
    assert seen["config"]["policy"]["on_stage_failure"] == "continue"
    assert "REPORT-TEXT" in capsys.readouterr().out


def test_main_returns_1_on_abort(tmp_path, monkeypatch):
    import json

    from salome_agent import cli
    from salome_agent.runner import RunResult

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"llm": {"base_url": "http://x", "model": "m"},
                               "salome": {"root": "/x"}, "policy": {}}))
    monkeypatch.setattr(
        "salome_agent.runner.run_request",
        lambda request, config, **kw: RunResult(kw["run_id"], "/x", {"aborted": True}, 1))
    assert cli.main(["x", "--no-session", "--config", str(cfg)]) == 1


def test_chat_with_batch_is_an_error(capsys):
    from salome_agent.cli import main
    assert main(["a box", "--chat", "--batch"]) == 2


def test_chat_with_no_session_is_an_error(capsys):
    from salome_agent.cli import main
    assert main(["a box", "--chat", "--no-session"]) == 2
