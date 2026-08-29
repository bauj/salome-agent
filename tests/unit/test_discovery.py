from salome_agent.session.discovery import discover_port


def test_discover_port_parses_contents_not_filename(tmp_path):
    cfg = tmp_path / ".omniORB_somehost_99999.cfg"
    cfg.write_text("InitRef = NameService=corbaname::localhost:12345\n")
    assert discover_port(str(tmp_path)) == 12345


def test_discover_port_none_when_no_cfg(tmp_path):
    assert discover_port(str(tmp_path)) is None
