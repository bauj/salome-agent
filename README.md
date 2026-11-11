# salome-agent

The core of [SALOME Agent](https://github.com/bauj/salome-agent-gui): it turns a request in plain
words, or a picture, into SALOME geometry and meshes, plus a standalone `result.py` that rebuilds
them.

There are two ways to use it:

- **In the SALOME window:** install [salome-agent-gui](https://github.com/bauj/salome-agent-gui),
  which includes this engine and adds a chat window. Most users want this.
- **From the command line:** use this repo directly, against a headless SALOME with no window, or
  against one already open with the plugin. This suits scripts, batch runs and automation.

The rest of this page covers the command line.

| More detail | |
|---|---|
| [Configuration reference](docs/configuration.md) | Every setting |
| [Development guide](docs/development.md) | Architecture, tests, code layout, adding tools |

## What it does

You write a request such as "make a 10 mm cube, then tetrahedral-mesh it". The agent then:

1. **Reads your images,** if you gave any, and turns them into a written description.
2. **Plans** the job as a list of stages, each one a piece of work for one SALOME module.
3. **Builds each stage** in a live SALOME session, using that module's tools, and checks the result.
4. **Writes `result.py`,** a script that rebuilds everything in SALOME with no agent involved, and
   `report.md`, a record of the run.

It works with three SALOME modules: **SHAPER** for CAD (the default), **GEOM** for CAD when your
request names it ("using GEOM"), and **SMESH** for meshing.

## Quick start

You need:

- **Python 3.10 or newer,** both on your system and in your SALOME
- **a SALOME installation**
- **an LLM:** any OpenAI-compatible endpoint that supports tool calling

**1. Install**

```bash
git clone https://github.com/bauj/salome-agent.git
cd salome-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**2. Configure.** Copy the example, then set `base_url`, `api_key` and `model` in its `llm` section:

```bash
cp config.example.json config.json
```

**3. Start SALOME for the agent.** In a second terminal, from the repo, start a headless SALOME
and leave it running. It is ready after about 10-20 seconds; stop it with Ctrl-C.

```bash
export SALOME_ROOT=/path/to/your/SALOME    # the directory holding the salome launcher
scripts/start-headless-server.sh
```

Prefer to watch it build? Open SALOME with the
[salome-agent-gui](https://github.com/bauj/salome-agent-gui) plugin instead and click
**Tools > Plugins > AI tools > SALOME agent**: the command line then drives that window.

**4. Run a request,** from the repo root, where `config.json` is:

```bash
salome-agent "make a 10mm cube" --batch
```

**5. Get the result.** The run prints its `output dir`, which holds `result.py` and `report.md`.
With the example config, that is under `~/.local/share/salome-agent/runs/`. To rebuild the model
with no agent involved:

```bash
"$SALOME_ROOT"/salome shell -- python3 <output dir>/result.py
```

## Using it

### Review each run, or run unattended

Without `--batch`, the run stops to ask you twice:

- **After planning,** it prints the stages: `[Y]es`, `[e]dit` (opens the plan in `$EDITOR`) or
  `[n]o` to abort.
- **If a stage fails:** `[h]int` (give the agent a tip and retry), `[s]kip stage` or `[a]bort run`.

With `--batch`, the plan is approved automatically. A failed stage then either stops the run or is
skipped: the example config stops it, and `--on-failure continue` skips it instead.

### Start from pictures

```bash
salome-agent "make this bracket" --images ./bracket-sketch.png --batch

# several views of the same part
salome-agent "make this bracket" --images ./front.png,./side.png,./top.bmp --batch
```

- Accepts PNG, JPEG, WEBP, GIF and BMP, up to 10 images and 20 MB in all. You need a vision-capable
  model; [`llm.modules.vision`](docs/configuration.md) can use a different model just for this.
- Several images are read as views of **one** part, not as separate parts.
- Numbers in your text win over numbers read from the image. Values the agent had to estimate are
  marked `~`.

### Keep talking to refine it

```bash
salome-agent "a small house" --chat
```

The session stays open after the first result:

```
chat> make the roof taller
chat> add a chimney
chat> /undo
```

An edit rebuilds only the object it changes and what depends on it, keeping the same names.
Ctrl-C cancels the turn in progress. `--chat` cannot be combined with `--batch`.

| Command | What it does |
|---|---|
| `/help` | List these commands |
| `/undo` | Revert the last turn |
| `/redo` | Re-apply the last undone turn (cleared by any new turn) |
| `/show` | Print the build graph |
| `/result` | Print `result.py` |
| `/rebuild` | Rebuild the whole study from scratch, when an undo failed to restore it |
| `/image PATH[,PATH] [request]` | A request with reference images; with no request, it builds what they show |
| `/quit` | Leave (Ctrl-D too) |

### Options

| Flag | What it does |
|---|---|
| `--batch` | Run without asking anything (see above) |
| `--chat` | Keep talking after the first result (see above) |
| `--images PATH,PATH,...` | Reference images (see above) |
| `--config PATH` | Use another config file (default: `./config.json`) |
| `--work-dir DIR` | Where to write the run's folder |
| `--on-failure {abort,continue}` | What `--batch` does when a stage fails |
| `--tools {hints,all}` | `hints` (default) offers the model a few tools per stage and loads more on demand; `all` offers every tool on each call |
| `--timeout SECONDS` | Time limit for one LLM call |
| `-v`, `-vv` | More logging; `-vv` also records every LLM call in the run's folder ([details](docs/development.md#logs-and-traces)) |

## Known limitations

- **Failed attempts stay in the SALOME window.** With the plugin, a failed stage is not rolled
  back: its objects stay in the study, prefixed `sa_<run_id>_`, and `report.md` names them. Save
  your work first. For unattended runs, use the headless server.
- **A stage that runs past its time limit stops the run** (exit 1). In the SALOME window, click the
  menu entry again to restart the server, or kill SALOME if it froze. A headless server exits;
  start it again.
- **Some geometry operations crash SALOME** instead of failing. SHAPER `fillet_edges` refuses a
  radius at which two fillets on one face would meet; GEOM has no such guard yet.
- **The 1 000 000 element limit on meshes is approximate.** It can refuse a NETGEN mesh well below
  the limit, and miss some manual setups ([details](docs/development.md#smesh-element-limit)).

## License

GNU Lesser General Public License, version 2.1 or later: see `COPYING`.
