import salome_pluginsmanager
from agent_server.gui_plugin import launch

salome_pluginsmanager.AddFunction(
    'Launch SALOME agent server',
    'Start (or restart) the SALOME-agent server in this GUI process',
    launch,
)
