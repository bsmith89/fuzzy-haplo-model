c = get_config()
# Act like code inside ipython notebooks is being executed from the project root.
c.IPKernelApp.exec_lines = [
    "import os as _os",
    "_os.chdir('..')",
    "import sys as _sys",
    "_sys.path.append('./scripts')"
]
