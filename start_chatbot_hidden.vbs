Set shell = CreateObject("WScript.Shell")
projectPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
command = "cmd /c cd /d """ & projectPath & """ && start """" http://127.0.0.1:8001/ && python -m uvicorn app:app --host 127.0.0.1 --port 8001"
shell.Run command, 0, False
