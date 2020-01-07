SET mypath=%~dp0
CD %mypath%

SET PATH=%PATH%;%USERPROFILE%\AppData\Local\continuum\Anaconda3\
SET PATH=%PATH%;%USERPROFILE%\AppData\Local\continuum\Anaconda3\Scripts
SET PATH=%PATH%;%USERPROFILE%\AppData\Local\continuum\Anaconda3\Library\mingw-w64\bin
SET PATH=%PATH%;%USERPROFILE%\AppData\Local\continuum\Anaconda3\Library\usr\bin
SET PATH=%PATH%;%USERPROFILE%\AppData\Local\continuum\Anaconda3\Library\bin
jupyter-notebook.exe
pause
