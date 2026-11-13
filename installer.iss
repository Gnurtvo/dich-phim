[Setup]
AppName=Video AI Tool
AppVersion=1.0
DefaultDirName={autopf}\VideoAITool
DefaultGroupName=Video AI Tool
OutputDir=Output
OutputBaseFilename=VideoAITool-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\VideoAITool\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Video AI Tool"; Filename: "{app}\VideoAITool.exe"
Name: "{autodesktop}\Video AI Tool"; Filename: "{app}\VideoAITool.exe"

[Run]
Filename: "{app}\VideoAITool.exe"; Description: "Chạy Video AI Tool"; Flags: nowait postinstall skipifsilent
