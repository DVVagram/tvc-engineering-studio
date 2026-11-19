; Inno Setup 6 script for TVC Engineering Studio.
; Build the application first with packaging\build_windows.bat, then compile this file.
; The output is unsigned; Windows SmartScreen will warn until the binary is signed.

#define AppName "TVC Engineering Studio"
#define AppVersion "0.1.0"
#define AppExe "TVC Engineering Studio.exe"

[Setup]
AppId={{8E2E1C1A-6F64-4C55-9E77-5B6C2F0C1A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=TVC Engineering Studio contributors
AppPublisherURL=https://github.com/DVVagram/tvc-engineering-studio
AppSupportURL=https://github.com/DVVagram/tvc-engineering-studio/issues
AppUpdatesURL=https://github.com/DVVagram/tvc-engineering-studio/releases
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=TVC-Engineering-Studio-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
CloseApplications=yes
RestartApplications=no
VersionInfoCompany=TVC Engineering Studio contributors
VersionInfoDescription={#AppName} installer
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
VersionInfoVersion={#AppVersion}
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\TVC Engineering Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent; Check: NotSilentInstall

[Code]
function NotSilentInstall: Boolean;
begin
  { Inno still executes postinstall entries during an unattended install unless the
    check below is explicit, which would leave the app running after a silent setup. }
  Result := not WizardSilent;
end;
